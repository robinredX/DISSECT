from typing import Any
import io
import wandb
import pandas as pd
import PIL
import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq
import scanpy as sc
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.utils.metrics import calc_mean_corr, calc_mean_rmse


class DeconvolutionModel(pl.LightningModule):
    def __init__(
        self,
        net,
        weight_decay=0.0,
        l1_lambda=1e-5,
        l2_lambda=1e-5,
        beta=None,
        celltype_names=None,
        sample_names=None,
        spatial_data=None,
        sim_loss_fn="kl_div",
        alpha_min=0.1,
        alpha_max=0.9,
        normalize=True,
    ):
        super().__init__()
        self.net = net
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.sim_loss_fn = sim_loss_fn
        
        # only used for plotting during validation
        if spatial_data is not None:
            self.st_data = spatial_data.copy()
        else:
            self.st_data = None
        self.celltype_names = celltype_names
        self.sample_names = sample_names
        self.beta = beta
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.normalize = normalize
        
        # this accesses the init parameters of the model
        self.save_hyperparameters(
            ignore=["net", "spatial_data", "sample_names", "celltype_names"]
        )

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # https://pytorch-lightning.readthedocs.io/en/stable/guides/speed.html
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        # configure optimizer
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=1e-5, weight_decay=self.weight_decay
        )
        # TODO
        # configure lr scheduler
        # scheduler = ReduceLROnPlateau(optimizer, ...)
        return [optimizer]  # , [scheduler]

    def forward(self, graph):
        # maybe make this normalization optional and instead do it in the dataset
        if self.normalize:
            graph.x = normalize_per_sample(graph.x, with_pos=False)
        y_hat = self.net(
            graph.x,
            graph.edge_index,
            graph.edge_weight,
            graph.edge_attr,
            pos=graph.pos,
            batch=graph.batch,
        )
        return y_hat

    # by default runs the forward method
    def predict_step(self, batch, idx):
        # select real graph
        return self(batch[0])

    def training_step(self, train_batch, batch_idx):
        g_real, g_sim = train_batch

        # create mixture graph on the fly
        # maybe increase amount of real data over time per node
        alpha = alpha_scheduler(self.global_step, self.alpha_min, self.alpha_max)
        
        # TODO refine mixture graph definition
        g_mix = Data(
            x=alpha * g_real.x + (1 - alpha) * g_sim.x,
            edge_index=g_real.edge_index,
            edge_weight=g_real.edge_weight,
            edge_attr=g_real.edge_attr,
            pos=g_real.pos,
            batch=g_real.batch,
        )

        # simulated ground truth celltype abundances
        y_sim = g_sim.y

        # change loss function based on global step
        # should be done in a callback
        if self.beta is None:
            beta = beta_scheduler(self.global_step)
        else:
            beta = self.beta

        # forward pass
        # TODO check whether we can put everything into one batch
        y_hat_real = self(g_real)
        y_hat_sim = self(g_sim)
        y_hat_mix = self(g_mix)


        sim_loss, mix_loss = calc_loss(
            y_sim, y_hat_sim, y_hat_real, y_hat_mix, alpha, sim_loss_fn=self.sim_loss_fn
        )

        total_loss = sim_loss + beta * mix_loss
        # optionally add l1 and l2 regularization
        l1_loss = ln_loss(self.net, n=1, ln_lambda=self.l1_lambda, only_matrices=True)
        self.log("train/l1_loss", l1_loss)
        total_loss += l1_loss

        l2_loss = ln_loss(self.net, n=2, ln_lambda=self.l2_lambda, only_matrices=True)
        self.log("train/l2_loss", l2_loss)
        total_loss += l2_loss

        # log losses
        self.log("train/total_loss", total_loss)
        self.log("train/sim_loss", sim_loss)
        self.log("train/mix_loss", mix_loss)
        self.log("train/beta", beta)

        return total_loss

    def validation_step(self, val_batch, batch_idx):
        # TODO put into callback
        data = val_batch[0]
        y_hat = self(data)
        y = data.y
        if self.st_data is not None and self.celltype_names is not None:
            celltype_indices = np.array(np.argmax(y_hat.cpu().detach().numpy(), axis=1))
            # map celltypes onto celltype list
            pred_celltypes = [self.celltype_names[i] for i in celltype_indices]
            # add new column to adata
            self.st_data.obs["celltype"] = pred_celltypes
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            sc.pl.spatial(self.st_data, color=["celltype"], ax=ax, show=False)
            pil_img = buffer_plot_and_get(fig)
            plt.close(fig)
            self.logger.log_image(
                key="validation",
                images=[wandb.Image(pil_img)],
                caption=[f"Current epoch: {self.current_epoch}"],
            )
            del self.st_data.uns["celltype_colors"]

        # check if we have ground truth
        if y is not None:
            mean_rmse, mean_corr = compare_with_gt(
                y_hat.cpu().detach().numpy(), y.cpu().detach().numpy()
            )
            self.log("validation/mean_rmse", mean_rmse)
            self.log("validation/mean_corr", mean_corr)
        # in any case save predictions
        y_hat_df = pd.DataFrame(y_hat.cpu().detach().numpy())
        if self.celltype_names is not None:
            y_hat_df.columns = self.celltype_names
        if self.sample_names is not None:
            y_hat_df["sample_names"] = self.sample_names
        self.logger.log_table(
            f"predictions-step-{self.global_step}", dataframe=y_hat_df
        )
        return y_hat


def compare_with_gt(y_hat, y):
    # y_hat:.shape (n_samples, n_celltypes)
    # average rmse cell type wise
    mean_rmse = calc_mean_rmse(y_hat, y)[0]
    mean_corr = calc_mean_corr(y_hat, y)[0]
    return mean_rmse, mean_corr


def calc_loss(y_sim, y_hat_sim, y_hat_real, y_hat_mix, alpha, sim_loss_fn="kl_div"):
    # compute mixture ground truth
    y_mix = alpha * y_hat_real + (1 - alpha) * y_hat_sim
    # calc kl divergence
    # requires log probabilities for the predicted input
    if sim_loss_fn == "kl_div":
        sim_loss = F.kl_div(
            torch.log(y_hat_sim),
            y_sim,
            reduction="batchmean",
        )
    elif sim_loss_fn == "mse":
        sim_loss = F.mse_loss(y_hat_sim, y_sim)
    elif sim_loss_fn == "cross_entropy":
        sim_loss = F.cross_entropy(y_hat_sim, y_sim)
    elif sim_loss_fn == "js_div":
        raise NotImplementedError("JS divergence not implemented")
    else:
        raise ValueError(f"sim_loss_fn {sim_loss_fn} not implemented")
    # calc mse
    mix_loss = F.mse_loss(y_mix, y_hat_mix)
    return sim_loss, mix_loss


def normalize_per_sample(x, eps=1e-8, with_pos=False):
    # Computes log1p-MinMax on gene expression features, sample-wise
    if with_pos:
        pos = x[:, -2::]
        x = x[:, :-2]
    # log to the basis 2
    x = torch.log1p(x) / np.log(2)

    min_val = torch.min(x, dim=1, keepdim=True)[0].expand(-1, x.shape[1])
    max_val = torch.max(x, dim=1, keepdim=True)[0].expand(-1, x.shape[1])
    # min_val.shape = (batch_size*nodes, genes)

    x_normed = (x - min_val) / (max_val - min_val + eps)
    if with_pos:
        x_normed = torch.concat([x_normed, pos], dim=-1)
    return x_normed


def alpha_scheduler(step, min_val=0.1, max_val=0.9, max_steps=2000):
    # alpha = min(0.8, 0.8 * (step / max_steps)) + min_val
    alpha = np.random.uniform(min_val, max_val)
    return alpha


def beta_scheduler(
    step, phase_1=0.4, phase_2=0.8, phase_1_val=15, phase_2_val=10, max_steps=5000
):
    if step < phase_1 * max_steps:
        return 0.0
    elif (step >= phase_1 * max_steps) and (step < phase_2 * max_steps):
        return phase_1_val
    else:
        return phase_2_val


def buffer_plot_and_get(fig):
    buff = io.BytesIO()
    fig.savefig(buff, bbox_inches="tight")
    buff.seek(0)
    return PIL.Image.open(buff)


def convert_fig_to_array(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im


def ln_loss(model, n=1, ln_lambda=1e-5, only_matrices=True):
    ln_reg = 0.0
    for w in model.parameters():
        if len(w.shape) == 1 and only_matrices:
            continue
        else:
            ln_reg += torch.sum(torch.pow(torch.abs(w), n))
    return ln_lambda * ln_reg