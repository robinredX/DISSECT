from typing import Any
import io
import wandb
import PIL
import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class DeconvolutionModel(pl.LightningModule):
    def __init__(
        self,
        net,
        node_positions=True,
        spatial_data=None,
        celltype_list=None,
        weight_decay=0.0,
        l1_lambda=1e-5,
        l2_lambda=1e-5,
    ):
        super().__init__()
        self.net = net
        self.node_positions = node_positions
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        # this accesses the init parameters of the model
        self.save_hyperparameters(ignore=["net", "spatial_data", "celltype_list"])
        # self.alpha = 0.5
        # self.lambda_ = 0.5
        self.st_data = spatial_data.copy()
        self.celltype_list = celltype_list

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
        graph.x = normalize_per_sample(graph.x, with_pos=False)
        graph.x = torch.cat([graph.x, graph.pos], dim=1)
        y_hat = self.net(graph.x, graph.edge_index, graph.edge_weight)
        return y_hat

    # by default runs the forward method
    def predict_step(self, batch, idx):
        # select real graph
        return self(batch[0])

    def training_step(self, train_batch, batch_idx):
        g_real, g_sim = train_batch

        # create mixture graph on the fly
        # maybe increase amount of real data over time per node
        alpha = alpha_scheduler(self.global_step)
        g_mix = Data(
            x=alpha * g_real.x + (1 - alpha) * g_sim.x,
            pos=g_real.pos,
        )

        # simulated ground truth celltype abundances
        y_sim = g_sim.y

        # forward pass
        y_hat_real = self(g_real)
        y_hat_sim = self(g_sim)
        y_hat_mix = self(g_mix)

        # change loss function based on global step
        # should be done in a callback
        beta = beta_scheduler(self.global_step)

        sim_loss, mix_loss = calc_loss(y_sim, y_hat_sim, y_hat_real, y_hat_mix, alpha)

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

        return total_loss

    def validation_step(self, val_batch, batch_idx):
        # TODO put into callback
        if self.st_data is not None and self.celltype_list is not None:
            result = self(val_batch[0])
            cell_type_indices = np.array(np.argmax(result.cpu().numpy(), axis=1))
            # map celltypes onto cdelltype list
            cell_types = [self.celltype_list[i] for i in cell_type_indices]
            # add new column to adata
            self.st_data.obs["celltype"] = cell_types
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            sq.pl.spatial_scatter(self.st_data, color=["celltype"], ax=ax)
            pil_img = buffer_plot_and_get(fig)
            plt.close(fig)
            self.logger.log_image(
                key="predictions",
                images=[wandb.Image(pil_img)],
                caption=[f"Current epoch: {self.current_epoch}"],
            )
            # self.logger.log_table(key="prediction", columns=log_columns, data=log_data)
            del self.st_data.uns["celltype_colors"]
            return result
        else:
            return None


def calc_loss(y_sim, y_hat_sim, y_hat_real, y_hat_mix, alpha):
    # compute mixture ground truth
    y_mix = alpha * y_hat_real + (1 - alpha) * y_hat_sim
    # calc kl divergence
    # requires log probabilities for the predicted input
    sim_loss = F.kl_div(
        torch.log(y_hat_sim),
        y_sim,
        reduction="batchmean",
    )
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


def beta_scheduler(step):
    max_steps = 5000
    if step < 0.4 * max_steps:
        return 0.0
    elif (step >= 0.4 * max_steps) and (step < 0.8 * max_steps):
        return 15
    else:
        return 10


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
