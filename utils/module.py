import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from utils.gnn import CelltypeDeconvolver


class DeconvolutionModel(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.save_hyperparameters(ignore=['net'])

    def forward(self, batch):
        return self.predict_step(batch)

    def predict_step(self, batch, idx):
        g_real = batch[0]
        X_real = normalize_per_batch(g_real.x)

        y_hat_real = self.net(X_real, g_real.edge_index, g_real.edge_weight)
        return y_hat_real

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        g_real, g_sim, g_mix = train_batch
        X_real, X_sim, X_mix = g_real.x, g_sim.x, g_mix.x

        # maybe combine position and gene expression here
        
        # maybe create mixture graph on the fly
        # maybe increase amount of real data over time per node


        # potentially do some data processing here
        # or maybe do it in the dataloader
        X_real = normalize_per_batch(X_real)
        X_sim = normalize_per_batch(X_sim)
        X_mix = normalize_per_batch(X_mix)

        # simulated ground truth celltype abundances
        y_sim = g_sim.y

        y_hat_real = self.net(X_real, g_real.edge_index, g_real.edge_weight)
        y_hat_sim = self.net(X_sim, g_sim.edge_index)
        y_hat_mix = self.net(X_mix, g_mix.edge_index, g_mix.edge_weight)

        # change loss function based on global step
        # should be done in a callback
        beta = beta_scheduler(self.global_step)
        reg_loss, cons_loss = calc_loss(y_sim, y_hat_sim, y_hat_real, y_hat_mix)
        loss = reg_loss + beta * cons_loss
        self.log("train/loss", loss)
        return loss


def alpha_scheduler(step):
    return min(0.9, 0.9 * (step / 1000))


def beta_scheduler(step):
    max_epochs = 100
    if step < 0.4*max_epochs:
        return 0.0
    elif (step >= 0.4*max_epochs) and (step < 0.8*max_epochs):
        return 0.15
    else:
        return 0.1


def calc_loss(y_sim, y_hat_sim, y_hat_real, y_hat_mix, alpha=0.5):
    # compute mixture ground truth
    y_mix = alpha * y_hat_real + (1 - alpha) * y_hat_sim
    # calc kl divergence
    reg_loss = F.kl_div(
        torch.log(y_hat_sim),
        y_sim,

    )
    # calc mse
    cons_loss = F.mse_loss(y_mix, y_hat_mix)
    return reg_loss, cons_loss


def normalize_per_batch(x, eps=1e-8):
    # Computes log1p-MinMax on gene expression features
    pos = x[:, -2::]
    x = x[:, :-2]
    x = torch.log1p(x)
    min_val = torch.min(x, dim=1, keepdim=True)[0].expand(-1, x.shape[1])
    max_val = torch.max(x, dim=1, keepdim=True)[0].expand(-1, x.shape[1])
    # min_val.shape = (batch_size*nodes, genes)
    x_normed = (x - min_val) / (max_val - min_val + eps)
    x_normed = torch.concat([x_normed, pos], dim=-1)
    return x_normed


class CustomCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        print("hello world")
