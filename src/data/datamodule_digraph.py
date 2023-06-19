from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.loader import DataLoader
import scanpy as sc
import pandas as pd

from src.data.datasets import DiSpatialGraphDataset
from src.data.utils import (
    load_prepared_data,
    load_spatial_data,
    load_celltypes,
    load_sample_names,
    load_gene_names,
    sparse_to_array,
    filter_genes,
    make_var_names_unique,
)


class DiSpatialDataModule(LightningDataModule):
    def __init__(
        self,
        st_path_real: str,
        st_path_sim: str,
        radius: float = 0.02,
        target_sum: float = None,
        var_cutoff: float = 0.0,
        duplicate_aggregation: str = "first",
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        log_hparams: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=log_hparams)
        self.target_sum = target_sum
        self.var_cutoff = var_cutoff
        self.duplicate_aggregation = duplicate_aggregation

        self.st_data = sc.read_h5ad(st_path_real)
        self.st_data_sim = sc.read_h5ad(st_path_sim)
        self.preprocess()

        # standard formatting for simulated data
        self.y_sim_celltypes = list(self.st_data_sim.obs.columns[0::])
        self.y_sim = self.st_data_sim.obs[self.y_sim_celltypes].to_numpy()

        # ground truth only available for slideseq, seqfish, merfish, starmap data
        techniques = ["slide", "FISH", "star"]
        if any([tech in self.hparams.st_path_real for tech in techniques]):
            self.y_real_celltypes = list(self.st_data.obs.columns[2::])
            assert set(self.y_sim_celltypes) == set(
                self.y_real_celltypes
            ), "Celltypes do not match between real and simulated data"
            self.y_real = self.st_data.obs[self.y_sim_celltypes].to_numpy()
        else:
            self.y_real_celltypes = None
            self.y_real = None

        self.sample_names = list(self.st_data.obs.index)
        self.celltype_names = self.y_sim_celltypes
        self.gene_names = list(self.st_data.var.index)
        self.num_celltypes = len(self.celltype_names)
        self.num_genes = len(self.gene_names)
        self.num_spots = len(self.sample_names)

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None

    def preprocess(self) -> None:
        # optionally set specific parameters based on considered dataset
        techniques = ["seqFISH", "MERFISH"]
        if any([tech in self.hparams.st_path_real for tech in techniques]):
            self.target_sum = None
            self.var_cutoff = 0.0

        self.st_data = make_var_names_unique(
            self.st_data, aggregation=self.duplicate_aggregation
        )
        self.st_data_sim = make_var_names_unique(
            self.st_data_sim, aggregation=self.duplicate_aggregation
        )

        # convert to dense matrix
        self.st_data.X = sparse_to_array(self.st_data.X)
        self.st_data_sim.X = sparse_to_array(self.st_data_sim.X)

        # apply variance cutoff
        print(f"Removing genes with variance below {self.var_cutoff}")
        print(f"Genes in real data before filtering: {self.st_data.n_vars}")
        print(f"Genes in simulated data before filtering: {self.st_data_sim.n_vars}")
        
        self.st_data = filter_genes(self.st_data, self.var_cutoff)
        self.st_data_sim = filter_genes(self.st_data_sim, self.var_cutoff)
        
        print(f"Genes in real data after filtering: {self.st_data.n_vars}")
        print(f"Genes in simulated data after filtering: {self.st_data_sim.n_vars}")

        # make sure genes intersect        
        gene_intersection = list(
            set(self.st_data.var_names).intersection(
                set(self.st_data_sim.var_names)
            )
        )
        print(f"There are {len(gene_intersection)} common genes in both datasets.")

        self.st_data = self.st_data[:, gene_intersection]
        self.st_data_sim = self.st_data_sim[:, gene_intersection]

        # perform DISSECT processing on data
        if self.target_sum is not None:
            sc.pp.normalize_total(self.st_data, target_sum=self.target_sum)
            sc.pp.normalize_total(self.st_data_sim, target_sum=self.target_sum)

    def prepare_data(self) -> None:
        # should not be used to set variables like self.train_data
        pass

    def setup(self, stage: str = None) -> None:
        """Load data. Set variables: `self.train_data`, `self.val_data`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load data only of not done already
        if not self.train_data and not self.val_data:
            self.train_data = DiSpatialGraphDataset(
                self.st_data,
                self.st_data_sim,
                y_real=self.y_real,
                y_sim=self.y_sim,
                radius=self.hparams.radius,
                test=False,
            )
            self.val_data = DiSpatialGraphDataset(
                self.st_data,
                self.st_data_sim,
                y_real=self.y_real,
                y_sim=self.y_sim,
                radius=self.hparams.radius,
                test=True,
            )
            # maybe later also include separate test set
            # when applying the method to separate test set

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def check_data(self):
        # initialize internal datasets
        self.setup()
        val_loader = self.val_dataloader()
        train_loader = self.train_dataloader()
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        print(f"Elements in train batch: {len(train_batch)}")
        print("Train batch element 0:", train_batch[0])
        print("Train batch element 1:", train_batch[1])
        print(f"Elements in val batch: {len(val_batch)}")
        print("Val batch element 0:", val_batch[0])
        return train_batch, val_batch

    def move_to_device(self, device):
        self.train_data.move_to_device(device)
        self.val_data.move_to_device(device)
