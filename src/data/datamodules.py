from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.loader import DataLoader
import scanpy as sc
import pandas as pd

from src.data.datasets import get_spatial_train_and_test_set
from src.data.utils import (
    load_prepared_data,
    load_spatial_data,
    load_celltypes,
    load_sample_names,
    load_gene_names,
)


class SpatialDataModule(LightningDataModule):
    def __init__(
        self,
        st_path: str,
        reference_dir: str,
        radius: float = 0.02,
        p: float = 0.0,
        num_samples: int = 32,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        log_hparams: bool = True,
        use_rctd_genes: bool = False,
        with_platform_effects: bool = True,
        renormalize: bool = False,
    ) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=log_hparams)

        self.st_data = load_spatial_data(self.hparams.st_path)
        self.X_real, self.X_real_train, self.X_sim, self.y_sim = load_prepared_data(
            self.hparams.reference_dir
        )
        # ground truth only available for slideseq, seqfish, merfish, starmap data
        techniques = ["slide", "FISH", "star"]
        if any([tech in self.hparams.st_path for tech in techniques]):
            self.y_real_celltypes = list(self.st_data.obs.columns[2::])
            self.y_real = self.st_data.obs[self.y_real_celltypes].to_numpy()
        else:
            self.y_real_celltypes = None
            self.y_real = None

        self.celltype_names = load_celltypes(f"{reference_dir}/datasets/celltypes.txt")

        # load sample names
        self.sample_names = load_sample_names(
            f"{reference_dir}/datasets/sample_names.txt"
        )

        self.gene_names = load_gene_names(f"{reference_dir}/datasets/genes.txt")

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None

        self.postprocess()

    @property
    def num_celltypes(self) -> int:
        return self.y_sim.shape[1]

    @property
    def num_spots(self) -> int:
        return self.X_real.shape[0]

    @property
    def num_genes(self) -> int:
        return self.X_real.shape[1]

    def postprocess(self) -> None:
        # select rctd genes if available
        base_index = self.hparams.st_path.rfind("data")
        base_path = self.hparams.st_path[0:base_index - 1]
        if self.hparams.use_rctd_genes:
            if not self.hparams.with_platform_effects:
                rctd_gene_path = f"{base_path}/data/extras/rctd_genelists/de_genes_no_platformEffect.csv"
            elif "-2a" in self.hparams.reference_dir:
                rctd_gene_path = f"{base_path}/data/extras/rctd_genelists/de_genes_UMOD-WT.WT-2a_resolution75.csv"
               
            elif "-4b" in self.hparams.reference_dir or "v2_105" in self.hparams.reference_dir:
                rctd_gene_path = f"{base_path}/data/extras/rctd_genelists/de_genes_UMOD-KI.KI-4b_resolution105.csv"
            else:
                rctd_gene_path = None

            if rctd_gene_path is not None:
                rctd_gene_names = pd.read_csv(rctd_gene_path)["Gene"].to_list()
                # intersect genes with rctd genes
                gene_intersection = list(
                    set(self.gene_names).intersection(set(rctd_gene_names))
                )
                X_real_df = pd.DataFrame(self.X_real, columns=self.gene_names)
                X_sim_df = pd.DataFrame(self.X_sim, columns=self.gene_names)

                self.X_real = X_real_df[gene_intersection].to_numpy()
                self.X_sim = X_sim_df[gene_intersection].to_numpy()

                print(
                    f"Using {len(gene_intersection)} rctd genes instead of {len(self.gene_names)} original genes for training."
                )

                self.gene_names = gene_intersection

                # potentially perform renormalization
                if self.hparams.renormalize:
                    # normalize to counts per million
                    pass

    def prepare_data(self) -> None:
        # potentially run data dissect data preparation here
        # should not be used to set variables like self.train_data
        pass

    def setup(self, stage: str = None) -> None:
        """Load data. Set variables: `self.train_data`, `self.val_data`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load data only of not done already
        if not self.train_data and not self.val_data:
            self.train_data, self.val_data = get_spatial_train_and_test_set(
                self.st_data,
                self.X_real,
                self.X_sim,
                self.y_sim,
                self.hparams.radius,
                p=self.hparams.p,
                y_real=self.y_real,
                num_samples=self.hparams.num_samples,
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
