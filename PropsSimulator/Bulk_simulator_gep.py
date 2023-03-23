import os
import sys
import random
import numpy as np
import pandas as pd
import anndata as ad
from anndata import AnnData
import scanpy as sc
from main_config import config
import shutil 
from tqdm import tqdm 
import json
import matplotlib.pyplot as plt

class Simulate(object):
    def __init__(self, config):
        self.config = config
        if config["scdata_type"]=="txt":
            self.sc_adata = pd.read_table(config["scdata"], index_col=0)
            self.sc_adata = AnnData(self.sc_adata)
            self.sc_adata.obs[config["celltype_col"]] = self.sc_adata.obs.index.tolist()
        else:
            self.sc_adata = sc.read(config["scdata"])
        if "sparse" in str(type(self.sc_adata.X)):
            self.sc_adata.X = np.array(self.sc_adata.X.todense())
        self.celltypes = np.sort(np.array(self.sc_adata.obs[config["celltype_col"]].unique()))
        self.n_celltypes = len(self.celltypes)

    def generate_props(self):
        if not self.config["n_samples"]:
            self.config["n_samples"] = 1000*self.n_celltypes
        if not self.config["cells_per_sample"]:
            self.config["cells_per_sample"] = 1000
        self.n_sparse = int(self.config["n_samples"]*self.config["prop_sparse"])
        self.n_complete = self.config["n_samples"] - self.n_sparse

        ##### Complete
        if not self.config["concentration"]:
            self.config["concentration"] = np.ones(self.n_celltypes)
        self.props_complete = np.random.dirichlet(self.config["concentration"], self.n_complete).astype(np.float32)
        self.min_prc = 1/self.config["cells_per_sample"]
        self.props_complete[self.props_complete<self.min_prc] = 0
        # Re-normalize
        self.props_complete = (self.props_complete.T/self.props_complete.sum(axis=1)).T
        self.cells_complete = self.props_complete*self.config["cells_per_sample"]
        self.cells_complete = np.round(self.cells_complete,0).astype(int)
        # Update props to maintain summing to 1
        self.props_complete = (self.cells_complete.T/self.cells_complete.sum(axis=1)).T.astype(np.float32)
    
        ##### Sparse
        keep_sparse = np.ones((self.n_sparse, self.n_celltypes), dtype=np.float32)
        
        no_keep = np.random.randint(1, self.n_celltypes, size=self.n_sparse)
        no_keeps = []
        for i in no_keep:
            no_keeps.append(np.random.choice(
                    list(range(self.n_celltypes)), size=i, replace=False
                ))
            
        for i in range(len(no_keep)):
            keep_sparse[i, no_keeps[i]] = 1e-6
            
        self.props_sparse = np.zeros((self.n_sparse, self.n_celltypes), dtype=np.float32)
        for i in range(self.n_sparse):
            self.props_sparse[i,:] = np.random.dirichlet(keep_sparse[i], 1)[0]

        self.props_sparse[self.props_sparse<self.min_prc] = 0
        self.props_sparse = (self.props_sparse.T/self.props_sparse.sum(axis=1)).T

        self.cells_sparse = self.props_sparse*self.config["cells_per_sample"]
        self.cells_sparse = np.round(self.cells_sparse,0).astype(int)
        self.props_sparse = (self.cells_sparse.T/self.cells_sparse.sum(axis=1)).T.astype(np.float32)

        if not os.path.exists(config["experiment_folder"]):
            os.mkdir(config["experiment_folder"])
        else:
            from datetime import datetime
            now = datetime.now()
            date_time = now.strftime("%m.%d.%Y_%H.%M.%S")
            name = "experiment_" + date_time
            print("Specified experiment_folder {} already exists. Creating a new folder with name {}.".format(config["experiment_folder"], name))
            config["experiment_folder"] = name
            os.mkdir(config["experiment_folder"])
        fig = plt.figure()
        ax=plt.boxplot(self.props_complete, labels=self.celltypes) #
        plt.ylabel("Proportion")
        plt.title("Proportions of cell-types in generated samples")
        plt.savefig(os.path.join(config["experiment_folder"], "boxplot_props_complete.pdf"))
        fig = plt.figure()
        ax=plt.boxplot(self.cells_complete, labels=self.celltypes)
        plt.ylabel("Count")
        plt.title("Counts of cell-types in generated samples")
        plt.savefig(os.path.join(config["experiment_folder"], "boxplot_ncells_complete.pdf"))

        fig = plt.figure()
        ax=plt.boxplot(self.props_sparse, labels=self.celltypes) #
        plt.ylabel("Proportion")
        plt.title("Proportions of cell-types in generated samples")
        plt.savefig(os.path.join(config["experiment_folder"], "boxplot_props_sparse.pdf"))       
        fig = plt.figure()
        ax=plt.boxplot(self.cells_sparse, labels=self.celltypes)
        plt.ylabel("Count")
        plt.title("Counts of cell-types in generated samples")
        plt.savefig(os.path.join(config["experiment_folder"], "boxplot_ncells_sparse.pdf"))

        self.props = np.concatenate([self.props_complete, self.props_sparse], axis=0)
        self.cells = np.concatenate([self.cells_complete, self.cells_sparse], axis=0)

    def preprocess(self):
        self.sc_adata.var_names_make_unique() 
        sc.pp.filter_cells(self.sc_adata, min_genes=self.config["filter"]["min_genes"])
        sc.pp.filter_genes(self.sc_adata, min_cells=self.config["filter"]["min_cells"])     

        self.sc_adata.var['mt'] = self.sc_adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(self.sc_adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True) 
        self.sc_adata = self.sc_adata[self.sc_adata.obs.pct_counts_mt < self.config["filter"]["mt_cutoff"], :]
        sc.pp.normalize_per_cell(self.sc_adata)

        tmp = self.sc_adata.copy()
        sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp)
        heg = tmp.var[tmp.var.means > self.config["filter"]["min_expr"]].index
        self.sc_adata = self.sc_adata[:, heg]
        del tmp
        del heg

    def simulate(self):
        adatas = {}
        genes = self.sc_adata.var_names
        for celltype in self.celltypes:
            adatas[celltype] = np.array(self.sc_adata[self.sc_adata.obs[self.config["celltype_col"]]==celltype].X)
        self.sc_adata = None

        adata = AnnData(np.zeros((self.config["n_samples"], len(genes))),
                            var=pd.DataFrame(index=genes),
                            obs=pd.DataFrame(self.props, columns=self.celltypes))
        
        for i in tqdm(range(self.props.shape[0])):
            for j in range(self.n_celltypes):            
                sample = adatas[self.celltypes[j]]*self.props[i][j]
                if j==0:
                    total_sample = sample
                else:
                    total_sample = total_sample + sample
            adata.X[i,:] = total_sample

        adata.write(os.path.join(config["experiment_folder"], "simulated.h5ad"))

def save_dict_to_file(dic):
    f = open(os.path.join(config["experiment_folder"], 'simulation_config.py'),'w')
    f.write("config = ")
    f.write(str(dic))
    f.close()

s = 42
random.seed(s)
np.random.seed(s)

sim=Simulate(config)
if config["preprocess"]:
    sim.preprocess()
sim.generate_props()
batch_col = sim.config["batch_col"]
columns = sim.sc_adata.obs.columns

sim.simulate()
save_dict_to_file(config)



