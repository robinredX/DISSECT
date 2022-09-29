import os
import sys
import random
import numpy as np
import pandas as pd
import anndata as ad
from anndata import AnnData
import scanpy as sc
from configs.main_config import config
import shutil 
from tqdm import tqdm 
import json

def check_dataset(config, sparsemat=False):
    adata = sc.read(config["reference"])
    if sparsemat:
        adata.X = np.array(adata.X.todense())
    columns = adata.obs.columns
    if "Celltype" not in columns:
        sys.exit("No column Celltype found in {}. The column with celltypes should be Celltype".format(config["reference"]))

    if config["simulation_params"]["normalize"]=="cpm":
        sc.pp.normalize_total(adata, target_sum=1e6)
    print("{} cells, {} genes".format(adata.shape[0], adata.shape[1]))     

    df_X = pd.DataFrame(adata.X, columns=adata.var_names.tolist(), index=adata.obs.index.tolist())
    df_y = pd.DataFrame(adata.obs.Celltype.tolist(), columns=["Celltype"])

    return df_X, df_y, df_y.Celltype.unique().tolist()

def create_fractions(no_celltypes):
    """
    Borrowed from SCADEN (Menden et. al., 2020)
    Create random fractions
    :param no_celltypes: number of fractions to create
    :return: list of random fractions of length no_celltypes
    """
    fracs = np.random.rand(no_celltypes)
    fracs_sum = np.sum(fracs)
    fracs = np.divide(fracs, fracs_sum)
    return fracs


def create_subsample(x, y, celltypes, sparse=False, sample_size=100, reference_type="sc",
                    n_celltypes=None):
    """
    Borrowed from SCADEN (Menden et. al., 2020)
    Generate artifical bulk subsample with random fractions of celltypes
    If sparse is set to true, add random celltypes to the missing celltypes
    @param x:
    @param y:
    @param celltypes:
    @param sparse:
    @return:
    """
    available_celltypes = celltypes
    n = len(available_celltypes)
    if n_celltypes:
        n = n_celltypes
    if sparse:
        no_keep = np.random.randint(1, n)
        keep = np.random.choice(
            list(range(len(available_celltypes))), size=no_keep, replace=False
        )
        available_celltypes = [available_celltypes[i] for i in keep]

    no_avail_cts = len(available_celltypes)

    # Create fractions for available celltypes
    fracs = create_fractions(no_celltypes=no_avail_cts)
    samp_fracs = np.multiply(fracs, sample_size)
    samp_fracs = list(map(int, samp_fracs))

    # Make complete fracions
    fracs_complete = [0] * len(celltypes)
    for i, act in enumerate(available_celltypes):
        idx = celltypes.index(act)
        fracs_complete[idx] = fracs[i]

    artificial_samples = []
    for i in range(no_avail_cts):
        ct = available_celltypes[i]
        cells_sub = x.loc[np.array(y["Celltype"] == ct), :]
        if reference_type=="sc":
            cells_fraction = np.random.randint(0, cells_sub.shape[0], samp_fracs[i])
            cells_sub = cells_sub.iloc[cells_fraction, :]
        elif reference_type=="bulk":
            cells_fraction = np.random.randint(0, cells_sub.shape[0], 1)
            cells_sub = cells_sub.iloc[cells_fraction, :]*fracs_complete[ct]
        artificial_samples.append(cells_sub)

    df_samp = pd.concat(artificial_samples, axis=0)
    df_samp = df_samp.sum(axis=0)

    return df_samp, fracs_complete

def simulate(config, save_config=True, sparsemat=False):
    random.seed(42)
    np.random.seed(42)
    if not os.path.exists(config["experiment_folder"]):
        os.mkdir(config["experiment_folder"])
    else:
        sys.exit("Path {} already exists. Please choose a folder in which datasets folder doesn't exist.".format(config["experiment_folder"]))
    
    df_X, df_y, celltypes = check_dataset(config, sparsemat)

    #if copy:
    #    shutil.copyfile("configs/main_config.py", os.path.join(config["experiment_folder"], "main_config.py"))
    if save_config:
        with open(os.path.join(config["experiment_folder"], "main_config.py"), "w") as f:
            json.dump(config, f)
    n_samples = config["simulation_params"]["n_samples"]
    sparse_prop = config["simulation_params"]["sparse"]

    sparse_num = int(sparse_prop*n_samples)
    regular_num = int(n_samples - sparse_num)

    if config["simulation_params"]["unknown"]:
        unknown_celltypes =  config["simulation_params"]["unknown"]
        df_y.loc[df_y.Celltype.isin(unknown_celltypes),"Celltype"] = "unknown"
        celltypes = df_y.Celltype.unique().tolist() 

    sim_x, sim_y = [], []

    print("Simulating")
    print("Generating regular samples")
    for i in tqdm(range(regular_num)):
        if config["simulation_params"]["cells_range"]:
            cells_range = config["simulation_params"]["cells_range"].copy()
            sample_size = np.random.choice(list(range(int(cells_range[0]), 
                                                     int(cells_range[1]))))
        else:
            sample_size = 100
            print("No sample size is provided in config. Default 100 is selected.")
        if config["simulation_params"]["celltypes_range"]:
            celltypes_range = config["simulation_params"]["celltypes_range"]
            n_celltypes = np.random.choice(list(range(int(celltypes_range[0]), 
                                                     int(celltypes_range[1]))))
        else:
            n_celltypes = None
        sample, label = create_subsample(df_X, df_y, celltypes, sample_size=sample_size, n_celltypes=n_celltypes)
        sim_x.append(sample)
        sim_y.append(label)

    print("Generating sparse samples")
    for i in tqdm(range(sparse_num)):
        if config["simulation_params"]["cells_range"]:
            cells_range = config["simulation_params"]["cells_range"].copy()
            sample_size = np.random.choice(list(range(int(cells_range[0]), 
                                            int(cells_range[1]))))
        else:
            sample_size = 100
            print("No sample size is provided in config. Default 100 is selected.")
        if config["simulation_params"]["celltypes_range"]:
            celltypes_range = config["simulation_params"]["celltypes_range"]
            n_celltypes = np.random.choice(list(range(int(celltypes_range[0]), 
                                                     int(celltypes_range[1]))))
        else:
            n_celltypes = None
        sample, label = create_subsample(df_X, df_y, celltypes, sample_size=sample_size, sparse=True, n_celltypes=n_celltypes)
        sim_x.append(sample)
        sim_y.append(label)
    
    sim_x = pd.concat(sim_x, axis=1).T
    sim_y = pd.DataFrame(sim_y, columns=celltypes)

    adata = AnnData(sim_x, obs=sim_y)
    adata.uns["cell_types"] = np.array(adata.obs.columns, dtype=object)
    if config["simulation_params"]["unknown"]:
        adata.uns["unknown"] = np.array(config["simulation_params"]["unknown"], dtype=object)
    else:
        adata.uns["unknown"] = np.array(["unknown"], dtype=object) # dummy
    adata.obs["ds"] = [config["simulation_params"]["name"]]*adata.shape[0]
    
    if config["simulation_params"]["downsample"]:
        sc.pp.downsample_counts(adata, counts_per_cell=np.array(adata.X.sum(1))*config["simulation_params"]["downsample"])

    savepath = os.path.join(config["experiment_folder"], config["simulation_params"]["name"]+".h5ad")
    adata.write(savepath)

    print("Done")

import glob
def merge(batch):
    folders = os.listdir(batch)
    ls_datasets = []
    for folder in folders:
        for f in os.listdir(os.path.join(batch, folder)):
            if "h5ad" in f:
                ls_datasets.append(os.path.join(batch, folder, f))
    adata = sc.read(ls_datasets[0])
    if len(ls_datasets)>1:
        for i in range(1, len(ls_datasets)):
            adata = adata.concatenate(sc.read(ls_datasets[i]), uns_merge="same")
    adata.write(os.path.join(batch, "data.h5ad"))

if __name__ == "__main__":
    simulate(config, save_config=True)