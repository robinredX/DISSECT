import numpy as np
import scanpy as sc
import pandas as pd


def load_spatial_data(st_path="data/V1_Mouse_Brain_Sagittal_Anterior.h5ad"):
    st_data = sc.read_h5ad(st_path)
    return st_data


def load_prepared_data(experiment_dir):
    X_real = np.load(f"{experiment_dir}/datasets/X_real_test.npy")
    X_real_train = np.load(f"{experiment_dir}/datasets/X_real_train.npy")
    X_sim = np.load(f"{experiment_dir}/datasets/X_sim.npy")
    y_sim = np.load(f"{experiment_dir}/datasets/y_sim.npy")
    return X_real, X_real_train, X_sim, y_sim


def load_real_groundtruth(path=None, col_order=None):
    try:
        if col_order is None:
            y_real_df = pd.read_csv(path, sep="\t", index_col=0)
        else:
            y_real_df = pd.read_csv(path, sep="\t", index_col=0)[col_order]
        y_real = y_real_df.to_numpy()
        return y_real, y_real_df
    except Exception as e:
        print(e)
        return None


def load_sample_names(path):
    with open(path, "r") as f:
        sample_list = f.readlines()
    sample_list = [x.strip() for x in sample_list][1::]
    return sample_list


def load_celltypes(path):
    with open(path, "r") as f:
        cell_type_list = f.readlines()
    cell_type_list = [x.strip() for x in cell_type_list][1::]
    return cell_type_list

def load_gene_names(path):
    with open(path, "r") as f:
        gene_list = f.readlines()
    gene_list = [x.strip() for x in gene_list][1::]
    return gene_list
