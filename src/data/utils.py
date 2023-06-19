import numpy as np
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
import anndata as ad


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


def add_path_to_paths(path, paths):
    updated_paths = [f"{path}/{p}" for p in paths]
    return updated_paths


def get_sc_and_st_paths(base_path, sc_data_files, st_data_files, data_dir="data"):
    sc_paths = [f"{base_path}/{data_dir}/{file}" for file in sc_data_files]
    st_paths = [f"{base_path}/{data_dir}/{file}" for file in st_data_files]
    return sc_paths, st_paths


def get_paths_for_processing(
    base_path,
    st_data_files,
    sc_data_files,
    celltype_cols,
    extra_settings_sim,
    extra_settings_prep,
    data_dir="data",
    use_old_experiment_paths=False,
    experiment_dirs=None,
):
    sc_paths, st_paths = get_sc_and_st_paths(
        base_path, sc_data_files, st_data_files, data_dir
    )
    if base_path is None:
        base_path = ""

    simulation_paths = [
        f"{base_path}/data/simulated/" + "/".join(path.split("/")[-2::]).split(".")[0]
        for path in sc_paths
    ]
    simulation_paths = [
        f"{path}_{celltype_cols[i]}" for i, path in enumerate(simulation_paths)
    ]
    # add extra settings if not none
    simulation_paths = [
        f"{path}+{setting}" if setting is not None else path
        for path, setting in zip(simulation_paths, extra_settings_sim)
    ]

    experiment_paths = [
        sc_path.split("/")[-2] + "/" + st_path.split("/")[-1].split(".h5ad")[0]
        for sc_path, st_path in zip(sc_paths, st_paths)
    ]
    experiment_paths = [
        f"{base_path}/experiments/experiment_{path}" for path in experiment_paths
    ]
    # add simulation folder name
    experiment_paths = [
        f"{path}+{simulation_path.split('/')[-1]}"
        for path, simulation_path in zip(experiment_paths, simulation_paths)
    ]
    # add extra settings if not none
    experiment_paths = [
        f"{path}+{setting}" if setting is not None else path
        for path, setting in zip(experiment_paths, extra_settings_prep)
    ]

    if use_old_experiment_paths:
        if experiment_dirs is None:
            print("No experiment_dirs provided, using default")
        else:
            for i, path in enumerate(experiment_dirs):
                experiment_paths[i] = f"{base_path}/{path}"

    assert len(sc_paths) == len(st_paths) == len(celltype_cols), print(
        f"len(sc_paths) = {len(sc_paths)}, len(st_paths) = {len(st_paths)}, len(celltype_cols) = {len(celltype_cols)}"
    )
    return sc_paths, st_paths, simulation_paths, experiment_paths


def get_paths_for_training(
    st_data_files,
    sc_data_files,
    celltype_cols,
    extra_settings_sim,
    extra_settings_prep,
    use_old_experiment_paths=False,
    experiment_dirs=None,
    **kwargs,
):
    sc_paths, st_paths, simulation_paths, experiment_paths = get_paths_for_processing(
        base_path=None,
        st_data_files=st_data_files,
        sc_data_files=sc_data_files,
        celltype_cols=celltype_cols,
        extra_settings_sim=extra_settings_sim,
        extra_settings_prep=extra_settings_prep,
        use_old_experiment_paths=use_old_experiment_paths,
        experiment_dirs=experiment_dirs,
        **kwargs,
    )
    experiment_paths = [path[1::] for path in experiment_paths]
    st_paths = ["${paths.data_dir}" + f for f in st_data_files]

    if use_old_experiment_paths and experiment_dirs is None:
        print("No experiment_dirs provided, using default")

    if use_old_experiment_paths and experiment_dirs is not None:
        experiment_paths_slice = [
            "${paths.root_dir}" + "/" + path
            for path in experiment_paths[len(experiment_dirs) : :]
        ]
        experiment_paths = [
            "${paths.root_dir}" + "/" + dir for dir in experiment_dirs
        ] + experiment_paths_slice
    else:
        experiment_paths = [
            "${paths.root_dir}" + "/" + path for path in experiment_paths
        ]
    return st_paths, experiment_paths


def get_dataset_map(experiment_paths, dataset_names):
    dataset_map = {
        name: path.split("experiments")[-1]
        for name, path in zip(dataset_names, experiment_paths)
    }
    return dataset_map


def get_dataset_path_map(st_paths, dataset_names):
    dataset_path_map = {name: path for name, path in zip(dataset_names, st_paths)}
    return dataset_path_map


def sparse_to_array(X):
    if isinstance(X, csr_matrix) or isinstance(X, csc_matrix):
        return X.toarray()
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise ValueError("X must be either a numpy array or a sparse matrix")


def filter_genes(adata, min_var=0.0):
    X_df = pd.DataFrame(
        adata.X, index=adata.obs.index.tolist(), columns=adata.var_names.tolist()
    ).T
    genes_to_keep = X_df.var(axis=1) > min_var
    adata = adata[:, genes_to_keep]
    return adata


def make_var_names_unique(adata, join: str = "-", aggregation: str = None):
    if aggregation is not None:
        X_df = pd.DataFrame(
            adata.X, index=adata.obs.index.tolist(), columns=adata.var_names.tolist()
        )
        # support different aggregations on duplicate var names
        if aggregation == "first":
            adata = adata[:, ~X_df.columns.duplicated(keep="first")]
        elif aggregation == "last":
            adata = adata[:, ~X_df.columns.duplicated(keep="last")]
        elif aggregation == "mean":
            X_df = X_df.groupby(X_df.columns, axis=1).mean()
            adata = adata[:, X_df.columns]
            adata.X = X_df.values
        elif aggregation == "sum":
            X_df = X_df.groupby(X_df.columns, axis=1).sum()
            adata = adata[:, X_df.columns]
            adata.X = X_df.values
    else:
        adata.var_names_make_unique(join=join)
    return adata
