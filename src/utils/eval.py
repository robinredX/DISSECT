import pandas as pd
import scanpy as sc

from src.utils.metrics import calc_metrics_df


def create_metrics_df(
    mean_corrs, mean_corrs_sample, mean_rmses, mean_rmses_sample, method="DISSECT"
):
    try:
        length = len(mean_corrs)
    except:
        length = 1
    return pd.DataFrame(
        {
            "Correlation": mean_corrs,
            "Samplewise correlation": mean_corrs_sample,
            "RMSE": mean_rmses,
            "Samplewise RMSE": mean_rmses_sample,
            "Method": [method] * length,
        }
    )


def compare_methods(dissect_results, gnn_results: dict, y_real, verbose=0):
    dissect_metrics = {}
    for k, result in enumerate(dissect_results):
        dissect_metrics[f"DISSECT-{k}"] = calc_metrics_df(
            result, y_real, verbose=verbose, exclude_cols=None
        )

    # put all results into one dataframe
    dissect_results_dfs = []
    for key, dissect_metric in dissect_metrics.items():
        dissect_results_dfs.append(
            create_metrics_df(
                dissect_metric["mean_corr"],
                dissect_metric["mean_corr_sample"],
                dissect_metric["mean_rmse"],
                dissect_metric["mean_rmse_sample"],
                method=key.split("-")[0],
            )
        )
    if gnn_results is not None:
        gnn_metrics = {}
        for key, gnn_result in gnn_results.items():
            gnn_metrics[key] = calc_metrics_df(
                gnn_result, y_real, verbose=verbose, exclude_cols=None
            )
        gnn_results_dfs = []
        for key, gnn_metric in gnn_metrics.items():
            gnn_results_dfs.append(
                create_metrics_df(
                    gnn_metric["mean_corr"],
                    gnn_metric["mean_corr_sample"],
                    gnn_metric["mean_rmse"],
                    gnn_metric["mean_rmse_sample"],
                    method=key,
                )
            )
    else:
        gnn_results_dfs = []
    comparison_df = pd.concat(
        [*dissect_results_dfs, *gnn_results_dfs], axis=0, ignore_index=True
    )
    return comparison_df


def compare_methods_new(
    results: list,
    ground_truth: pd.DataFrame,
    methods=None,
    samplewise=False,
    verbose=0,
    datasets=None,
):
    assert (results[0].columns == ground_truth.columns).all()
    assert (results[0].index == ground_truth.index).all()

    sample_names = results[0].index
    celltypes = results[0].columns

    if methods is None:
        methods = [f"method {i}" for i in range(len(results))]

    # in case we have a method name multiple times add a number to it
    # assume the same methods follow each other
    method_indices = []
    current_method = None
    for method in methods:
        if current_method != method:
            current_method = method
            k = 0
        else:
            k += 1
        method_indices.append(k)

    metrics_dfs = []
    for k, (method, result) in enumerate(zip(methods, results)):
        metrics_for_result = calc_metrics_df(
            ground_truth,
            result,
            verbose=verbose,
            exclude_cols=None,
            samplewise=samplewise,
        )
        # add celltypes or sample names to metrics
        if samplewise:
            metrics_for_result["sample"] = sample_names
        else:
            metrics_for_result["celltype"] = celltypes
        metrics_for_result["method"] = method
        metrics_for_result["fold"] = method_indices[k]
        metrics_for_result["method fold"] = f"{method}-{method_indices[k]}"
        if datasets is not None:
            metrics_for_result["dataset"] = datasets[k]

        metrics_dfs.append(metrics_for_result)

    comparison_df = pd.concat(metrics_dfs, axis=0, ignore_index=True)
    # make all columns first letter uppercase
    comparison_df.columns = [col[0].upper() + col[1:] for col in comparison_df.columns]
    return comparison_df


def load_dissect_results(experiment_path):
    # load celltype abundance data
    dissect_results = []
    for i in range(5):
        df = pd.read_csv(
            f"{experiment_path}/dissect_fractions_{i}.txt", index_col=0, delimiter="\t"
        )
        dissect_results.append(df)
    ensemble_result = pd.read_csv(
        f"{experiment_path}/dissect_fractions_ens.txt", index_col=0, delimiter="\t"
    )
    return dissect_results, ensemble_result


def load_groundtruth(path):
    st_data = sc.read_h5ad(path)
    # y_real = st_data.obs[st_data.obs.columns[2::]].to_numpy()
    y_real = st_data.obs[st_data.obs.columns[2::]]

    if "Forebrain/Midbrain/Hindbrain" in y_real.columns:
        y_real = y_real.rename(
            {"Forebrain/Midbrain/Hindbrain": "Forebrain_Midbrain_Hindbrain"}, axis=1
        )
    return y_real


def extract_dataset_name(string, dataset_map):
    for key, value in dataset_map.items():
        if value in string:
            return key


def filter_data_by_dataset(dataset, dataset_names, data_list):
    filtered_data = []
    for dataset_name, data in zip(dataset_names, data_list):
        if dataset == dataset_name:
            filtered_data.append(data)
    return filtered_data


def get_base_name(tag):
    if "hybrid" in tag:
        return "hybrid"
    elif "transformer" in tag:
        return "transformer"
    else:
        return "gnn"


def get_method_names(base_names, *identifiers):
    tmp = pd.DataFrame({"base_name": base_names})
    ids = []
    for k, identifier in enumerate(identifiers):
        tmp[k] = identifier
        ids.append(k)
    tmp["idx"] = 0
    for base_name in base_names:
        tmp.loc[tmp["base_name"] == base_name, "idx"] = (
            tmp.loc[tmp["base_name"] == base_name].groupby(ids).ngroup()
        )
    tmp["method_name"] = tmp["base_name"] + "-" + tmp["idx"].astype(str)
    method_names = tmp["method_name"].to_list()
    return method_names
