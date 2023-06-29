import pandas as pd
import scanpy as sc
from tqdm import tqdm
import imgkit
import difflib

from src.utils.metrics import calc_metrics_df, harmonize_dfs
from src.utils.wandb import get_result_for_run
from src.utils.plotting import *
from src.utils.wandb import *


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


def compare_methods_new(
    results: list,
    ground_truth: pd.DataFrame,
    methods=None,
    samplewise=False,
    verbose=0,
    datasets=None,
):
    if len(results) == 0:
        return pd.DataFrame()
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
        if verbose:
            print(f"Calculating metrics for {method}...")
        df_true, df_pred = harmonize_dfs(ground_truth, result)
        sample_names = df_true.index
        celltypes = df_true.columns

        metrics_for_result = calc_metrics_df(
            df_true,
            df_pred,
            verbose=verbose,
            exclude_cols=None,
            samplewise=samplewise,
            harmonize=False,
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
    # sort dataset map
    saved_keys = []
    saved_values = []
    for key, value in dataset_map.items():
        if value in string:
            saved_keys.append(key)
            saved_values.append(value)

    assert len(saved_keys) > 0, print(f"Could not find dataset name in {string}")
    # select key with longest value
    key = saved_keys[np.argmax([len(value) for value in saved_values])]
    return key


def get_dataset_name_for_run(
    run,
    dataset_map,
    dataset_path_map,
    st_ident="st_path",
    sc_ident="sc_path",
    n=3,
    cutoff=0.6,
):
    orig_sc_string = run.config[sc_ident]
    orig_st_string = run.config[st_ident]
    # first check st path and select all suitable dataset names
    closest_matches_st = difflib.get_close_matches(
        orig_st_string, dataset_path_map.values()
    )
    # get corresponding dataset names
    # closest_datasets_st = [
    #     k for k, v in dataset_path_map.items() if v in closest_matches_st
    # ]
    closest_datasets_st = sum(
        [get_keys_for_value(dataset_path_map, match) for match in closest_matches_st],
        [],
    )

    closest_matches_sc = difflib.get_close_matches(
        orig_sc_string, dataset_map.values(), n=n, cutoff=cutoff
    )
    # closest_datasets_sc = [k for k, v in dataset_map.items() if v in closest_matches_sc]
    closest_datasets_sc = sum(
        [get_keys_for_value(dataset_map, match) for match in closest_matches_sc], []
    )

    # get overlapping dataset
    # closest_datasets = list(set(closest_datasets_st).intersection(closest_datasets_sc))
    closest_datasets = [
        dataset_st
        for dataset_st in closest_datasets_st
        if dataset_st in closest_datasets_sc
    ]
    # print(closest_datasets)
    if len(closest_datasets) == 0:
        print(f"Could not find dataset for run {run.name}")
        print("matches sc", closest_matches_sc)
        print("matches st", closest_matches_st)
        print("sc", closest_datasets_sc)
        print("st", closest_datasets_st)

    if len(closest_datasets) > 1:
        print(f"Found more than one dataset for run {run.name}")
        # print("matches sc", closest_matches_sc)
        # print("matches st", closest_matches_st)
        # print("sc", closest_datasets_sc)
        # print("st", closest_datasets_st)
    return closest_datasets[0]


def get_keys_for_value(dictionary, value):
    keys = []
    for key, val in dictionary.items():
        if val == value:
            keys.append(key)
    if len(keys) > 0:
        return keys
    else:
        raise ValueError(f"Could not find key for value {value}")


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


def results_to_df(
    results,
    dataset_names,
    method_names,
    dataset_map,
    dataset_path_map,
    dataset_filter=None,
    verbose=0,
):
    all_cellwise_results = []
    all_samplewise_results = []
    for k, v in tqdm(dataset_map.items()):
        if "mouse" in k or "lymph" in k:
            continue
        if dataset_filter is not None and k not in dataset_filter:
            continue
        if verbose:
            print(f"Loading groundtruth for {k}...")
        groundtruth = load_groundtruth(dataset_path_map[k])
        if verbose:
            print(f"Loaded groundtruth {dataset_path_map[k]}...")

        filtered_results = filter_data_by_dataset(k, dataset_names, results)
        filtered_method_names = filter_data_by_dataset(k, dataset_names, method_names)

        cellwise_results = compare_methods_new(
            filtered_results,
            groundtruth,
            methods=filtered_method_names,
            samplewise=False,
            verbose=verbose,
        )
        cellwise_results["dataset"] = k
        all_cellwise_results.append(cellwise_results)
        samplewise_results = compare_methods_new(
            filtered_results,
            groundtruth,
            methods=filtered_method_names,
            samplewise=True,
            verbose=verbose,
        )
        samplewise_results["dataset"] = k
        all_samplewise_results.append(samplewise_results)
    all_cellwise_results = pd.concat(all_cellwise_results, ignore_index=True)
    all_samplewise_results = pd.concat(all_samplewise_results, ignore_index=True)
    return all_cellwise_results, all_samplewise_results


def extract_identifier_values(
    runs,
    identifiers,
):
    id_values = []
    for id in identifiers:
        sub_id_values = []
        for run in runs:
            if id not in run.config:
                sub_id_values.append("None")
            elif run.config[id] is None:
                sub_id_values.append("default")
            else:
                sub_id_values.append(run.config[id])
        id_values.append(sub_id_values)
    return id_values


def eval_runs(
    runs,
    dataset_map,
    identifiers,
    step=None,
    verbose=0,
):
    run_names = [run.name for run in runs]
    run_tags = [run.tags for run in runs]
    results = [
        get_result_for_run(run, verbose=verbose, step=step) for run in tqdm(runs)
    ]
    dataset_names = [
        extract_dataset_name(run.config["data/reference_dir"], dataset_map)
        for run in runs
    ]
    # id_values = [
    #     [
    #         run.config[identifier] if run.config[identifier] is not None else "default"
    #         for run in runs
    #     ]
    #     for identifier in identifiers
    # ]
    id_values = extract_identifier_values(runs, identifiers)
    id_values_dict = {
        identifier: values for identifier, values in zip(identifiers, id_values)
    }
    # later define more sophisticated check
    base_names = [get_base_name(tag) for tag in run_tags]

    method_names = get_method_names(base_names, *id_values)
    methods_df = pd.DataFrame(
        {
            "method": method_names,
            "dataset": dataset_names,
            "run_name": run_names,
            **id_values_dict,
        }
    )
    print(f"Number of methods: {len(method_names)}")
    return results, dataset_names, method_names, methods_df


def aggregate_and_combine_results(
    *results, group_cols=["Method", "dataset"], rename_dict=None
):
    group_cols = ["Method", "dataset"]
    mean_results = []
    std_results = []
    for result in results:
        result_grouped_mean = result.groupby(group_cols).mean(numeric_only=True)
        result_grouped_std = result.groupby(group_cols).std(numeric_only=True)
        mean_results.append(result_grouped_mean)
        std_results.append(result_grouped_std)
    mean_results = pd.concat(mean_results, axis=1)
    std_results = pd.concat(std_results, axis=1)
    if rename_dict is not None:
        mean_results["Method"] = mean_results["Method"].replace(rename_dict)
        std_results["Method"] = std_results["Method"].replace(rename_dict)
    return mean_results, std_results


# TODO transform into eval pipeline
def evaluation_pipeline(
    dataset_map,
    dataset_path_map,
    id_values,
    name=None,
    baseline_tag="latest",
    extra_tag="ablation",
    extra_results=None,
    extra_dataset_names=None,
    extra_method_names=None,
    project="multi-channel-gnn",
    metrics=["Correlation", "CCC", "RMSE"],
    dataset_filter=None,
    save_path=None,
    return_all_dfs=False,
    step=None,
    **plot_kwargs,
):
    # load specific results
    identifiers = []
    values = []
    ablation_filter = {}
    for id, val in id_values.items():
        identifiers.append(id)
        values.append(val)
        ablation_filter[f"config.{id}"] = {"$in": val}

    runs = get_runs_for_tags_and_filters(
        baseline_tag,
        extra_tags=extra_tag,
        run_filter=ablation_filter,
        project=project,
    )

    results, dataset_names, method_names, methods_df = eval_runs(
        runs, dataset_map, identifiers, step=step
    )

    if extra_results is not None:
        assert (
            len(extra_results) == len(extra_dataset_names) == len(extra_method_names)
        ), print(
            "Lengths of extra results, extra dataset names and extra method names must be equal"
        )
        print(f"Adding {len(extra_results)} extra results to evaluation...")
        # add to dataset names etc
        dataset_names += extra_dataset_names
        method_names += extra_method_names
        results += extra_results

    # compute big result dataframes
    all_cellwise_results, all_samplewise_results = results_to_df(
        results,
        dataset_names,
        method_names,
        dataset_map,
        dataset_path_map,
        dataset_filter,
        verbose=0,
    )

    mean_results, std_results = aggregate_and_combine_results(
        all_cellwise_results, all_samplewise_results
    )

    # create method mapping automatically
    method_names = methods_df["method"].unique()
    special_names = []
    for id, vals in id_values.items():
        local_vals = []
        for val in vals:
            sub_df = methods_df[methods_df[id] == val]
            if len(sub_df) > 0:
                special_names.append(sub_df["method"].unique()[0])
                local_vals.append(val)
        if name is None:
            local_name = id
        else:
            local_name = name
        special_mapping = {
            special_name: f"{local_name}: {val}"
            for special_name, val in zip(special_names, local_vals)
        }

    print(f"Number of special names: {len(special_names)}")
    base_method = list(
        set(method_names) - set(special_names) - set(extra_method_names)
    )[0]
    base_identifier_value = methods_df.loc[
        methods_df["method"] == base_method, identifiers[0]
    ].values[0]

    if name is None:
        base_mapping = {base_method: f"MultiChannelGNN"}
    else:
        base_mapping = {
            base_method: f"MultiChannelGNN ({name.lower()}: {base_identifier_value})"
        }
    method_mapping = {
        **base_mapping,
        **special_mapping,
    }
    mean_results = mean_results.reset_index()
    mean_results["Method"] = mean_results["Method"].replace(method_mapping)
    all_cellwise_results["Method"] = all_cellwise_results["Method"].replace(
        method_mapping
    )
    all_samplewise_results["Method"] = all_samplewise_results["Method"].replace(
        method_mapping
    )

    # plot results
    if save_path is not None:
        bar_save_path = save_path.replace(".png", "_per_method.png")
    else:
        bar_save_path = None
    method_colors = plot_mean_performance_over_datasets_per_method(
        mean_results,
        method_mapping=method_mapping,
        metrics=metrics,
        save_path=bar_save_path,
        **plot_kwargs,
    )

    if save_path is not None:
        box_save_path = save_path.replace(".png", "_per_dataset.png")
    else:
        box_save_path = None
    plot_performance_per_dataset_and_method(
        all_cellwise_results,
        method_mapping=method_mapping,
        metrics=metrics,
        save_path=box_save_path,
        method_colors=method_colors,
        **plot_kwargs,
    )

    # table form
    tabular_results = (
        mean_results.groupby("Method")
        .mean(numeric_only=True)
        .drop(columns="Fold")
        .reset_index()
    )
    tabular_results_per_dataset = (
        mean_results.groupby(["Method", "dataset"]).mean().drop(columns="Fold")
    )
    # replace method name
    tabular_results["Method"] = tabular_results["Method"].replace(method_mapping)
    tabular_results = tabular_results.set_index("Method")
    if save_path is not None:
        # save dataframe as image
        # Convert the DataFrame to HTML
        html = tabular_results.style.set_table_attributes("border=1").render()

        # Save the HTML as an image
        imgkit.from_string(html, save_path.replace(".png", "_table.png"))
        pass
    if return_all_dfs:
        return (
            tabular_results,
            tabular_results_per_dataset,
            mean_results,
            std_results,
            all_cellwise_results,
            all_samplewise_results,
            methods_df,
        )
    else:
        return tabular_results, tabular_results_per_dataset
