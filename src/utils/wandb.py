import wandb
import json
import pandas as pd
from tqdm import tqdm
import os


def get_run_for_name(run_name, entity="dschaub", project="DISSECT-src"):
    api = wandb.Api()
    runs = api.runs(entity + "/" + project, filters={"display_name": run_name})
    try:
        return runs[0]
    except IndexError:
        print("No run found with this name")
        return None


def get_run_config(run_name, entity="dschaub", project="DISSECT-src"):
    run = get_run_for_name(run_name, entity=entity, project=project)
    if run is not None:
        return run.config
    else:
        return None


def get_result_for_run_name(run_name, entity="dschaub", project="dissect-spatial", verbose=0, **kwargs):
    if verbose:
        print("Loading", run_name)
    run = get_run_for_name(run_name, entity=entity, project=project)
    return get_result_for_run(run, verbose=verbose, **kwargs)


def get_result_for_run(run, verbose=0, identifier="step5000"):
    try:
        # filter file
        result_files = [
            file for file in run.files() if "table" in file.name and "step-" in file.name
        ]
        # order files by step num
        result_file = sorted(
            result_files, key=lambda x: int(x.name.split("-")[-1].split("_")[0])
        )[-1]
        # load file in memory and convert to df
        path = result_file.download(replace=False, exist_ok=True).name
        name = result_file.name
        assert "p-0" not in name, f"{run.name} has no result file"
    except:
        print("No result file found looking for artifacts instead")
        # TODO
        artifacts = [art for art in run.logged_artifacts() if identifier in art.name]
        assert len(artifacts) == 1
        name = artifacts[0].file().split("/")[-1]
        base_path = "./wandb/artifacts"
        path = artifacts[0].download(base_path)
        path = f"{path}/{name}"

    if verbose:
        print("Loaded", name, "for run", run.name)
    
    with open(path, "r") as f:
        result = json.load(f)
    # os.remove(f"{path}")
    try:
        result_df = pd.DataFrame(result["data"], columns=result["columns"]).set_index(
            "sample_names"
        )
    except:
        result_df = pd.DataFrame(result["data"], columns=result["columns"])
    # delete file
    return result_df


def get_sweep_runs_for_id(sweep_id, entity="dschaub", project="DISSECT-src"):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    return sweep.runs


def get_filtered_runs(filter, entity="dschaub", project="dissect-spatial"):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=filter)
    return runs


def extract_metrics_from_runs(runs, max_runs=1e5):
    run_names = []
    mean_cccs = []
    mean_rmses = []
    t = 0
    for run in tqdm(runs):
        try:
            mean_ccc = run.history(keys=["validation/mean_ccc"])[
                "validation/mean_ccc"
            ].iloc[-1]
            mean_rmse = run.history(keys=["validation/mean_rmse"])[
                "validation/mean_rmse"
            ].iloc[-1]
        except:
            continue
        if isinstance(mean_ccc, str):
            continue
        mean_cccs.append(mean_ccc)
        mean_rmses.append(mean_rmse)
        run_names.append(run.name)
        t += 1
        if t == max_runs:
            break
    return run_names, mean_cccs, mean_rmses


def get_runs_for_tags_and_filters(baseline_tag, extra_tags=None, run_filter=None, project="multi-channel-gnn"):
    if not isinstance(baseline_tag, list):
        baseline_tag = [baseline_tag]
    if extra_tags is not None and not isinstance(extra_tags, list):
        extra_tags = [extra_tags]
    runs = []
    # load baseline:
    filter_ = {"tags": {"$in": baseline_tag}, "state": "finished"}
    runs_per_tag = list(get_filtered_runs(filter=filter_, project=project))
    runs.extend(runs_per_tag)
    if extra_tags is not None:
        if run_filter is None:
            run_filter = {}
        for tag in extra_tags:
            filter_ = {"tags": {"$in": [tag]}, "state": "finished", **run_filter}
            runs_per_tag = list(get_filtered_runs(filter=filter_, project=project))
            runs.extend(runs_per_tag)
    print(f"Loaded {len(runs)} runs")
    return runs