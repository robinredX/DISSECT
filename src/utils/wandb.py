import wandb
import json
import pandas as pd
from tqdm import tqdm


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


def get_result_for_run_name(run_name, entity="dschaub", project="dissect-spatial"):
    run = get_run_for_name(run_name, entity=entity, project=project)
    # filter file
    result_files = [
        file for file in run.files() if "table" in file.name and "step-" in file.name
    ]
    # order files by step num
    result_file = sorted(
        result_files, key=lambda x: int(x.name.split("-")[-1].split("_")[0])
    )[-1]
    print("Loaded", result_file.name)
    # load file in memory and convert to df
    path = result_file.download(replace=False, exist_ok=True).name
    with open(path, "r") as f:
        result = json.load(f)
    result_df = pd.DataFrame(result["data"], columns=result["columns"]).set_index(
        "sample_names"
    )
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
