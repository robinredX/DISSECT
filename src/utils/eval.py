import pandas as pd
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
    gnn_metrics = {}
    for key, gnn_result in gnn_results.items():
        gnn_metrics[key] = calc_metrics_df(
            gnn_result, y_real, verbose=verbose, exclude_cols=None
        )

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
    comparison_df = pd.concat(
        [*dissect_results_dfs, *gnn_results_dfs], axis=0, ignore_index=True
    )
    return comparison_df
