import numpy as np
import pandas as pd
from scipy.spatial import distance

def harmonize_dfs(df_true, df_pred, verbose=0, drop_nans=True):
    # make sure the columns are the same
    # get overlapping columns
    assert set(df_pred.index) == set(df_true.index), "Indices are not the same"
    # reindex
    df_pred = df_pred.reindex(df_true.index)
    # get column sets
    true_col_set = set(df_true.columns)
    pred_col_set = set(df_pred.columns)
    if not true_col_set == pred_col_set:
        overlaping_cols = list(set(true_col_set).intersection(set(pred_col_set)))
        if verbose > 0:
            print(f"Number of columns in true: {len(true_col_set)}")
            print(f"Number of columns in pred: {len(pred_col_set)}")
            print(f"Number of overlapping columns: {len(overlaping_cols)}")
            print(f"Overlapping columns: {overlaping_cols}")
            print(f"Non-overlapping columns in true: {true_col_set - pred_col_set}")
            print(f"Non-overlapping columns in pred: {pred_col_set - true_col_set}")
        df_true_reduced = df_true[overlaping_cols].copy()
        df_pred_reduced = df_pred[overlaping_cols].copy()
        # renormalize each row
        df_true_reduced = df_true_reduced.div(df_true_reduced.sum(axis=1), axis=0)
        df_pred_reduced = df_pred_reduced.div(df_pred_reduced.sum(axis=1), axis=0)
        if drop_nans:
            # get rows with nans in either one df
            nan_rows_true = df_true_reduced.isna().any(axis=1)
            nan_rows_pred = df_pred_reduced.isna().any(axis=1)
            nan_rows = nan_rows_true | nan_rows_pred
            # drop nan rows
            df_true_reduced = df_true_reduced.loc[~nan_rows]
            df_pred_reduced = df_pred_reduced.loc[~nan_rows]
        else:
            # replace all nan values with zeros
            df_true_reduced = df_true_reduced.fillna(0)
            df_pred_reduced = df_pred_reduced.fillna(0)
        return df_true_reduced, df_pred_reduced
    else:
        return df_true[df_pred.columns], df_pred


def calc_jsd(y_true, y_pred):
    # y_true.shape = (n_samples, n_celltypes)
    # y_pred.shape = (n_samples, n_celltypes)
    jsds = np.square(distance.jensenshannon(y_true, y_pred, axis=1))
    return jsds


def calc_jsd_df(df_true, df_pred, verbose=0, exclude_cols=None):
    if exclude_cols is not None:
        df_true = df_true.drop(exclude_cols, axis=1)
        df_pred = df_pred.drop(exclude_cols, axis=1)
    # make sure the columns are the same
    df_true = df_true[df_pred.columns]
    jsds = calc_jsd(df_true.values, df_pred.values)
    mean_jsd = np.mean(jsds)
    if verbose > 1:
        for k, col in enumerate(df_true.columns):
            print(f"JSD {col}: {jsds[k]}")
    if verbose > 0:
        print(f"Mean JSD: {mean_jsd}")

    return mean_jsd, jsds


def ccc_fn(y_true, y_pred):
    # y_true.shape = (1, n_celltypes)
    # y_pred.shape = (1, n_celltypes)
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator


def calc_ccc(y_true, y_pred, samplewise=False):
    if not samplewise:
        y_true = y_true.T
        y_pred = y_pred.T
    cccs = [ccc_fn(y_true_i, y_pred_i) for y_true_i, y_pred_i in zip(y_true, y_pred)]
    return cccs

# TODO rename function and harmonize code (calc_mean_ccc_df) same for function above
def calc_ccc_df(df_true, df_pred, samplewise=False, verbose=0, exclude_cols=None):
    if exclude_cols is not None:
        df_true = df_true.drop(exclude_cols, axis=1)
        df_pred = df_pred.drop(exclude_cols, axis=1)
    # make sure the columns are the same
    df_true = df_true[df_pred.columns]
    cccs = calc_ccc(df_true.values, df_pred.values, samplewise=samplewise)
    mean_ccc = np.mean(cccs)
    if samplewise:
        suffix = " (samplewise)"
    else:
        suffix = ""

    if verbose > 1:
        for k, col in enumerate(df_true.columns):
            print(f"CCC {col}{suffix}: {cccs[k]}")
    if verbose > 0:
        print(f"Mean CCC{suffix}: {mean_ccc}")

    return mean_ccc, cccs


def calc_mean_corr(x, y, transpose=True):
    if transpose:
        x = x.T
        y = y.T
    corrs = [np.corrcoef(x_i, y_i)[0, 1] for x_i, y_i in zip(x, y)]
    return np.mean(corrs), corrs


def calc_mean_corr_df(df_1, df_2, transpose=True, verbose=2, exclude_cols=None):
    # make sure the columns are the same
    if exclude_cols is not None:
        df_1 = df_1.drop(exclude_cols, axis=1)
        df_2 = df_2.drop(exclude_cols, axis=1)
    df_1 = df_1[df_2.columns]
    mean_corr, corrs = calc_mean_corr(df_1.values, df_2.values, transpose=transpose)
    if transpose:
        suffix = ""
    else:
        suffix = " (samplewise)"

    if verbose > 1:
        for k, col in enumerate(df_1.columns):
            print(f"Correlation {col}{suffix}: {corrs[k]}")
    if verbose > 0:
        print(f"Mean Correlation{suffix}: {mean_corr}")

    return mean_corr, corrs


def get_corr(df1, df2, col):
    x, y = df1[col].tolist(), df2[col].tolist()
    r = np.corrcoef(x, y)[0, 1]
    return r


def calc_mean_rmse(x, y, samplewise=False):
    if samplewise:
        axis = 1
    else:
        axis = 0
    rmses = list(np.sqrt(np.mean(np.square((x - y)), axis=axis)))
    return np.mean(rmses), rmses


def calc_mean_rmse_df(df_1, df_2, verbose=2, exclude_cols=None, samplewise=False):
    # make sure the columns are the same
    if exclude_cols is not None:
        df_1 = df_1.drop(exclude_cols, axis=1)
        df_2 = df_2.drop(exclude_cols, axis=1)
    df_1 = df_1[df_2.columns]
    mean_rmse, rmses = calc_mean_rmse(df_1.values, df_2.values, samplewise=samplewise)
    if samplewise:
        term = " (samplewise)"
    else:
        term = ""
    if verbose > 1:
        for k, col in enumerate(df_1.columns):
            print(f"RMSE {col}{term}: {rmses[k]}")
    if verbose > 0:
        print(f"Mean RMSE{term}: {mean_rmse}")
    df_1 = df_1[df_2.columns]
    return mean_rmse, rmses



def calc_metrics_df(df_true, df_pred, verbose=1, exclude_cols=None, samplewise=False, harmonize=True):
    if harmonize:
        df_true, df_pred = harmonize_dfs(df_true, df_pred)
    mean_corr, corrs = calc_mean_corr_df(
        df_true, df_pred, verbose=verbose, exclude_cols=exclude_cols, transpose=not samplewise
    )

    mean_rmse, rmses = calc_mean_rmse_df(
        df_true, df_pred, verbose=verbose, exclude_cols=exclude_cols, samplewise=samplewise
    )

    mean_ccc, cccs = calc_ccc_df(
        df_true, df_pred, verbose=verbose, exclude_cols=exclude_cols, samplewise=samplewise
    )
    metrics_names = ["correlation", "RMSE", "CCC"]
    metrics = [corrs, rmses, cccs]
    
    if samplewise:
        metrics_names = [f"{name} (samplewise)" for name in metrics_names]
        mean_jsd, jsds = calc_jsd_df(
            df_true, df_pred, verbose=verbose, exclude_cols=exclude_cols
        )
        metrics_names.append("JSD")
        metrics.append(jsds)
         
    metrics_df = pd.DataFrame(
        {name: metric for name, metric in zip(metrics_names, metrics)}
    )
    return metrics_df
