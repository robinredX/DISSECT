import numpy as np


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
    if verbose > 1:
        for k, col in enumerate(df_1.columns):
            print(f"Correlation {col}: {corrs[k]}")
    if verbose > 0:
        print(f"Mean Correlation: {mean_corr}")
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
    if verbose > 1:
        for k, col in enumerate(df_1.columns):
            print(f"RMSE {col}: {rmses[k]}")
    if verbose > 0:
        print(f"Mean RMSE: {mean_rmse}")
    df_1 = df_1[df_2.columns]
    return mean_rmse, rmses


def calc_metrics_df(df_1, df_2, verbose=1, exclude_cols=None):
    mean_corr, corrs = calc_mean_corr_df(
        df_1, df_2, verbose=verbose, exclude_cols=exclude_cols
    )
    mean_corr_sample, corrs_sample = calc_mean_corr_df(
        df_1, df_2, transpose=False, verbose=verbose, exclude_cols=exclude_cols
    )

    mean_rmse, rmses = calc_mean_rmse_df(
        df_1, df_2, verbose=verbose, exclude_cols=exclude_cols
    )
    mean_rmse_sample, rmses_sample = calc_mean_rmse_df(
        df_1, df_2, samplewise=True, verbose=verbose, exclude_cols=exclude_cols
    )

    return (
        mean_corr,
        corrs,
        mean_corr_sample,
        corrs_sample,
        mean_rmse,
        rmses,
        mean_rmse_sample,
        rmses_sample,
    )
