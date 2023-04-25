import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_mean_performance_over_datasets_per_method(
    mean_results,
    method_mapping=None,
    metrics=["Correlation", "CCC", "RMSE"],
    y_lims=None,
    add_line_plot=False,
    save_path=None,
):
    mpl.style.use("seaborn-paper")
    plot_df = mean_results.reset_index()
    plot_df["Method"] = plot_df["Method"].replace(method_mapping)

    width = len(metrics) * (9 / 3)
    fig, axs = plt.subplots(
        1,
        len(metrics),
        figsize=(width, 4),
        sharey=False,
        sharex=False,
        layout="constrained",
    )
    axs = axs.ravel()
    for k, (ax, metric) in enumerate(zip(axs, metrics)):
        sns.barplot(plot_df, x="Method", y=metric, errorbar="sd", ax=ax)
        if add_line_plot:
            sns.lineplot(
                plot_df,
                x="Method",
                y=metric,
                ax=ax,
                color="black",
                linewidth=1,
                marker="o",
                markersize=4,
            )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(True)
        if y_lims is not None:
            if y_lims[metric] is not None:
                ax.set_ylim(*y_lims[metric])
        ax.set_xlabel("Method")
        # handles, labels = ax.get_legend_handles_labels()
        # ax.get_legend().remove()
    # fig.legend(handles, labels, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.1), bbox_transform=plt.gcf().transFigure)
    # plt.suptitle(suptitle)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.tight_layout()
    plt.show()


def plot_performance_per_dataset_and_method(
    data,
    methods=None,
    metrics=["Correlation", "CCC", "RMSE"],
    y_lims=None,
    # suptitle="Celltype-wise performance for selected methods",
    method_mapping=None,
    save_path=None,
):
    # sns.set_theme()
    # sns.set_style("whitegrid")
    # sns.set_context("paper")
    mpl.style.use("seaborn-paper")

    if methods is None:
        methods = data["Method"].unique()

    methods_mask = data["Method"].isin(methods)

    if method_mapping is not None:
        data = data.copy()
        data["Method"] = data["Method"].replace(method_mapping)

    width = len(metrics) * (11 / 3)
    fig, axs = plt.subplots(
        1,
        len(metrics),
        figsize=(width, 4),
        sharey=False,
        sharex=False,
        layout="constrained",
    )
    axs = axs.ravel()
    for k, (ax, metric) in enumerate(zip(axs, metrics)):
        sns.boxplot(
            data=data.loc[methods_mask, :],
            x="dataset",
            y=metric,
            hue="Method",
            ax=ax,
            flierprops={"marker": "."},
            saturation=1.0,
            # boxprops={'fill': None},
            # palette="pastel",
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(True)
        if y_lims is not None:
            if y_lims[metric] is not None:
                ax.set_ylim(*y_lims[metric])
        ax.set_xlabel("Dataset")
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        # if k < len(metrics) - 1:
        #     ax.get_legend().remove()
        # else:
        #     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(methods),
        bbox_to_anchor=(0.5, 1.1),
        bbox_transform=plt.gcf().transFigure,
    )
    # plt.suptitle(suptitle)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.tight_layout()
    plt.show()
