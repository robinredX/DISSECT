import pyrootutils

base_path = pyrootutils.setup_root(
    search_from=".",
    indicator=[".gitignore"],
    project_root_env_var=True,  # set the PROJECT_ROOT environment variable to root directory
    dotenv=True,  # load environment variables from .env if exists in root directory
    pythonpath=True,  # add root directory to the PYTHONPATH (helps with imports)
    cwd=True,  # change current working directory to the root directory (helps with filepaths)
)
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.utilities.model_summary import ModelSummary
import wandb
import gc
import copy
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import _flush_loggers, configure_log

from src.data.datasets import *
from src.utils.wandb import *
from src.utils.config_utils import *
from src.utils.experiment import run_experiment
from src.utils.data_paths_valid import get_paths_for_training

# define ablation settings
ablation_settings = {
    "data": {
        "radius": [0.01, 0.03, 0.04, 0.05],
    },
    "net": {
        # "activation": ["relu6"],
        "latent_dim": [128, 512],
        "use_pos": [True],
    },
    "encoder_kwargs": {
        "num_heads": [1, 2, 8],
        "inter_skip": [False],
        "mha_channel": [False],
        "lin_channel": [False],
        "spatial_channel": [False],
        "num_layers": [2, 3],
        "plain_last": [True],
        "fusion": ["concat", "gating", "concat_simple"],
        "use_pos": [True],
    },
    "spatial_channel_kwargs": {
        # "activation": ["elu"],
        # "plain_last": [False],
        "num_layers": [2],
    },
    "mha_channel_kwargs": {
        "num_layers": [2],
        # "activation": ["elu"],
    },
    "model": {
        # "l2_lambda": [0.0, 1e-5, 1e-7],
        "alpha_max": [0.1, 0.3, 0.4, 0.5],
        "beta": [0.0, 7.5, "v2", "v3"],
        # "learning_rate": [5e-5, 1e-6],
    },
}

if __name__ == "__main__":
    # st_data_files = [
    #     "spatial/simulations_kidney_slideSeq_v2/UMOD-WT.WT-2a_resolution75.h5ad",
    #     "spatial/simulations_kidney_slideSeq_v2/UMOD-KI.KI-4b_resolution105.h5ad",
    #     "spatial/simulations_heart_seqFISH/embryo1_resolution0.11.h5ad",
    #     "spatial/simulations_heart_seqFISH/embryo2_resolution0.11.h5ad",
    #     # "spatial/simulations_heart_seqFISH/embryo3_resolution0.11.h5ad",
    #     "spatial/simulations_hypothalamus_MERFISH/0_resolution0.04.h5ad",
    #     "spatial/simulations_hypothalamus_MERFISH/1_resolution0.04.h5ad",
    #     # "spatial/V1_Mouse_Brain_Sagittal_Anterior.h5ad",
    #     # "spatial/lymph_node/st_lymph.h5ad",
    # ]
    # experiment_dirs = [
    #     "experiments/experiment_kidney_slideSeq_v2_UMOD-WT.WT-2a_resolution75",
    #     "experiments/experiment_kidney_slideSeq_v2_105",
    #     "experiments/experiment_heart_seqFISH/embryo1_resolution0.11",
    #     "experiments/experiment_heart_seqFISH/embryo2_resolution0.11",
    #     # "experiments/experiment_heart_seqFISH/embryo3_resolution0.11",
    #     "experiments/experiment_hypothalamus_MERFISH/0_resolution0.04",
    #     "experiments/experiment_hypothalamus_MERFISH/1_resolution0.04",
    #     # "experiments/experiment_mouse_st",
    #     # "experiments/experiment_lymph_node"
    # ]
    # st_paths = ["${paths.data_dir}" + f for f in st_data_files]
    # experiment_paths = ["${paths.root_dir}" + "/" + dir for dir in experiment_dirs]
    st_paths, experiment_paths = get_paths_for_training()
    st_paths = st_paths[-6::]
    experiment_paths = experiment_paths[-6::]

    run_name = "electric-field-1"
    run_name = "eager-wind-245"
    config = get_run_config(run_name, project="multi-channel-gnn")
    config = convert_wandb_to_dict_config(config)
    config.experiment = "multi_channel"

    tags = ["ablation-hard"]

    wandb_mode = "online"
    if wandb_mode == "online":
        config.trainer.fast_dev_run = False
    else:
        config.trainer.fast_dev_run = True
    config.trainer.deterministic = True

    # one big loop over all settings
    for setting, entries in ablation_settings.items():
        for entry, values in entries.items():
            for value in values:
                config_copy = copy.deepcopy(config)
                # TODO: write function to resolve config changes
                if setting == "data":
                    config_copy.data[entry] = value
                if setting == "net":
                    config_copy.net[entry] = value
                if setting == "encoder_kwargs":
                    config_copy.net.encoder_kwargs[entry] = value
                if setting == "model":
                    config_copy.model[entry] = value
                if setting == "spatial_channel_kwargs":
                    config_copy.net.encoder_kwargs.spatial_channel_kwargs[entry] = value
                if setting == "mha_channel_kwargs":
                    config_copy.net.encoder_kwargs.mha_channel_kwargs[entry] = value
                for st_path, experiment_path in list(zip(st_paths, experiment_paths))[
                    0::
                ]:
                    print(
                        f"Running experiment on {st_path} and experiment path {experiment_path}"
                    )
                    print(setting, entry, value)
                    metric_dict, object_dict = run_experiment(
                        config_copy,
                        st_path,
                        experiment_path,
                        wandb_mode=wandb_mode,
                        tags=tags,
                        config_path="../../configs",
                        progress_bar=False,
                        device=6,
                        save_predictions=True,
                        project="multi-channel-gnn"
                    )
                    del metric_dict, object_dict
                    gc.collect()
                    with torch.no_grad():
                        torch.cuda.empty_cache()
