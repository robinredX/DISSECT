from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from src.utils.wandb import *
from src.utils.config_utils import *
from src.train import train


def run_experiment(
    exp_config: DictConfig,
    experiment_name=None,
    project="dissect-spatial",
    wandb_mode="disabled",
    tags=None,
    sweep_id=None,
    device=7,
    config_path="../configs",
    config_name="train.yaml",
    save_predictions=True,
    plotting=False,
    print_config=False,
    progress_bar=True,
    overrides=[],
):
    assert isinstance(exp_config, DictConfig)
    # configure overrides
    
    if experiment_name is not None:
        exp_name = experiment_name
        overrides.extend([f"experiment={exp_name}"])
    if "experiment" in exp_config:
        exp_name = exp_config["experiment"]
        overrides.extend([f"experiment={exp_name}"])
    else:
        exp_name = None
    # load base config
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base="1.3", config_path=config_path)
    config = compose(
        config_name=config_name,
        overrides=overrides,
        return_hydra_config=True,
    )
    HydraConfig.instance().set_config(config)
    OmegaConf.set_struct(config, False)
    del config["hydra"]
    config.experiment = exp_name
    

    # set extra values
    exp_config.paths = config.paths
    config = update_config(config, exp_config)
    config = convert_nones(config)
    config.paths.output_dir = config.paths.log_dir
    
    # configure more for training
    if not progress_bar:
        del config.callbacks.rich_progress_bar
        config.trainer.enable_progress_bar = False

    if sweep_id is not None:
        tags = [sweep_id, f"{exp_name}", "test"]
        config.logger.wandb.tags = tags
        config.tags = tags
    if tags is not None:
        if isinstance(tags, str):
            tags = [tags]
        config.logger.wandb.tags = tags
        config.tags = tags
    config.extras.print_config = print_config
    config.model.save_predictions = save_predictions
    config.model.plotting = plotting
    config.trainer.devices = [device]
    config.logger.wandb.mode = wandb_mode
    config.logger.wandb.project = project
    
    if config.trainer.deterministic:
        print("Using deterministic mode")
    # run experiment
    metric_dict, object_dict = train(config)
    return metric_dict, object_dict