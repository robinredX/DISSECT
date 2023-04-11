from omegaconf import DictConfig
from omegaconf import OmegaConf


def print_config(cfg):
    print(OmegaConf.to_yaml(cfg))


def convert_nones(cfg):
    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            cfg[k] = convert_nones(v)
        if v == "None":
            cfg[k] = None
    return cfg


def update_config(base_cfg, update_cfg):
    # only update values that are in the base config
    for k, v in update_cfg.items():
        if k in base_cfg:
            if isinstance(v, DictConfig) and isinstance(base_cfg[k], DictConfig):
                base_cfg[k] = update_config(base_cfg[k], v)
            else:
                base_cfg[k] = v
        else:
            print(f"Key {k} not in base config.")
            base_cfg[k] = v
    return base_cfg


def convert_wandb_to_dict_config(run_config):
    # delete all dotted entries from run config
    dotted_cleaned_run_config = []
    for k, v in run_config.items():
        if "." not in k:
            # remove / from keys and replace by dots
            dotted_cleaned_run_config.append(f"{k.replace('/', '.')}={v}")
    processed_run_config = OmegaConf.from_dotlist(dotted_cleaned_run_config)
    
    # extract net entries if necessary
    net_in_config = False
    for k in run_config.keys():
        if k.startswith("net/"):
            net_in_config = True
            break
    # print result
    print(f"Net in config: {net_in_config}")
    if not net_in_config:
        new_net_config = []
        for k, v in run_config.items():
            if k.startswith("net"):
                new_net_config.append(f"{k}={v}")
        new_net_config = OmegaConf.from_dotlist(new_net_config)["net"]
        processed_run_config.net = new_net_config
    return processed_run_config

def prepare_config(base_config, run_config):
    processed_run_config = convert_wandb_to_dict_config(run_config)

    config = update_config(base_config, processed_run_config)
    config = convert_nones(config)
    config.paths.output_dir = config.paths.log_dir
    return config