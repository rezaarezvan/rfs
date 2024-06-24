from omegaconf import DictConfig

def flatten_config(cfg):
    flat_config = {}
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            for sub_key, sub_value in flatten_config(value).items():
                flat_config[f"{key}.{sub_key}"] = sub_value
        else:
            flat_config[key] = value
    return flat_config
