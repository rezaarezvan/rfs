import yaml

def get_config(config_file="config/config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config
