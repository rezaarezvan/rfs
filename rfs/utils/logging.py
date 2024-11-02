import os
import logging


def setup_logging(run_name):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%I:%M:%S",
    )
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(f"models/{run_name}", exist_ok=True)
    os.makedirs(f"results/{run_name}", exist_ok=True)
