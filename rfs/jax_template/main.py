import jax
import optax
import hydra
import jax.numpy as jnp

from model.model import get_model
from flax.training import train_state
from utils.config import flatten_config
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import train_epoch, evaluate
from data_loader.data_loader import get_dataloaders

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    flat_config = flatten_config(cfg)

    rng = jax.random.PRNGKey(cfg.seed)

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    rng, init_rng = jax.random.split(rng)
    model = get_model(cfg)
    params = model.init(init_rng, jnp.ones([1, 28, 28, 1]))['params']

    tx = optax.adam(cfg.training.lr)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    for epoch in range(cfg.training.epochs):
        rng, train_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_loader, train_rng)
        val_loss, val_accuracy = evaluate(state, val_loader)

        print(f"Epoch {epoch+1}/{cfg.training.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    test_loss, test_accuracy = evaluate(state, test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
