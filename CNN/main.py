import jax
import jax.numpy as jnp
import optax
import torch
from jaxtyping import Array, Float, Int, PyTree
import equinox as eqx

from data_loader import trainloader, testloader
from CNN import CNN
from parameters import LEARNING_RATE, PRINT_EVERY, SEED, STEPS
from loss import loss_fn
from eval import evaluate
from train import train


def main():
    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key, 2)
    model = CNN(subkey)
    optim = optax.adamw(LEARNING_RATE)
    model = train(model, trainloader, testloader, optim, STEPS, PRINT_EVERY)


if __name__ == "__main__":
    main()
