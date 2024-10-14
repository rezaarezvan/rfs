import jax
import optax

from CNN import CNN
from train import train
from data_loader import trainloader, testloader
from parameters import LEARNING_RATE, PRINT_EVERY, SEED, STEPS


def main():
    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key, 2)
    model = CNN(subkey)
    optim = optax.adamw(LEARNING_RATE)
    model = train(model, trainloader, testloader, optim, STEPS, PRINT_EVERY)


if __name__ == "__main__":
    main()
