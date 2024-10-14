import jax
import torch
import equinox as eqx
import jax.numpy as jnp

from CNN import CNN
from loss import loss_fn
from jaxtyping import Float, Int, Array


@eqx.filter_jit
def compute_accuracy(
    model: CNN, x: Float[Array, "batch 1, 28, 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


def evaluate(model: CNN, testloader: torch.utils.data.DataLoader):
    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        avg_loss += loss_fn(model, x, y)
        avg_acc += compute_accuracy(model, x, y)
    return avg_loss / len(testloader), avg_acc / len(testloader)
