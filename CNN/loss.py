import jax
import equinox as eqx
import jax.numpy as jnp

from CNN import CNN
from jaxtyping import Float, Int, Array


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


def loss_fn(
    model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)


loss_fn = eqx.filter_jit(loss_fn)
