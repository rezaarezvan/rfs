import jax
import optax
import jax.numpy as jnp

from flax.training import train_state

def compute_metrics(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss, acc = compute_metrics(logits, batch['label'])
        return loss, (loss, acc)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (loss, acc)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc

def train_epoch(state, train_ds, rng):
    batch_losses, batch_accuracies = [], []
    for batch in train_ds:
        state, loss, accuracy = train_step(state, batch)
        batch_losses.append(loss)
        batch_accuracies.append(accuracy)

    return state, jnp.mean(jnp.array(batch_losses)), jnp.mean(jnp.array(batch_accuracies))

@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch['image'])
    return compute_metrics(logits, batch['label'])

def evaluate(state, test_ds):
    losses, accuracies = [], []
    for batch in test_ds:
        loss, accuracy = eval_step(state, batch)
        losses.append(loss)
        accuracies.append(accuracy)

    return jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(accuracies))
