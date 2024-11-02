from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.nn.datasets import mnist
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import GlobalCounters, getenv, colored, trange

from rfs.models.fastkan import FastKAN


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = mnist()
    model = FastKAN([28 * 28, 64, 10])
    opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=1e-3, weight_decay=1e-4)

    @TinyJit
    @Tensor.train()
    def train_step():
        def loss_fn(out, Y):
            return out.sparse_categorical_crossentropy(Y)

        # Get batch
        samples = Tensor.randint(getenv("BS", 128), high=X_train.shape[0])
        images, labels = X_train[samples], Y_train[samples]

        # Forward
        out = model(images.flatten(1))
        loss = loss_fn(out, labels)

        # Backward
        opt.zero_grad()
        loss.backward()
        opt.step()

        preds = out.argmax(axis=1)
        acc = (preds == labels).mean() * 100.0

        return loss, acc

    test_acc = float("0.0")
    for i in (t := trange(getenv("STEPS", 70))):
        GlobalCounters.reset()
        loss, test_acc = train_step()
        t.set_description(f"loss: {loss.item():6.2f} acc: {
                          test_acc.item():5.2f}%")

    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))
