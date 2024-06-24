from flax import linen as nn

class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

def get_model(cfg):
    return SimpleNN()
