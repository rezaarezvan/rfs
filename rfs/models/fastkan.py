import numpy as np

from tinygrad import nn
from tinygrad.tensor import Tensor


class SplineLinear:
    def __init__(self, in_features,  out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.linear_function = nn.Linear(in_features, out_features, bias=False)

    def __call__(self, x):
        return self.linear_function(x)


class RBF:
    def __init__(self, grid_min=-2., grid_max=2., num_grids=8, denominator=None):
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = Tensor(np.linspace(grid_min, grid_max,
                           num_grids, dtype=np.float32), requires_grad=True)
        self.denominator = denominator or (
            grid_max - grid_min) / (num_grids - 1)

    def __call__(self, x):
        return (-(((x[..., None] - self.grid) / self.denominator).pow(2))).exp()


class FastKANLayer:
    def __init__(
        self,
        input_dim,
        output_dim,
        grid_min=-2.0,
        grid_max=2.0,
        num_grids=8,
        use_base_update=True,
        use_layernorm=True,
        base_activation=Tensor.silu,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RBF(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(
            input_dim * num_grids, output_dim)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def __call__(
        self, x, use_layernorm=True
    ):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        spline_basis_view = spline_basis.view(*spline_basis.shape[:-2], -1)
        ret = self.spline_linear(spline_basis_view)
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class FastKAN:
    def __init__(
        self,
        layers_hidden,
        grid_min=-2.0,
        grid_max=2.0,
        num_grids=8,
        use_base_update=True,
        base_activation=Tensor.silu,
    ):
        self.layers = [
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ]

    def __call__(self, x):
        return x.sequential(self.layers)
