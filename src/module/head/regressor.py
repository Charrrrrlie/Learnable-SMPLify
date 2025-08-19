import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

class MultiLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['n_head', 'in_features', 'out_features']
    n_head: int
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, n_head: int, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiLinear, self).__init__()
        self.n_head = n_head
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.empty((n_head, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(n_head, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        out = torch.einsum('kij, bkj -> bki', self.weight, input)
        if self.bias is not None:
            out += self.bias
        return out.contiguous()

    def extra_repr(self) -> str:
        return 'n_head={}, in_features={}, out_features={}, bias={}'.format(
            self.n_head, self.in_features, self.out_features, self.bias is not None
        )

class Regressor(nn.Module):
    def __init__(self, pred_pose_num, input_dim, hidden_dim, output_dim, **kwargs):
        super(Regressor, self).__init__()
        self.layers = nn.Sequential(
            self._make_multilinear(1, pred_pose_num, input_dim , hidden_dim),
            MultiLinear(pred_pose_num, hidden_dim, output_dim)
        )

    def _make_multilinear(self, num, n_head, input_dim, hidden_dim):
        plane = input_dim
        layers = []
        for i in range(num):
            layer = [MultiLinear(n_head, plane, hidden_dim),
                     nn.ReLU(inplace=True)]
            layers.extend(layer)

            plane = hidden_dim

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)