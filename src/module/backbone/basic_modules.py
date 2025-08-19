import torch
import torch.nn as nn

import math

def conv_init(module):
    # he_normal
    n = module.out_channels
    for k in module.kernel_size:
        n *= k
    module.weight.data.normal_(0, math.sqrt(2. / n))


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Unit2D(nn.Module):
    def __init__(self,
                 D_in,
                 D_out,
                 kernel_size,
                 stride=1,
                 dim=2,
                 dropout=0,
                 bias=True):
        super(Unit2D, self).__init__()
        pad = int((kernel_size - 1) / 2)
        if dim == 2:
            self.conv = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(kernel_size, 1),
                padding=(pad, 0),
                stride=(stride, 1),
                bias=bias)
        elif dim == 3:
            self.conv = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(1, kernel_size),
                padding=(0, pad),
                stride=(1, stride),
                bias=bias)
        else:
            raise ValueError()

        self.bn = nn.SyncBatchNorm(D_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # initialize
        conv_init(self.conv)

    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.bn(self.conv(x)))
        return x


class Unit_GCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 use_local_bn=False,
                 kernel_size=1,
                 stride=1,
                 mask_learning=False):
        super(Unit_GCN, self).__init__()

        # ==========================================
        # number of nodes
        self.V = A.size()[-1]

        # the adjacency matrixes of the graph
        self.A = nn.Parameter(
            A.clone(), requires_grad=False).view(-1, self.V, self.V)

        # number of input channels
        self.in_channels = in_channels

        # number of output channels
        self.out_channels = out_channels

        # if true, use mask matrix to reweight the adjacency matrix
        self.mask_learning = mask_learning

        # number of adjacency matrix (number of partitions)
        self.num_A = self.A.size()[0]

        # if true, each node have specific parameters of batch normalizaion layer.
        # if false, all nodes share parameters.
        self.use_local_bn = use_local_bn
        # ==========================================

        self.conv_list = nn.ModuleList([
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1)) for i in range(self.num_A)
        ])

        if mask_learning:
            self.mask = nn.Parameter(torch.ones(self.A.size()))
        if use_local_bn:
            self.bn = nn.SyncBatchNorm(self.out_channels * self.V)
        else:
            self.bn = nn.SyncBatchNorm(self.out_channels)

        self.relu = nn.ReLU()

        # initialize
        for conv in self.conv_list:
            conv_init(conv)

    def forward(self, x):
        N, C, T, V = x.size()
        self.A = self.A.cuda(x.get_device())
        A = self.A

        # reweight adjacency matrix
        if self.mask_learning:
            A = A * self.mask

        # graph convolution
        for i, a in enumerate(A):
            xa = x.view(-1, V).mm(a).view(N, C, T, V)

            if i == 0:
                y = self.conv_list[i](xa)
            else:
                y = y + self.conv_list[i](xa)

        # batch normalization
        if self.use_local_bn:
            y = y.permute(0, 1, 3, 2).contiguous().view(
                N, self.out_channels * V, T)
            y = self.bn(y)
            y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
        else:
            y = self.bn(y)

        # nonliner
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 A,
                 kernel_size=9,
                 stride=1,
                 dropout=0.5,
                 use_local_bn=False,
                 mask_learning=False):
        super(TCN_GCN_unit, self).__init__()
        half_out_channel = out_channel / 2
        self.A = A
        self.V = A.size()[-1]
        self.C = in_channel

        self.gcn1 = Unit_GCN(
            in_channel,
            out_channel,
            A,
            use_local_bn=use_local_bn,
            mask_learning=mask_learning)
        self.tcn1 = Unit2D(
            out_channel,
            out_channel,
            kernel_size=kernel_size,
            dropout=dropout,
            stride=stride)
        if (in_channel != out_channel) or (stride != 1):
            self.down1 = Unit2D(
                in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.down1 = None

    def forward(self, x):
        # N, C, T, V = x.size()
        x = self.tcn1(self.gcn1(x)) + (x if
                                       (self.down1 is None) else self.down1(x))
        return x


class TCN_GCN_unit_multiscale(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 kernel_size=9,
                 stride=1,
                 **kwargs):
        super(TCN_GCN_unit_multiscale, self).__init__()
        self.unit_1 = TCN_GCN_unit(
            in_channels,
            out_channels / 2,
            A,
            kernel_size=kernel_size,
            stride=stride,
            **kwargs)
        self.unit_2 = TCN_GCN_unit(
            in_channels,
            out_channels - out_channels / 2,
            A,
            kernel_size=kernel_size * 2 - 1,
            stride=stride,
            **kwargs)

    def forward(self, x):
        return torch.cat((self.unit_1(x), self.unit_2(x)), dim=1)