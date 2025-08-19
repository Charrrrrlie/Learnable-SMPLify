import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_modules import Unit2D, Unit_GCN, TCN_GCN_unit, TCN_GCN_unit_multiscale


default_backbone = [(64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1),
                    (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]


class STGCN(nn.Module):
    """ Spatial temporal graph convolutional networks
                        for skeleton-based action recognition.

    Input shape:
        Input shape should be (N, C, T, V)
        where N is the number of samples,
              C is the number of input channels,
              T is the length of the sequence,
              V is the number of joints or graph nodes
    
    Arguments:
        About shape:
            kp_dim (int): Number of channels in the input keypoints
            window_size (int): Length of input sequence
            num_point (int): Number of joints or graph nodes
        About net:
            backbone_config: The structure of backbone networks
        About graph convolution:
            graph: The graph of skeleton, represented by a adjacency matrix
            mask_learning: If true, use mask matrices to reweight the adjacency matrices
            use_local_bn: If true, each node in the graph have specific parameters of batch normalzation layer
        About temporal convolution:
            multiscale: If true, use multi-scale temporal convolution
            temporal_kernel_size: The kernel size of temporal convolution
            dropout: The drop out rate of the dropout layer in front of each temporal convolution layer

    """

    def __init__(self,
                 kp_dim,
                 window_size,
                 num_point,
                 graph,
                 backbone_config=None,
                 mask_learning=True,
                 use_local_bn=False,
                 multiscale=False,
                 temporal_kernel_size=9,
                 dropout=0.5):
        super(STGCN, self).__init__()

        self.kp_dim = kp_dim
        self.num_point = num_point
        self.graph = graph
        self.A = torch.from_numpy(self.graph.A).float()

        self.multiscale = multiscale

        self.data_bn = nn.SyncBatchNorm(kp_dim * num_point)

        kwargs = dict(
            A=self.A,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn,
            dropout=dropout,
            kernel_size=temporal_kernel_size)

        if self.multiscale:
            unit = TCN_GCN_unit_multiscale
        else:
            unit = TCN_GCN_unit

        # backbone
        if backbone_config is None:
            backbone_config = default_backbone

        backbone_in_c = backbone_config[0][0]
        backbone_out_c = backbone_config[-1][1]
        backbone_out_t = window_size
        backbone = []
        for in_c, out_c, stride in backbone_config:
            backbone.append(unit(in_c, out_c, stride=stride, **kwargs))
            if backbone_out_t % stride == 0:
                backbone_out_t = backbone_out_t // stride
            else:
                backbone_out_t = backbone_out_t // stride + 1
        self.backbone = nn.ModuleList(backbone)

        # head
        self.gcn0 = Unit_GCN(
            kp_dim,
            backbone_in_c,
            self.A,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn)

        self.tcn0 = Unit2D(backbone_in_c, backbone_in_c, kernel_size=9)

    def forward(self, x):
        N, C, T, V = x.size()

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)

        x = self.data_bn(x)

        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # model
        x = self.gcn0(x)
        x = self.tcn0(x)
        for m in self.backbone:
            x = m(x)

        # V pooling, x in [N, C, T // downsample, V]
        x = F.avg_pool2d(x, kernel_size=(1, V)).squeeze(-1)

        return x