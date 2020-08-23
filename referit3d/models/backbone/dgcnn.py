"""Adapting original DGCNN implementation from:
https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
"""
import torch
import torch.nn as nn


def knn(x, k):
    """ K-nearest neighbors
    :param x:
    :param k:
    :return:
    Note: Identical to original DGCNN implementation
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, subtract=True):
    """
    :param x:
    :param k:
    :param idx:
    :param subtract:
    :return:
    Note, Note: Identical to original DGCNN implementation, except when idx is not None, which they did not cover well.
    """
    batch_size = x.size(0)
    device = x.device

    num_points = x.size(2)  # for ReferIt3D this corresponds to number of objects.
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx.shape[-1]

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    #  batch_size * num_points * k + range(0, batch_size*num_points)

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if subtract:
        interaction = feature - x
    else:
        interaction = feature

    feature = torch.cat((interaction, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class DGCNN(nn.Module):
    """ The basic structure of DGCNN with a more flexibility to change (easily) the meta-parameters, e.g. depth.
    """

    def __init__(self, initial_dim, out_dim, k_neighbors,
                 intermediate_feat_dim=[64, 64, 128, 256], subtract_from_self=True):
        super(DGCNN, self).__init__()
        print('Building DGCNN will have {} graph convolutions'.format(len(intermediate_feat_dim)))
        self.k = k_neighbors
        self.layers = nn.ModuleList()
        self.subtract_from_self = subtract_from_self

        for fdim in intermediate_feat_dim:
            # Each time we multiply  by 2, since we apply a convolution to the concat signal of [x, knn(x)]
            # i.e., we double the dimensions.
            layer = nn.Sequential(nn.Conv2d(initial_dim * 2, fdim, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(fdim),
                                  nn.LeakyReLU(negative_slope=0.2))

            initial_dim = fdim
            self.layers.append(layer)

        self.final_conv = nn.Sequential(nn.Conv1d(sum(intermediate_feat_dim), out_dim, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(out_dim),
                                        nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, transpose_input_output=True, spatial_knn=None):
        """ Feed forward.
        :param x: Tensor, [B x Num-objects x Feat-Dim], if transpose_input_output is True else the dims are
            [B x Feat-Dim x Num-objects]
        :return: the result of forwarding x to DGCN
        """
        if transpose_input_output:
            x = x.transpose(2, 1)  # feat-dim first, then objects

        intermediate_features = []
        for layer in self.layers:
            x = get_graph_feature(x, k=self.k, subtract=self.subtract_from_self, idx=spatial_knn)
            x = layer(x)
            x = x.max(dim=-1)[0]
            intermediate_features.append(x)

        x = torch.cat(intermediate_features, dim=1)
        x = self.final_conv(x)

        if transpose_input_output:
            x = x.transpose(2, 1)
        return x
