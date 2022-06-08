from collections import OrderedDict

import spconv.pytorch as spconv
import torch
from spconv.pytorch.modules import SparseModule
from torch import nn


class MLP(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)


# current 1x1 conv in spconv2x has a bug. It will be removed after the bug is fixed
class Custom1x1Subm3d(spconv.SparseConv3d):

    def forward(self, input):
        features = torch.mm(input.features, self.weight.view(self.out_channels, self.in_channels).T)
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(features, input.indices, input.spatial_shape,
                                             input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class ResidualBlock(SparseModule):

    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                Custom1x1Subm3d(in_channels, out_channels, kernel_size=1, bias=False))

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels), nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key), norm_fn(out_channels), nn.ReLU(),
            spconv.SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key))

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape,
                                           input.batch_size)
        output = self.conv_branch(input)
        out_feats = output.features + self.i_branch(identity).features
        output = output.replace_feature(out_feats)

        return output


class UBlock(nn.Module):

    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {
            'block{}'.format(i):
            block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            self.u = UBlock(
                nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):

        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape,
                                           output.batch_size)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)
        return output

class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

def conv_with_kaiming_uniform(norm=None, activation=None, use_sep=False):
    def make_conv(in_channels, out_channels):
        conv_func = Conv1d
        if use_sep:
            assert in_channels == out_channels
            groups = in_channels
        else:
            groups = 1

        conv = conv_func(in_channels,
                         out_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0,
                         groups=groups,
                         bias=(norm is None))

        nn.init.kaiming_uniform_(conv.weight, a=1)
        if norm is None:
            nn.init.constant_(conv.bias, 0)

        module = [conv,]
        if norm is not None and len(norm) > 0:
            norm_module = torch.nn.BatchNorm1d(out_channels)
            module.append(norm_module)
        if activation is not None:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv