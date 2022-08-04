# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified from DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from softgroup.model.detr.attention import MultiheadAttention as CustionMultiheadAttention
from softgroup.model.detr.helper import (
    ACTIVATION_DICT,
    NORM_DICT,
    WEIGHT_INIT_DICT,
    BatchNormDim1Swap,
    GenericMLP,
    get_clones,
)
from softgroup.model.detr.pos_embedding import PositionEmbeddingCoordsSine
from typing import Optional


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm_fn_name="ln",
        return_intermediate=False,
        weight_init_name="xavier_uniform",
    ):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = None
        if norm_fn_name is not None:
            self.norm = NORM_DICT[norm_fn_name](self.layers[0].linear2.out_features)
        self.return_intermediate = return_intermediate
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        relative_pos: Optional[Tensor] = None,
        transpose_swap: Optional[bool] = False,
        return_attn_weights: Optional[bool] = False,
    ):
        if transpose_swap:
            tgt = tgt.permute(2, 0, 1)
            memory = memory.permute(2, 0, 1)  # memory: bs, c, t -> t, b, c

            if pos is not None:
                pos = pos.permute(2, 0, 1)
            if query_pos is not None:
                query_pos = query_pos.permute(2, 0, 1)
            if relative_pos is not None:
                relative_pos = relative_pos.permute(2, 3, 0, 1)

        output = tgt

        intermediate = []
        # attns = []

        n_points = [1024, 1024, 2048, 2048, 4096, 4096]
        # n_points = [1024, 2048, 2048, 4096, 4096, 8192]
        for l, layer in enumerate(self.layers):
            memory_ = memory[: n_points[l], ...]
            pos_ = pos[: n_points[l], ...]
            relative_pos_ = relative_pos[:, : n_points[l], :, :]
            output, attn = layer(
                output,
                memory_,
                pos=pos_,
                query_pos=query_pos,
                relative_pos=relative_pos_,
                return_attn_weights=return_attn_weights,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            # if return_attn_weights:
            #     attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=4,
        dim_feedforward=256,
        dropout=0.1,
        dropout_attn=None,
        activation="relu",
        normalize_before=True,
        use_rel=False,
        norm_fn_name="ln",
    ):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm1 = NORM_DICT[norm_fn_name](d_model)
        self.norm2 = NORM_DICT[norm_fn_name](d_model)

        self.norm3 = NORM_DICT[norm_fn_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.dropout3 = nn.Dropout(dropout, inplace=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = ACTIVATION_DICT[activation](inplace=False)
        self.normalize_before = normalize_before

        self.use_rel = use_rel

        self.nhead = nhead

        if self.use_rel:
            self.attn_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.v_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
            )
            self.out_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
            )
        else:
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.nhead
        return (
            x.reshape(batch_size, seq_len, self.nhead, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.nhead, seq_len, sub_dim)
        )

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.nhead
        return (
            x.reshape(batch_size, self.nhead, seq_len, in_feature)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )

    def forward_pre_rel(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        relative_pos: Optional[Tensor] = None,
        return_attn_weights: Optional[bool] = False,
    ):

        # NOTE self attn between queries themself: use absolute euclid pos
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        # NOTE cross attn between queries and contexst: use relative pos
        n_queries, n_context, batch, channel = relative_pos.shape
        tgt2_expand = tgt2[:, None, :, :].repeat(1, n_context, 1, 1)

        memory_expand = memory[None, :, :, :].repeat(n_queries, 1, 1, 1)

        sim = self.attn_mlp(tgt2_expand - memory_expand + relative_pos)
        attn = F.softmax(sim / np.sqrt(sim.shape[-1]), dim=1)

        v2 = self.v_mlp(memory_expand + relative_pos)
        tgt = torch.einsum("qcbf,qcbf->qbf", attn, v2)  # n_queries, batch, channel
        tgt = self.out_mlp(tgt)
        ###########################################################################################

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        relative_pos: Optional[Tensor] = None,
        return_attn_weights: Optional[bool] = False,
    ):
        return self.forward_pre_rel(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            relative_pos,
            return_attn_weights,
        )
