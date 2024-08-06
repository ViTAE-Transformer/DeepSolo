# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from adet.utils.misc import inverse_sigmoid
from .ms_deform_attn import MSDeformAttn
from scipy.special import comb as n_over_k
from adet.utils.curve_utils import upcast
from adet.modeling.model.utils import MLP, gen_point_pos_embed


class DeformableTransformer(nn.Module):
    def __init__(
            self,
            temp=10000,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            return_intermediate_dec=False,
            num_feature_levels=4,
            dec_n_points=4,
            enc_n_points=4,
            num_proposals=300,
            num_points=25
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_proposals = num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableCompositeTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points
        )
        self.decoder = DeformableCompositeTransformerDecoder(
            temp,
            decoder_layer,
            num_decoder_layers,
            return_intermediate_dec,
            d_model
        )

        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model)
        )
        self.bezier_coord_embed = None
        self.bezier_class_embed = None
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.num_points = num_points

        Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
        BezierCoeff = lambda ts: [[Mtk(3, t, k) for k in range(4)] for t in ts]
        curve_token = torch.linspace(0, 1, num_points)
        self.bernstein_matrix = torch.tensor(BezierCoeff(curve_token), requires_grad=False)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def init_points_from_bezier_proposals(self, reference_bezier):
        bz = reference_bezier.shape[0]
        initial_reference_points = reference_bezier.view(bz, self.num_proposals, 4, 2)
        initial_reference_points = torch.matmul(
            upcast(self.bernstein_matrix.to(initial_reference_points.device)),
            upcast(initial_reference_points)
        )
        return initial_reference_points

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            proposal = grid.repeat(1, 1, 1, 4)
            assert proposal.shape[-1] == 8
            proposal = proposal.view(N_, -1, 8)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten
        )

        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        enc_outputs_class = self.bezier_class_embed(output_memory)
        enc_outputs_coord_unact = self.bezier_coord_embed(output_memory) + output_proposals

        # select top-k curves
        topk = self.num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact,
            1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 8)
        )
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()  # bs, nq, 8
        reference_points = self.init_points_from_bezier_proposals(reference_points)  # bs, nq, num_points, 2
        init_reference_out = reference_points

        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1, -1)

        hs, inter_references = self.decoder(
            query_embed,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_pos=None,
            src_padding_mask=mask_flatten
        )
        inter_references_out = inter_references
        return (hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model=256,
            d_ffn=1024,
            dropout=0.1,
            activation="relu",
            n_levels=4,
            n_heads=8,
            n_points=4
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
            self,
            src,
            pos,
            reference_points,
            spatial_shapes,
            level_start_index,
            padding_mask=None
    ):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
            self,
            src,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            pos=None,
            padding_mask=None
    ):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask
            )

        return output


class DeformableCompositeTransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model=256,
            d_ffn=1024,
            dropout=0.1,
            activation="relu",
            n_levels=4,
            n_heads=8,
            n_points=4
    ):
        super().__init__()

        # self attention (intra)
        self.attn_intra = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm_intra = nn.LayerNorm(d_model)
        self.dropout_intra = nn.Dropout(dropout)

        # self attention (inter)
        self.attn_inter = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_inter = nn.Dropout(dropout)
        self.norm_inter = nn.LayerNorm(d_model)

        # cross attention
        self.attn_cross = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout_cross = nn.Dropout(dropout)
        self.norm_cross = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
            self,
            tgt,
            query_pos,
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask=None
    ):
        ## input size
        # tgt:                bs, n_q, n_pts, embed_dim
        # query_pos:          bs, n_q, n_pts, embed_dim

        q = k = self.with_pos_embed(tgt, query_pos)
        # q.flatten(0, 1).transpose(0, 1): (n_pts, bs*n_q, dim)
        tgt2 = self.attn_intra(
            q.flatten(0, 1).transpose(0, 1),
            k.flatten(0, 1).transpose(0, 1),
            tgt.flatten(0, 1).transpose(0, 1),
        )[0].transpose(0, 1).reshape(q.shape)
        tgt = tgt + self.dropout_intra(tgt2)
        tgt = self.norm_intra(tgt)

        q_inter = k_inter = tgt_inter = torch.swapdims(tgt, 1, 2)  # (bs, n_pts, n_q, dim)
        # q_inter.flatten(0, 1).transpose(0, 1): n_q, bs*n_pts, dim
        tgt2_inter = self.attn_inter(
            q_inter.flatten(0, 1).transpose(0, 1),
            k_inter.flatten(0, 1).transpose(0, 1),
            tgt_inter.flatten(0, 1).transpose(0, 1)
        )[0].transpose(0, 1).reshape(q_inter.shape)
        tgt_inter = tgt_inter + self.dropout_inter(tgt2_inter)
        tgt_inter = torch.swapdims(self.norm_inter(tgt_inter), 1, 2)

        # cross attention
        if len(reference_points.shape) == 4:
            reference_points_loc = reference_points[:, :, None, :, :].repeat(1, 1, tgt_inter.shape[2], 1, 1)
        else:
            assert reference_points.shape[2] == tgt_inter.shape[2]
            reference_points_loc = reference_points

        tgt2 = self.attn_cross(
            self.with_pos_embed(tgt_inter, query_pos).flatten(1, 2),
            reference_points_loc.flatten(1, 2),
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask
        ).reshape(tgt_inter.shape)
        tgt_inter = tgt_inter + self.dropout_cross(tgt2)
        tgt = self.norm_cross(tgt_inter)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableCompositeTransformerDecoder(nn.Module):
    def __init__(
            self,
            temp,
            decoder_layer,
            num_layers,
            return_intermediate=False,
            d_model=256
    ):
        super().__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.ctrl_point_coord = None
        self.ref_point_head = MLP(d_model, d_model, d_model, 2)
        self.temp = temp
        self.d_model = d_model

    def forward(
            self,
            tgt,
            reference_points,
            src,
            src_spatial_shapes,
            src_level_start_index,
            src_valid_ratios,
            query_pos=None,
            src_padding_mask=None
    ):
        # reference_points: bs, 100, 4
        # src_valid_ratios: bs, 4, 2
        # query_pos: bs, 100, n_pts, d_model
        output = tgt  # bs, 100, n_pts, d_model
        assert query_pos is None and reference_points.shape[-1] == 2

        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                # reference_points: (bs, nq, n_pts, 2), reference_points_input: (bs, nq, n_pts, 4, 2)
                reference_points_input = reference_points[:, :, :, None] * src_valid_ratios[:, None, None]

            query_pos = gen_point_pos_embed(reference_points_input[:, :, :, 0, :], self.d_model, self.temp)
            query_pos = self.ref_point_head(query_pos)

            output = layer(output, query_pos, reference_points_input, src,
                           src_spatial_shapes, src_level_start_index, src_padding_mask)

            if self.ctrl_point_coord is not None:
                tmp = self.ctrl_point_coord[lid](output)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
