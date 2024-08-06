import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from adet.layers.deformable_transformer import DeformableTransformer
from adet.utils.misc import (
    NestedTensor,
    inverse_sigmoid,
    nested_tensor_from_tensor_list,
    sigmoid_offset
)
from adet.modeling.model.utils import MLP


class DETECTION_TRANSFORMER(nn.Module):
    def __init__(self, cfg, backbone):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = backbone

        self.d_model = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.nhead = cfg.MODEL.TRANSFORMER.NHEADS
        self.num_encoder_layers = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dim_feedforward = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.TRANSFORMER.DROPOUT
        self.activation = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.dec_n_points = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.pos_embed_scale = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.num_classes = 1 # text or not text
        self.voc_sizes = cfg.MODEL.TRANSFORMER.LANGUAGE.VOC_SIZES
        self.language_classes = len(cfg.MODEL.TRANSFORMER.LANGUAGE.CLASSES)

        self.num_proposals = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.num_points = cfg.MODEL.TRANSFORMER.NUM_POINTS
        self.point_embed = nn.Embedding(self.num_proposals * self.num_points, self.d_model)
        self.language_token = nn.Embedding(self.num_proposals, self.d_model)

        self.transformer = DeformableTransformer(
            temp=cfg.MODEL.TRANSFORMER.TEMPERATURE,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels,
            dec_n_points=self.dec_n_points,
            enc_n_points=self.enc_n_points,
            num_proposals=self.num_proposals,
            num_points=self.num_points
        )

        if self.num_feature_levels > 1:
            strides = [8, 16, 32]
            if cfg.MODEL.BACKBONE.NAME == 'build_swin_backbone':
                if cfg.MODEL.SWIN.TYPE == 'tiny' or 'small':
                    num_channels = [192, 384, 768]
                else:
                    raise NotImplementedError
            elif cfg.MODEL.BACKBONE.NAME == 'build_vitaev2_backbone':
                if cfg.MODEL.ViTAEv2.TYPE == 'vitaev2_s':
                    num_channels = [128, 256, 512]
                else:
                    raise NotImplementedError
            else:
                num_channels = [512, 1024, 2048]

            num_backbone_outs = len(strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            num_channels[0], self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                ]
            )
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

        # bezier center line proposal after the encoder
        # x_0, y_0, ... , x_3, y_3
        self.bezier_proposal_coord = MLP(self.d_model, self.d_model, 8, 3)
        self.bezier_proposal_class = nn.Linear(self.d_model, self.num_classes)  # text or non-text
        # task specific heads after the decoder
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.boundary_offset = MLP(self.d_model, self.d_model, 4, 3)  # to rebuild the text boundary from queries
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)  # text or non-text
        self.inst_language = nn.Linear(self.d_model, self.language_classes)  # classify the language type of each instance

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.bezier_proposal_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.inst_language.bias.data = torch.ones(self.language_classes) * bias_value

        nn.init.constant_(self.bezier_proposal_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.bezier_proposal_coord.layers[-1].bias.data, 0)
        self.transformer.bezier_coord_embed = self.bezier_proposal_coord
        self.transformer.bezier_class_embed = self.bezier_proposal_class
        nn.init.constant_(self.ctrl_point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.ctrl_point_coord.layers[-1].bias.data, 0)
        nn.init.constant_(self.boundary_offset.layers[-1].weight.data, 0)
        nn.init.constant_(self.boundary_offset.layers[-1].bias.data, 0)

        self.ctrl_point_text = nn.ModuleList(
            nn.Linear(self.d_model, item[1] + 1) for item in self.voc_sizes
        )
        for lan_id, lan_head in enumerate(self.ctrl_point_text):
            lan_head.bias.data = torch.ones(self.voc_sizes[lan_id][1] + 1) * bias_value

        ######################################################################
        # shared prediction heads
        ######################################################################
        num_pred = self.num_decoder_layers
        self.ctrl_point_coord = nn.ModuleList(
            [self.ctrl_point_coord for _ in range(num_pred)]
        )
        self.boundary_offset = nn.ModuleList(
            [self.boundary_offset for _ in range(num_pred)]
        )
        self.ctrl_point_class = nn.ModuleList(
            [self.ctrl_point_class for _ in range(num_pred)]
        )
        self.inst_language = nn.ModuleList(
            [self.inst_language for _ in range(num_pred)]
        )
        self.ctrl_point_text = nn.ModuleList(
            [self.ctrl_point_text for _ in range(num_pred)]
        )

        self.transformer.decoder.ctrl_point_coord = self.ctrl_point_coord
        self.to(self.device)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks[0]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # (n_inst, d_model) -> (n_inst, 1, d_model)
        language_token = self.language_token.weight.reshape(self.num_proposals, 1, self.d_model)
        # (n_inst x n_pts, d_model) -> (n_inst, n_pts, d_model)
        point_embed = self.point_embed.weight.reshape((self.num_proposals, self.num_points, self.d_model))
        point_embed = torch.cat([point_embed, language_token], dim=1)

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact
        ) = self.transformer(srcs, masks, pos, point_embed)
        hs, language_hs = torch.split(hs, hs.shape[3] - 1, dim=3)
        # hs: [layer_num, bs, 100, n_pts, dim]
        # language_hs: [layer_num, bs, 100, 1, dim]

        outputs_coords = []
        outputs_bd_coords = []
        outputs_classes = []
        outputs_languages = []
        outputs_texts = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            # hs shape: (bs, n_proposal, n_pts, d_model)
            reference = inverse_sigmoid(reference)
            outputs_class = self.ctrl_point_class[lvl](hs[lvl])
            outputs_language = self.inst_language[lvl](language_hs[lvl]).squeeze(2) # [bs, 100, lang_cls]
            outputs_text = [self.ctrl_point_text[lvl][lan_id](hs[lvl])
                            for lan_id in range(self.language_classes)]
            tmp = self.ctrl_point_coord[lvl](hs[lvl])
            tmp_bd = self.boundary_offset[lvl](hs[lvl])
            if reference.shape[-1] == 2:
                tmp += reference
                tmp_bd += reference.repeat(1, 1, 1, 2)
            else:
                raise NotImplementedError

            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
            outputs_bd_coord = tmp_bd.sigmoid()
            outputs_bd_coords.append(outputs_bd_coord)
            outputs_classes.append(outputs_class)
            outputs_languages.append(outputs_language)
            outputs_texts.append(outputs_text)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_bd_coords = torch.stack(outputs_bd_coords)
        outputs_languages = torch.stack(outputs_languages)
        # note that outputs_texts is a list but not tensor now

        out = {
            'pred_logits': outputs_classes[-1],
            'pred_text_logits': outputs_texts[-1],
            'pred_ctrl_points': outputs_coords[-1],
            'pred_bd_points': outputs_bd_coords[-1],
            'pred_lan_logits': outputs_languages[-1]
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_classes,
                outputs_texts,
                outputs_coords,
                outputs_bd_coords,
                outputs_languages
            )

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class,
            'pred_beziers': enc_outputs_coord
        }

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_classes, outputs_texts, outputs_coords, outputs_bd_coords, outputs_languages):
        return [
            {'pred_logits':a, 'pred_text_logits':b, 'pred_ctrl_points':c, 'pred_bd_points':d, 'pred_lan_logits':e}
            for a, b, c, d, e in zip(outputs_classes[:-1],outputs_texts[:-1],outputs_coords[:-1],outputs_bd_coords[:-1],outputs_languages[:-1])
        ]