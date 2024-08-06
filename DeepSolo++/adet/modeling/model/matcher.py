import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from adet.utils.curve_utils import BezierSampler


class CtrlPointHungarianMatcher(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            text_weight: float = 1,
            language_weight: float = 1,
            text_penalty: float = 20.0,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.text_weight = text_weight
        self.lan_weight = language_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.text_penalty = text_penalty
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        with torch.no_grad():
            sizes = [len(v["ctrl_points"]) for v in targets]
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # instance class cost
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                             (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * \
                             ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = (pos_cost_class[..., 0] - neg_cost_class[..., 0]).mean(-1, keepdims=True)

            # ctrl points class
            # ctrl points of the text center line: (bz, n_q, n_pts, 2) --> (bz x n_q, n_pts x 2)
            out_pts = outputs["pred_ctrl_points"].flatten(0, 1).flatten(-2)
            tgt_pts = torch.cat([v["ctrl_points"] for v in targets]).flatten(-2)
            cost_kpts = torch.cdist(out_pts, tgt_pts, p=1)  # (bz * n_q, num_gt)

            # language class cost
            out_lan = outputs["pred_lan_logits"]  # [bs, n_q, n_lan]
            text_head_indices_ori = torch.argmax(
                F.softmax(out_lan, dim=-1), dim=-1
            )  # [bs, n_q]
            text_head_indices = text_head_indices_ori.reshape(-1, 1).repeat(1, cost_kpts.shape[-1])
            out_lan = out_lan.flatten(0, 1).repeat(cost_kpts.shape[-1], 1)
            tgt_lan = torch.cat([v["languages"] for v in targets])
            cost_lan = F.cross_entropy(
                out_lan,
                tgt_lan.repeat_interleave(bs*num_queries),
                reduction='none'
            )
            cost_lan = cost_lan.reshape(-1, bs*num_queries).transpose(1, 0)
            text_head_indices = text_head_indices!=(tgt_lan.reshape(1, -1).repeat(bs*num_queries, 1))
            text_head_indices = torch.nonzero(text_head_indices).transpose(1,0)

            # text cost
            target_texts = torch.cat([v["texts"] for v in targets])
            target_lengths = (target_texts != 0).long().sum(dim=-1)
            text_cost = torch.zeros(
                [bs * num_queries, cost_kpts.shape[-1]], dtype=torch.float32, device=cost_kpts.device
            )
            language_set = set(tgt_lan.tolist())
            for lan in language_set:
                position_idx = torch.nonzero(tgt_lan==lan).reshape(-1)
                target_texts_temp, target_lengths_temp = target_texts[position_idx], target_lengths[position_idx]
                out_texts_temp = F.log_softmax(outputs['pred_text_logits'][int(lan)], dim=-1).flatten(0, 1)
                out_texts_temp = out_texts_temp.repeat(target_texts_temp.shape[0], 1, 1).permute(1, 0, 2)
                input_len = torch.full((out_texts_temp.size(1),), out_texts_temp.size(0),
                                       dtype=torch.long, device=out_texts_temp.device)
                target_texts_temp = torch.cat([
                    t[:target_lengths_temp[t_idx]].repeat(bs*num_queries) for t_idx, t in enumerate(target_texts_temp)
                ])
                target_lengths_temp = target_lengths_temp.repeat_interleave(bs*num_queries)
                text_cost_temp = F.ctc_loss(
                    out_texts_temp,
                    target_texts_temp,
                    input_len,
                    target_lengths_temp,
                    zero_infinity=True,
                    reduction='none'
                )
                text_cost_temp.div_(target_lengths_temp)
                text_cost[:, position_idx] = text_cost_temp.reshape(-1, bs*num_queries).transpose(1, 0)
            text_cost[text_head_indices[0], text_head_indices[1]] = self.text_penalty
            C = self.class_weight * cost_class + self.coord_weight * cost_kpts + \
                self.lan_weight * cost_lan + self.text_weight * text_cost
            C = C.view(bs, num_queries, -1).cpu()
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            indices = [
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
            ]
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            src_idx = torch.cat([src for (src, _) in indices])
            text_head_indices = text_head_indices_ori[batch_idx, src_idx]
            return indices, (batch_idx, src_idx), text_head_indices


class BezierHungarianMatcher(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            num_sample_points: int = 100,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        """Creates the matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: not the control points of bezier curve but the sampled points on curve,
            refer to "https://github.com/voldemortX/pytorch-auto-drive"
        """
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.num_sample_points = num_sample_points
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_beziers = outputs["pred_beziers"].flatten(0, 1).view(-1, 4, 2)  # (batch_size * num_queries, 4, 2)

            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_beziers = torch.cat([v["beziers"] for v in targets])  # (g, 4, 2)

            # Compute the classification cost.
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                             (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * \
                             ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost betweeen sampled points on Bezier curve
            cost_coord = torch.cdist(
                (self.bezier_sampler.get_sample_points(out_beziers)).flatten(start_dim=-2),
                (self.bezier_sampler.get_sample_points(tgt_beziers)).flatten(start_dim=-2),
                p=1
            )

            C = self.class_weight * cost_class + self.coord_weight * cost_coord
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["beziers"]) for v in targets]
            indices = [
                linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
            ]

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    cfg = cfg.MODEL.TRANSFORMER.LOSS
    return BezierHungarianMatcher(class_weight=cfg.BEZIER_CLASS_WEIGHT,
                                  coord_weight=cfg.BEZIER_COORD_WEIGHT,
                                  num_sample_points=cfg.BEZIER_SAMPLE_POINTS,
                                  focal_alpha=cfg.FOCAL_ALPHA,
                                  focal_gamma=cfg.FOCAL_GAMMA), \
           CtrlPointHungarianMatcher(class_weight=cfg.POINT_CLASS_WEIGHT,
                                     coord_weight=cfg.POINT_COORD_WEIGHT,
                                     text_weight=cfg.POINT_TEXT_WEIGHT,
                                     language_weight=cfg.LANGUAGE_WEIGHT,
                                     text_penalty=cfg.TEXT_PENALTY,
                                     focal_alpha=cfg.FOCAL_ALPHA,
                                     focal_gamma=cfg.FOCAL_GAMMA)