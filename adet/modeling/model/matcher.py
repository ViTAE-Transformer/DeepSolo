"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
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
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.text_weight = text_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        with torch.no_grad():
            sizes = [len(v["ctrl_points"]) for v in targets]
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()

            out_texts = F.log_softmax(outputs['pred_text_logits'], dim=-1)  # (bs, n_q, n_pts, voc+1)
            n_pts, voc = out_texts.shape[2], out_texts.shape[-1] - 1
            target_texts = torch.cat([v["texts"] for v in targets])
            target_lengths = (target_texts != voc).long().sum(dim=-1)
            target_texts = torch.split(target_texts, sizes, dim=0)
            target_lengths = torch.split(target_lengths, sizes)
            texts_cost_list = []
            for out_texts_batch, targe_texts_batch, target_len_batch in zip(out_texts, target_texts, target_lengths):
                out_texts_batch_temp = out_texts_batch.repeat(targe_texts_batch.shape[0], 1, 1).permute(1, 0, 2)
                input_len = torch.full((out_texts_batch_temp.size(1),), out_texts_batch_temp.size(0), dtype=torch.long)
                targe_texts_batch_temp = torch.cat([
                    t[:target_len_batch[t_idx]].repeat(num_queries) for t_idx, t in enumerate(targe_texts_batch)
                ])
                target_len_batch_temp = target_len_batch.reshape((-1, 1)).repeat(1, num_queries).reshape(-1)
                text_cost = F.ctc_loss(
                    out_texts_batch_temp,
                    targe_texts_batch_temp,
                    input_len,
                    target_len_batch_temp,
                    blank=voc,
                    zero_infinity=True,
                    reduction='none'
                )
                text_cost.div_(target_len_batch_temp)
                text_cost_cpu = text_cost.reshape((-1, num_queries)).transpose(1, 0).cpu()
                texts_cost_list.append(text_cost_cpu)

            # ctrl points of the text center line: (bz, n_q, n_pts, 2) --> (bz * n_q, n_pts * 2)
            out_pts = outputs["pred_ctrl_points"].flatten(0, 1).flatten(-2)
            tgt_pts = torch.cat([v["ctrl_points"] for v in targets]).flatten(-2)

            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = (pos_cost_class[..., 0] - neg_cost_class[..., 0]).mean(-1, keepdims=True)
            cost_kpts = torch.cdist(out_pts, tgt_pts, p=1)  # (bz * n_q, num_gt)
            
            C = self.class_weight * cost_class + self.coord_weight * cost_kpts
            C = C.view(bs, num_queries, -1).cpu()

            indices = [linear_sum_assignment(
                c[i] + self.text_weight * texts_cost_list[i]
            ) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


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
                                     focal_alpha=cfg.FOCAL_ALPHA,
                                     focal_gamma=cfg.FOCAL_GAMMA)