import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from adet.utils.misc import accuracy, is_dist_avail_and_initialized
from detectron2.utils.comm import get_world_size
from adet.utils.curve_utils import BezierSampler


def sigmoid_focal_loss(inputs, targets, num_inst, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss.ndim == 4:
        return loss.mean((1, 2)).sum() / num_inst
    elif loss.ndim == 3:
        return loss.mean(1).sum() / num_inst
    else:
        raise NotImplementedError(f"Unsupported dim {loss.ndim}")


class SetCriterion(nn.Module):
    """
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
            self,
            num_classes,
            enc_matcher,
            dec_matcher,
            weight_dict,
            enc_losses,
            num_sample_points,
            dec_losses,
            voc_size,
            num_ctrl_points,
            focal_alpha=0.25,
            focal_gamma=2.0
    ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.enc_matcher = enc_matcher
        self.dec_matcher = dec_matcher
        self.weight_dict = weight_dict
        self.enc_losses = enc_losses
        self.num_sample_points = num_sample_points
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)
        self.dec_losses = dec_losses
        self.voc_size = voc_size
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_ctrl_points = num_ctrl_points

    def loss_labels(self, outputs, targets, indices, num_inst, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(src_logits.shape[:-1], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes_o = torch.cat([t["labels"][J]
                                      for t, (_, J) in zip(targets, indices)])
        if len(target_classes_o.shape) < len(target_classes[idx].shape):
            target_classes_o = target_classes_o[..., None]
        target_classes[idx] = target_classes_o

        shape = list(src_logits.shape)
        shape[-1] += 1
        target_classes_onehot = torch.zeros(shape,
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[..., :-1]
        # src_logits, target_classes_onehot: (bs, nq, n_ctrl_pts, 1)
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_inst,
                                     alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_beziers(self, outputs, targets, indices, num_inst):
        # may FIX: (1) seg valid points
        assert 'pred_beziers' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_beziers = outputs['pred_beziers'][idx]
        src_beziers =  self.bezier_sampler.get_sample_points(src_beziers.view(-1, 4, 2))
        target_beziers = torch.cat(
            [t['beziers'][i] for t, (_, i) in zip(targets, indices)],
            dim=0
        )
        target_beziers = self.bezier_sampler.get_sample_points(target_beziers)
        if target_beziers.numel() == 0:
            target_beziers = src_beziers.clone().detach()
        loss_bezier = F.l1_loss(src_beziers, target_beziers, reduction='none')
        losses = {}
        losses['loss_bezier'] = loss_bezier.sum() / num_inst
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_inst):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.mean(-2).argmax(-1) == 0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_texts(self, outputs, targets, indices, num_inst):
        # CTC loss for classification of points
        assert 'pred_text_logits' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_texts = outputs['pred_text_logits'][idx]  # shape: (n, length, voc_size+1)
        src_texts = src_texts.permute(1, 0, 2)
        src = F.log_softmax(src_texts, dim=-1)  # shape: (length, n, voc_size+1)

        target_texts = torch.cat([t['texts'][i] for t, (_, i) in zip(targets, indices)]) # n, length
        input_lengths = torch.full((src.size(1),), src.size(0), dtype=torch.long)
        target_lengths = (target_texts != self.voc_size).long().sum(dim=-1)
        target_texts = torch.cat([t[:l] for t, l in zip(target_texts, target_lengths)])

        return {
            'loss_texts': F.ctc_loss(
                src,
                target_texts,
                input_lengths,
                target_lengths,
                blank=self.voc_size,
                zero_infinity=True
            )
        }

    def loss_ctrl_points(self, outputs, targets, indices, num_inst):
        """Compute the L1 regression loss
        """
        assert 'pred_ctrl_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_ctrl_points = outputs['pred_ctrl_points'][idx]
        target_ctrl_points = torch.cat([t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_ctrl_points = F.l1_loss(src_ctrl_points, target_ctrl_points, reduction='sum')
        losses = {'loss_ctrl_points': loss_ctrl_points / num_inst}
        return losses

    def loss_bd_points(self, outputs, targets, indices, num_inst):
        assert 'pred_bd_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_bd_points = outputs['pred_bd_points'][idx]
        target_bd_points = torch.cat([t['bd_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bd_points = F.l1_loss(src_bd_points, target_bd_points, reduction='sum')
        losses = {'loss_bd_points': loss_bd_points / num_inst}
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                               for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_inst, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'ctrl_points': self.loss_ctrl_points,
            'beziers': self.loss_beziers,
            'texts': self.loss_texts,
            'bd_points': self.loss_bd_points
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_inst, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.dec_matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_inst = sum(len(t['ctrl_points']) for t in targets)
        num_inst = torch.as_tensor(
            [num_inst], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_inst)
        num_inst = torch.clamp(num_inst / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        for loss in self.dec_losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_inst, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.dec_matcher(aux_outputs, targets)
                for loss in self.dec_losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_inst, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.enc_matcher(enc_outputs, targets)
            for loss in self.enc_losses:
                kwargs = {}
                if loss == 'labels':
                    kwargs['log'] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, targets, indices, num_inst, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses