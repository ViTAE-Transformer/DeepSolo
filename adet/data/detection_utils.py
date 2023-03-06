import logging

import numpy as np
import torch
from detectron2.structures import Instances
from detectron2.data import transforms as T
from detectron2.data.detection_utils import \
    annotations_to_instances as d2_anno_to_inst
from detectron2.data.detection_utils import \
    transform_instance_annotations as d2_transform_inst_anno
from .augmentation import Pad
import random


def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):

    annotation = d2_transform_inst_anno(
        annotation,
        transforms,
        image_size,
        keypoint_hflip_indices=keypoint_hflip_indices,
    )

    if "beziers" in annotation:
        beziers = transform_ctrl_pnts_annotations(annotation["beziers"], transforms)
        annotation["beziers"] = beziers

    if "polyline" in annotation:
        polys = transform_ctrl_pnts_annotations(annotation["polyline"], transforms)
        annotation["polyline"] = polys

    if "boundary" in annotation:
        boundary = transform_ctrl_pnts_annotations(annotation["boundary"], transforms)
        annotation["boundary"] = boundary

    return annotation


def transform_ctrl_pnts_annotations(pnts, transforms):
    """
    Transform keypoint annotations of an image.

    Args:
        beziers (list[float]): Nx16 float in Detectron2 Dataset format.
        transforms (TransformList):
    """
    # (N*2,) -> (N, 2)
    pnts = np.asarray(pnts, dtype="float64").reshape(-1, 2)
    pnts = transforms.apply_coords(pnts).reshape(-1)

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = (
        sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
    )
    if do_hflip:
        raise ValueError("Flipping text data is not supported (also disencouraged).")

    return pnts


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """for line only annotations"""
    # instance = Instances(image_size)
    #
    # classes = [int(obj["category_id"]) for obj in annos]
    # classes = torch.tensor(classes, dtype=torch.int64)
    # instance.gt_classes = classes

    instance = d2_anno_to_inst(annos, image_size, mask_format)

    if not annos:
        return instance

    # add attributes
    if "beziers" in annos[0]:
        beziers = [obj.get("beziers", []) for obj in annos]
        instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)

    if "polyline" in annos[0]:
        polys = [obj.get("polyline", []) for obj in annos]
        instance.polyline = torch.as_tensor(polys, dtype=torch.float32)

    if "boundary" in annos[0]:
        boundary = [obj.get("boundary", []) for obj in annos]
        instance.boundary = torch.as_tensor(boundary, dtype=torch.float32)

    if "text" in annos[0]:
        texts = [obj.get("text", []) for obj in annos]
        instance.texts = torch.as_tensor(texts, dtype=torch.int32)

    return instance


def build_augmentation(cfg, is_train):
    """
    With option to don't use hflip

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert (
            len(min_size) == 2
        ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)

    augmentation = []
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train:
        augmentation.append(T.RandomContrast(0.3, 1.7))
        augmentation.append(T.RandomBrightness(0.3, 1.7))
        augmentation.append(T.RandomLighting(random.random() + 0.5))
        augmentation.append(T.RandomSaturation(0.3, 1.7))
        logger.info("Augmentations used in training: " + str(augmentation))
    if cfg.MODEL.BACKBONE.NAME == "build_vitaev2_backbone":
        augmentation.append(Pad(divisible_size=32))
    return augmentation


build_transform_gen = build_augmentation
"""
Alias for backward-compatibility.
"""
