import os
import argparse
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .datasets.text import register_text_instances
from adet.config import get_cfg
from detectron2.engine import default_argument_parser

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

_PREDEFINED_SPLITS_TEXT = {
    # 37 voc_size
    "syntext1": ("syntext1/train_images", "syntext1/annotations/train_37voc.json"),
    "syntext2": ("syntext2/train_images", "syntext2/annotations/train_37voc.json"),
    "mlt": ("mlt2017/train_images", "mlt2017/train_37voc.json"),
    "totaltext_train": ("totaltext/train_images", "totaltext/train_37voc.json"),
    "ic13_train": ("ic13/train_images", "ic13/train_37voc.json"),
    "ic15_train": ("ic15/train_images", "ic15/train_37voc.json"),
    "textocr1": ("textocr/train_images", "textocr/train_37voc_1.json"),
    "textocr2": ("textocr/train_images", "textocr/train_37voc_2.json"),

    # 96 voc_size
    "syntext1_96voc": ("syntext1/train_images", "syntext1/annotations/train_96voc.json"),
    "syntext2_96voc": ("syntext2/train_images", "syntext2/annotations/train_96voc.json"),
    "mlt_96voc": ("mlt2017/train_images", "mlt2017/train_96voc.json"),
    "totaltext_train_96voc": ("totaltext/train_images", "totaltext/train_96voc.json"),
    "ic13_train_96voc": ("ic13/train_images", "ic13/train_96voc.json"),
    "ic15_train_96voc": ("ic15/train_images", "ic15/train_96voc.json"),
    "ctw1500_train_96voc": ("ctw1500/train_images", "ctw1500/train_96voc.json"),

    # Chinese
    "chnsyn_train": ("chnsyntext/syn_130k_images", "chnsyntext/chn_syntext.json"),
    "rects_train": ("ReCTS/ReCTS_train_images", "ReCTS/rects_train.json"),
    "rects_val": ("ReCTS/ReCTS_val_images", "ReCTS/rects_val.json"),
    "lsvt_train": ("LSVT/rename_lsvtimg_train", "LSVT/lsvt_train.json"),
    "art_train": ("ArT/rename_artimg_train", "ArT/art_train.json"),

    # evaluation, just for reading images, annotations may be empty
    "totaltext_test": ("totaltext/test_images", "totaltext/test.json"),
    "ic15_test": ("ic15/test_images", "ic15/test.json"),
    "ctw1500_test": ("ctw1500/test_images", "ctw1500/test.json"),
    "inversetext_test": ("inversetext/test_images", "inversetext/test.json"),
    "rects_test": ("ReCTS/ReCTS_test_images", "ReCTS/rects_test.json"),
}

metadata_text = {
    "thing_classes": ["text"]
}


def register_all_coco(root="datasets", voc_size_cfg=37, num_pts_cfg=25):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            voc_size_cfg,
            num_pts_cfg
        )


# get the vocabulary size and number of point queries in each instance
# to eliminate blank text and sample gt according to Bezier control points
parser = default_argument_parser()
# add the following argument to avoid some errors while running demo/demo.py
parser.add_argument("--input", nargs="+", help="A list of space separated input images")
parser.add_argument(
    "--output",
    help="A file or directory to save output visualizations. "
    "If not given, will show output in an OpenCV window.",
)
parser.add_argument(
    "--opts",
    help="Modify config options using the command-line 'KEY VALUE' pairs",
    default=[],
    nargs=argparse.REMAINDER,
    )
args = parser.parse_args()
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
register_all_coco(voc_size_cfg=cfg.MODEL.TRANSFORMER.VOC_SIZE, num_pts_cfg=cfg.MODEL.TRANSFORMER.NUM_POINTS)
