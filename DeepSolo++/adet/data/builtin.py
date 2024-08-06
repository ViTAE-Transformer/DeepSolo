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
    "mlt19_train": ("mlt19/train_images", "mlt19/mlt19_train.json"),
    "Arabic": ("Arabic/train_images", "Arabic/train.json"),
    "Bangla": ("Bangla/train_images", "Bangla/train.json"),
    "Chinese": ("Chinese/train_images", "Chinese/train.json"),
    "Hindi": ("Hindi/train_images", "Hindi/train.json"),
    "Japanese": ("Japanese/train_images", "Japanese/train.json"),
    "Korean": ("Korean/train_images", "Korean/train.json"),
    "Latin": ("Latin/train_images", "Latin/train.json"),
    "RCTW": ("RCTW/train_images", "RCTW/train.json"),
    "ArT": ("ArT/rename_artimg_train", "ArT/train.json"),
    "LSVT": ("LSVT/rename_lsvtimg_train", "LSVT/train.json"),

    # evaluation, just for reading images, annotations may be empty
    "mlt19_test": ("mlt19/test_images", "mlt19/mlt19_test.json"),
    "mlt17_test": ("mlt17/test_images", "mlt17/mlt17_test.json"),

}

metadata_text = {
    "thing_classes": ["text"]
}


def register_all_coco(root="datasets", num_pts_cfg=25):
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
register_all_coco(num_pts_cfg=cfg.MODEL.TRANSFORMER.NUM_POINTS)
