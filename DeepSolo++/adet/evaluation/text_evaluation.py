import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
import glob
import shutil
from shapely.geometry import Polygon, LinearRing
import zipfile
import pickle
import editdistance
import cv2
from tqdm import tqdm


class TextEvaluator():
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._tasks = ("polygon", "recognition")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            raise AttributeError(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
            )
        self.voc_sizes = cfg.MODEL.TRANSFORMER.LANGUAGE.VOC_SIZES
        self.char_map = {}
        self.language_list = cfg.MODEL.TRANSFORMER.LANGUAGE.CLASSES
        for (language_type, voc_size) in self.voc_sizes:
            with open('char_map/idx2char/' + language_type + '.json') as f:
                idx2char = json.load(f)
            f.close()
            # index 0 is the background class
            assert len(idx2char) == voc_size
            self.char_map[language_type] = idx2char

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self.dataset_name = dataset_name
        self.submit = False
        # use dataset_name to decide eval_gt_path
        if "mlt19" in dataset_name:
            self.submit = True
            self._text_eval_gt_path = ""
            self.dataset_name = "mlt19"
        elif "mlt17" in dataset_name:
            self.submit = True
            self._text_eval_gt_path = ""
            self.dataset_name = "mlt17"
        else:
            raise NotImplementedError

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            txt_name = input['file_name'].split('/')[-1].split('.')[0].replace('ts', 'res')+'.txt'
            prediction = {"image_id": input["image_id"], "txt_name": txt_name}
            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = self.instances_to_coco_json(instances, input)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}
        PathManager.mkdirs(self._output_dir)

        if self.submit:
            self._logger.info("Saving results to {}".format(self._output_dir))
            if self.dataset_name == "mlt19":
                # for mlt19 task4: e2e text detection and recognition
                for prediction in tqdm(predictions):
                    file_path = os.path.join(self._output_dir, prediction["txt_name"])
                    with PathManager.open(file_path, "w") as f:
                        if len(prediction["instances"]) > 0:
                            for inst in prediction["instances"]:
                                write_poly,confidence = inst["polys"],inst["score"]
                                if confidence < 0.4: continue  # 0.4 for e2e task
                                write_lan,write_text = inst["language"],inst["rec"]
                                write_poly = ','.join(list(map(str,write_poly)))
                                f.write(write_poly+','+str(confidence)+','+write_text+'\n')
                        f.flush()
                zip_name = os.path.join(self._output_dir, 'mlt19_task4.zip')
                os.system('zip -rqj '+zip_name+' '+os.path.join(self._output_dir,'*.txt'))

                # for mlt19 task3: joint text detection and script identification
                for prediction in tqdm(predictions):
                    file_path = os.path.join(self._output_dir, prediction["txt_name"])
                    with PathManager.open(file_path, "w") as f:
                        if len(prediction["instances"]) > 0:
                            for inst in prediction["instances"]:
                                write_poly, confidence = inst["polys"], str(inst["score"])
                                write_lan, write_text = inst["language"], inst["rec"]
                                write_poly = ','.join(list(map(str, write_poly)))
                                f.write(write_poly + ',' + confidence + ',' + write_lan + '\n')
                        f.flush()
                zip_name = os.path.join(self._output_dir, 'mlt19_task3.zip')
                os.system('zip -rqj ' + zip_name + ' ' + os.path.join(self._output_dir, '*.txt'))

                # for mlt19 task1: multi-script text detection
                for prediction in tqdm(predictions):
                    file_path = os.path.join(self._output_dir, prediction["txt_name"])
                    with PathManager.open(file_path, "w") as f:
                        if len(prediction["instances"]) > 0:
                            for inst in prediction["instances"]:
                                write_poly, confidence = inst["polys"], str(inst["score"])
                                write_lan, write_text = inst["language"], inst["rec"]
                                write_poly = ','.join(list(map(str, write_poly)))
                                f.write(write_poly + ',' + confidence + '\n')
                        f.flush()
                zip_name = os.path.join(self._output_dir, 'mlt19_task1.zip')
                os.system('zip -rqj ' + zip_name + ' ' + os.path.join(self._output_dir, '*.txt'))
                os.system('rm -rf ' + os.path.join(self._output_dir, '*.txt'))
            elif self.dataset_name == "mlt17":
                for prediction in tqdm(predictions):
                    file_path = os.path.join(self._output_dir, prediction["txt_name"])
                    with PathManager.open(file_path, "w") as f:
                        if len(prediction["instances"]) > 0:
                            for inst in prediction["instances"]:
                                write_poly, confidence = inst["polys"], str(inst["score"])
                                write_lan, write_text = inst["language"], inst["rec"]
                                if write_lan=='Hindi':continue
                                write_poly = ','.join(list(map(str, write_poly)))
                                f.write(write_poly + ',' + confidence + ',' + write_lan + '\n')
                        f.flush()
                zip_name = os.path.join(self._output_dir, 'mlt17_task3.zip')
                os.system('zip -rqj ' + zip_name + ' ' + os.path.join(self._output_dir, '*.txt'))

                for prediction in tqdm(predictions):
                    file_path = os.path.join(self._output_dir, prediction["txt_name"])
                    with PathManager.open(file_path, "w") as f:
                        if len(prediction["instances"]) > 0:
                            for inst in prediction["instances"]:
                                write_poly, confidence = inst["polys"], str(inst["score"])
                                write_lan, write_text = inst["language"], inst["rec"]
                                write_poly = ','.join(list(map(str, write_poly)))
                                f.write(write_poly + ',' + confidence + '\n')
                        f.flush()
                zip_name = os.path.join(self._output_dir, 'mlt17_task1.zip')
                os.system('zip -rqj ' + zip_name + ' ' + os.path.join(self._output_dir, '*.txt'))
                os.system('rm -rf ' + os.path.join(self._output_dir, '*.txt'))
            else:
                raise NotImplementedError
            self._logger.info("Ready to submit results from {}".format(self._output_dir))

        self._results = OrderedDict()
        return copy.deepcopy(self._results)

    def instances_to_coco_json(self, instances, inputs):
        img_id = inputs["image_id"]
        width = inputs['width']
        height = inputs['height']
        num_instances = len(instances)
        if num_instances == 0:
            return []

        scores = instances.scores.tolist()
        languages = instances.languages.tolist()
        pnts = instances.bd.numpy()
        recs = instances.recs
        results = []
        if recs!=[]:
            for pnt, rec, score, language in zip(pnts, recs, scores, languages):
                lan = self.language_list[language]
                poly = self.pnt_to_polygon(pnt)
                poly = polygon2rbox(poly, height, width)  # only 4 points are required for MLT
                pgt = Polygon(poly)
                if not pgt.is_valid:
                    continue
                if not LinearRing(poly).is_ccw:
                    poly = poly[::-1]
                poly = poly.reshape(-1).tolist()
                s = self.ctc_decode(rec, lan)
                if lan == 'Arabic':
                    s = s[::-1]
                if s == '':
                    continue
                result = {
                    "image_id": img_id,
                    "category_id": 1,
                    "polys": poly,
                    "rec": s,
                    "score": score,
                    "language": lan
                }
                results.append(result)
        return results

    def pnt_to_polygon(self, ctrl_pnt):
        ctrl_pnt = np.hsplit(ctrl_pnt, 2)
        ctrl_pnt = np.vstack([ctrl_pnt[0], ctrl_pnt[1][::-1]])
        return ctrl_pnt.tolist()

    def ctc_decode(self, rec, lan):
        last_char = '###'
        s = ''
        for c in rec:
            c = int(c)
            if c != 0:
                if last_char != c:
                    s += self.char_map[lan][str(c)]
                    last_char = c
            else:
                last_char = '###'
        return s
            
def polygon2rbox(polygon, image_height, image_width):
    poly = np.array(polygon).reshape((-1, 2)).astype(np.float32)
    rect = cv2.minAreaRect(poly)
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = np.array(pts).reshape(-1,2)
    return pts

def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]
