import numpy as np
import pickle
from detectron2.utils.visualizer import Visualizer
import matplotlib.colors as mplc
import matplotlib.font_manager as mfm
import matplotlib as mpl
import matplotlib.figure as mplfigure
import random
from shapely.geometry import LineString
import math
import operator
from functools import reduce

class TextVisualizer(Visualizer):
    def __init__(self, image, metadata, instance_mode, cfg):
        Visualizer.__init__(self, image, metadata, instance_mode=instance_mode)
        self.voc_size = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.use_customer_dictionary = cfg.MODEL.TRANSFORMER.CUSTOM_DICT
        if self.voc_size == 96:
            self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        elif self.voc_size == 37:
            self.CTLABELS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
        else:
            with open(self.use_customer_dictionary, 'rb') as fp:
                self.CTLABELS = pickle.load(fp)
        # voc_size includes the unknown class, which is not in self.CTABLES
        assert(int(self.voc_size - 1) == len(self.CTLABELS)), "voc_size is not matched dictionary size, got {} and {}.".format(int(self.voc_size - 1), len(self.CTLABELS))

    def draw_instance_predictions(self, predictions):
        ctrl_pnts = predictions.ctrl_points.numpy()
        scores = predictions.scores.tolist()
        recs = predictions.recs
        bd_pts = np.asarray(predictions.bd)

        self.overlay_instances(ctrl_pnts, scores, recs, bd_pts)

        return self.output

    def _process_ctrl_pnt(self, pnt):
        points = pnt.reshape(-1, 2)
        return points

    def _ctc_decode_recognition(self, rec):
        last_char = '###'
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    if self.voc_size == 37 or self.voc_size == 96:
                        s += self.CTLABELS[c]
                        last_char = c
                    else:
                        s += str(chr(self.CTLABELS[c]))
                        last_char = c
            else:
                last_char = '###'
        return s

    def overlay_instances(self, ctrl_pnts, scores, recs, bd_pnts, alpha=0.4):
        colors = [(0,0.5,0),(0,0.75,0),(1,0,1),(0.75,0,0.75),(0.5,0,0.5),(1,0,0),(0.75,0,0),(0.5,0,0),
        (0,0,1),(0,0,0.75),(0.75,0.25,0.25),(0.75,0.5,0.5),(0,0.75,0.75),(0,0.5,0.5),(0,0.3,0.75)]

        for ctrl_pnt, score, rec, bd in zip(ctrl_pnts, scores, recs, bd_pnts):
            color = random.choice(colors)

            # draw polygons
            if bd is not None:
                bd = np.hsplit(bd, 2)
                bd = np.vstack([bd[0], bd[1][::-1]])
                self.draw_polygon(bd, color, alpha=alpha)

            # draw center lines
            line = self._process_ctrl_pnt(ctrl_pnt)
            line_ = LineString(line)
            center_point = np.array(line_.interpolate(0.5, normalized=True).coords[0], dtype=np.int32)
            # self.draw_line(
            #     line[:, 0],
            #     line[:, 1],
            #     color=color,
            #     linewidth=2
            # )
            # for pt in line:
            #     self.draw_circle(pt, 'w', radius=4)
            #     self.draw_circle(pt, 'r', radius=2)

            # draw text
            text = self._ctc_decode_recognition(rec)
            if self.voc_size == 37:
                text = text.upper()
            # text = "{:.2f}: {}".format(score, text)
            text = "{}".format(text)
            lighter_color = self._change_color_brightness(color, brightness_factor=0)
            if bd is not None:
                text_pos = bd[0] - np.array([0,15])
            else:
                text_pos = center_point
            horiz_align = "left"
            font_size = self._default_font_size
            self.draw_text(
                        text,
                        text_pos,
                        color=lighter_color,
                        horizontal_alignment=horiz_align,
                        font_size=font_size,
                        draw_chinese=False if self.voc_size == 37 or self.voc_size == 96 else True
                    )

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
        draw_chinese=False
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW
        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))
        
        x, y = position
        if draw_chinese:
            font_path = "./simsun.ttc"
            prop = mfm.FontProperties(fname=font_path)
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
                fontproperties=prop
            )
        else:
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
            )
        return self.output