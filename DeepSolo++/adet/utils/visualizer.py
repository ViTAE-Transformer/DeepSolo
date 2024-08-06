import numpy as np
import pickle
from detectron2.utils.visualizer import Visualizer
import matplotlib.colors as mplc
import matplotlib.font_manager as mfm
import matplotlib as mpl
import matplotlib.figure as mplfigure
import random
import json


class TextVisualizer(Visualizer):
    def __init__(self, image, metadata, instance_mode, cfg):
        Visualizer.__init__(self, image, metadata, instance_mode=instance_mode)
        self.voc_sizes = cfg.MODEL.TRANSFORMER.LANGUAGE.VOC_SIZES
        self.char_map = {}
        self.language_list = cfg.MODEL.TRANSFORMER.LANGUAGE.CLASSES
        for (language_type, voc_size) in self.voc_sizes:
            with open('char_map/idx2char/'+language_type+'.json') as f:
                idx2char = json.load(f)
            f.close()
            # index 0 is the background class
            assert len(idx2char) == int(voc_size)
            self.char_map[language_type] = idx2char

    def draw_instance_predictions(self, predictions):
        ctrl_pnts = predictions.ctrl_points.numpy()
        scores = predictions.scores.tolist()
        recs = predictions.recs
        languages = predictions.languages.tolist()
        bd_pts = np.asarray(predictions.bd)

        self.overlay_instances(ctrl_pnts, scores, recs, bd_pts, languages)

        return self.output

    def _process_ctrl_pnt(self, pnt):
        points = pnt.reshape(-1, 2)
        return points

    def _ctc_decode_recognition(self, rec, language):
        last_char = '###'
        s = ''
        for c in rec:
            c = int(c)
            if c !=0:
                if last_char != c:
                    s += self.char_map[language][str(c)]
                    last_char = c
            else:
                last_char = '###'
        return s

    def overlay_instances(self, ctrl_pnts, scores, recs, bd_pnts, languages, alpha=0.4):
        # color = (0.1, 0.2, 0.5)
        colors = [(0,0.5,0),(0,0.75,0),(1,0,1),(0.75,0,0.75),(0.5,0,0.5),(1,0,0),(0.75,0,0),(0.5,0,0),
        (0,0,1),(0,0,0.75),(0.75,0.25,0.25),(0.75,0.5,0.5),(0,0.75,0.75),(0,0.5,0.5),(0,0.3,0.75)]

        for ctrl_pnt, score, rec, bd, language in zip(ctrl_pnts, scores, recs, bd_pnts, languages):
            color = random.choice(colors)

            # draw polygons
            bd = np.hsplit(bd, 2)
            bd = np.vstack([bd[0], bd[1][::-1]])
            self.draw_polygon(bd, color, alpha=alpha)

            # draw center lines
            line = self._process_ctrl_pnt(ctrl_pnt)

            # draw text
            language = self.language_list[language]
            text = self._ctc_decode_recognition(rec, language)
            # if language == 'Arabic':
            #     text = text[::-1]
            text = "{} [{}]".format(text,language[:2])
            lighter_color = self._change_color_brightness(color, brightness_factor=0)
            text_pos = bd[0] - np.array([0,20])
            horiz_align = "left"
            font_size = self._default_font_size
            self.draw_text(
                        text,
                        text_pos,
                        color=lighter_color,
                        horizontal_alignment=horiz_align,
                        font_size=font_size,
                        language=language
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
        language='Latin'
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

        font_path = "./font/Arial-Unicode-MS.ttf"
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
        return self.output