"""
MIT License

Copyright (c) 2020-2021 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from os import path
import time

import cv2
import numpy as np

from . import media
from .config import YOLOConfig
from ._common import (
    get_yolo_detections as _get_yolo_detections,
    get_yolo_tiny_detections as _get_yolo_tiny_detections,
    fit_to_original as _fit_to_original,
)
from turbojpeg import TurboJPEG, TJPF_RGB


class BaseClass:
    def __init__(self):
        self.config = YOLOConfig()
        self.turboJpeg = TurboJPEG()

    def get_yolo_detections(self, yolos, prob_thresh: float) -> np.ndarray:
        """
        Warning!
            - change order
            - change c0 -> p(c0)

        @param `yolos`: List[Dim(1, height, width, 5 + classes)]
        @param `prob_thresh`

        @return `pred_bboxes`
            Dim(-1, (x, y, w, h, cls_id, prob))
        """
        if len(yolos) == 2:
            return _get_yolo_tiny_detections(
                yolo_0=yolos[0],
                yolo_1=yolos[1],
                mask_0=self.config.masks[0],
                mask_1=self.config.masks[1],
                anchors=self.config.anchors,
                beta_nms=self.config.metayolos[0].beta_nms,
                new_coords=self.config.metayolos[0].new_coords,
                prob_thresh=prob_thresh,
            )

        return _get_yolo_detections(
            yolo_0=yolos[0],
            yolo_1=yolos[1],
            yolo_2=yolos[2],
            mask_0=self.config.masks[0],
            mask_1=self.config.masks[1],
            mask_2=self.config.masks[2],
            anchors=self.config.anchors,
            beta_nms=self.config.metayolos[0].beta_nms,
            new_coords=self.config.metayolos[0].new_coords,
            prob_thresh=prob_thresh,
        )

    def fit_to_original(
        self, pred_bboxes: np.ndarray, origin_height: int, origin_width: int
    ):
        """
        Warning! change pred_bboxes directly

        @param `pred_bboxes`
            Dim(-1, (x, y, w, h, cls_id, prob))
        """
        _fit_to_original(
            pred_bboxes,
            self.config.net.height,
            self.config.net.width,
            origin_height,
            origin_width,
        )

    def resize_image(self, image, ground_truth=None):
        """
        @param image:        Dim(height, width, channels)
        @param ground_truth: [[center_x, center_y, w, h, class_id], ...]

        @return resized_image or (resized_image, resized_ground_truth)

        Usage:
            image = yolo.resize_image(image)
            image, ground_truth = yolo.resize_image(image, ground_truth)
        """
        input_shape = self.config.net.input_shape
        return media.resize_image(
            image,
            target_shape=input_shape,
            ground_truth=ground_truth,
        )

    def draw_bboxes(self, image, pred_bboxes):
        """
        @parma `image`:  Dim(height, width, channel)
        @parma `pred_bboxes`
            Dim(-1, (x, y, w, h, cls_id, prob))

        @return drawn_image

        Usage:
            image = yolo.draw_bboxes(image, bboxes)
        """
        return media.draw_bboxes(image, pred_bboxes, names=self.config.names)

    #############
    # Inference #
    #############

    def predict(self, frame: np.ndarray, prob_thresh: float):
        # pylint: disable=unused-argument, no-self-use
        return [[0.0, 0.0, 0.0, 0.0, -1]]

    def inference(
        self,
        media_path,
        is_image: bool = True,
        cv_apiPreference=None,
        cv_frame_size: tuple = None,
        cv_fourcc: str = None,
        cv_waitKey_delay: int = 1,
        prob_thresh: float = 0.25,
    ):
        if isinstance(media_path, str) and not path.exists(media_path):
            raise FileNotFoundError("{} does not exist".format(media_path))

        #cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)

        if is_image:
            with open(media_path, "rb") as f:
                imageFile = f.read()
                frame = self.turboJpeg.decode(imageFile)
                frame_rgb = self.turboJpeg.decode(
                    imageFile, pixel_format=TJPF_RGB
                )

            start_time = time.time()
            bboxes = self.predict(frame_rgb, prob_thresh=prob_thresh)
            exec_time = time.time() - start_time
            print("time: {:.2f} ms".format(exec_time * 1000))

            image = self.draw_bboxes(frame, bboxes)
            #cv2.imshow("result", image)
        else:
            if cv_apiPreference is None:
                cap = cv2.VideoCapture(media_path)
            else:
                cap = cv2.VideoCapture(media_path, cv_apiPreference)

            if cv_frame_size is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cv_frame_size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cv_frame_size[1])

            if cv_fourcc is not None:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cv_fourcc))

            prev_time = time.time()
            if cap.isOpened():
                while True:
                    try:
                        is_success, frame = cap.read()
                    except cv2.error:
                        continue

                    if not is_success:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    predict_start_time = time.time()
                    bboxes = self.predict(frame_rgb, prob_thresh=prob_thresh)
                    predict_exec_time = time.time() - predict_start_time

                    image = self.draw_bboxes(frame, bboxes)
                    curr_time = time.time()

                    cv2.putText(
                        image,
                        "FPS - predict(): {:.1f}, inference(): {:.1f}".format(
                            1 / predict_exec_time,
                            1 / (curr_time - prev_time),
                        ),
                        org=(5, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6,
                        color=(50, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                    prev_time = curr_time

                    cv2.imshow("result", image)
                    if cv2.waitKey(cv_waitKey_delay) & 0xFF == ord("q"):
                        break

            cap.release()

        print("YOLOv4: Inference is finished")
        #while cv2.waitKey(10) & 0xFF != ord("q"):
            #pass
        #cv2.destroyWindow("result")
