#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from picamera2 import Picamera2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--model_selection",
                        help='model_selection',
                        type=int,
                        default=0)
    parser.add_argument("--score_th",
                        help='score threshold',
                        type=float,
                        default=0.1)

    parser.add_argument("--bg_path",
                        help='back ground image path',
                        type=str,
                        default=None)

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_width = args.width
    cap_height = args.height

    model_selection = args.model_selection
    score_th = args.score_th

    if args.bg_path is not None:
        bg_image = cv.imread(args.bg_path)
    else:
        bg_image = None

    # カメラ準備 ###############################################################
    picam2 = Picamera2()
    try:
        # camver=1 or camver=2
        preview_config = picam2.create_preview_configuration({'format': 'XRGB8888', 'size': (cap_width, cap_height)}, raw=picam2.sensor_modes[3])
    except IndexError:
        try:
            # camver=3
            preview_config = picam2.create_preview_configuration({'format': 'XRGB8888', 'size': (cap_width, cap_height)}, raw=picam2.sensor_modes[2])
        except IndexError:
            preview_config = picam2.create_preview_configuration({'format': 'XRGB8888', 'size': (cap_width, cap_height)})
    picam2.configure(preview_config)
    picam2.start()

    # モデルロード #############################################################
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
        model_selection=model_selection)

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        image = picam2.capture_array()
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = selfie_segmentation.process(image)

        # 描画 ################################################################
        mask = np.stack((results.segmentation_mask, ) * 3, axis=-1) >= score_th

        if bg_image is None:
            bg_resize_image = np.zeros(image.shape, dtype=np.uint8)
            bg_resize_image[:] = (0, 255, 0)
        else:
            bg_resize_image = cv.resize(bg_image,
                                        (image.shape[1], image.shape[0]))
        debug_image = np.where(mask, debug_image, bg_resize_image)

        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27 or key == ord('q'):   # ESC or q
            break

        # 画面反映 #############################################################
        cv.imshow('MediaPipe Selfie Segmentation Demo', debug_image)

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
