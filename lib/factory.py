#- * - coding: UTF - 8 -*-
from configs import *

from .Darknet.yolo import Yolo
from .CenterFace.mask import Mask


def detector_factory(task):
    #task: 'SecurityControl'
    net = TASK_DICT[task]['NET']
    if   net == 'yolo':
        return Yolo(model_dir = model_path, image_resize = YOLO_SIZE, task=task, nmsThreshold=YOLO_NMS)
    elif net == 'centerface':
        return Mask(model_dir = model_path, image_resize = FACE_SIZE)
    else:
        log("###  Unrecognized task name: %s!!! Switch to SecurityControl..."%task)
        return Yolo()
