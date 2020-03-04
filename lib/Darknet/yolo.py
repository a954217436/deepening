#- * - coding: UTF - 8 -*-
import cv2
import time
import numpy as np

from .. import Detector
from ..logger import logger
from ..task_config import TASK_DICT



class Yolo(Detector.Detector):
    def __init__(self, 
                 model_dir = "/home/robot/gfb/models",
                 image_resize  = (608, 608),
                 task     = 'SecurityControl',
                 nmsThreshold  = 0.30, 
                ):

        self.image_resize = image_resize
        self.nmsThreshold = nmsThreshold
        self.frame = None
        self.task = task
        self.network = 'yolo'

        if task in TASK_DICT.keys():
            self.MODEL = TASK_DICT[task]['MODEL']
            self.confThreshold = TASK_DICT[task]['THRES']
            self.classes_keep = TASK_DICT[task]['CLASSES']
        else:
            logger("Unknown task type: %s !!!"%task)
            return
            #self.MODEL = 'YCXW'
            #self.confThreshold = 0.5
            #self.classes_keep = []
        
        logger("Initialize Model: {}, confThres = {}, nmsThres = {}, img_size = {}, classes = {}".format(self.MODEL,
                                                                                                      self.confThreshold,
                                                                                                      self.nmsThreshold,
                                                                                                      self.image_resize,
                                                                                                      self.classes_keep))
        try:
            self.net = cv2.dnn.readNetFromDarknet(model_dir + '/%s/yolo.cfg'%self.MODEL, 
                                                  model_dir + '/%s/yolo.weights'%self.MODEL)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)    #cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)         #Intel GPU：cv2.dnn.DNN_TARGET_OPENCL
            self.classnames = open(model_dir + '/%s/yolo.names'%self.MODEL, "rt").read().rstrip("\n").split("\n")
        except:
            self.net = None
            self.classnames = None
            logger('###  Yolo Initialize Failed... Please Check the Model Dir: %s/%s/'%(model_dir, self.MODEL))


    def update_class(self, NEW_TASK):
        logger('Update From <{}-{}> to <{}-{}>...'.format(self.classes_keep, self.confThreshold, 
                                                       TASK_DICT[NEW_TASK]['CLASSES'], TASK_DICT[NEW_TASK]['THRES']))
        self.task = NEW_TASK
        self.classes_keep = TASK_DICT[NEW_TASK]['CLASSES']
        self.confThreshold = TASK_DICT[NEW_TASK]['THRES']


    def update(self, taskType):
        if TASK_DICT[taskType]['MODEL'] == self.MODEL:
            self.update_class(taskType)
            return
        else:
            logger('Switching Model From %s to %s...'%(self.MODEL, TASK_DICT[taskType]['MODEL']))
            self.__init__(task = taskType)
            if self.net:
                logger('Switching Model Success!')
            else:
                logger('###  Switching Model Failed!')


    def detect_img(self, img_path, taskType):
        # img_path: /home/robot/task/ocr.jpg 
        # return：拼接后上报的字符串
        if self.task != taskType:
            self.update(taskType)
        if self.net is None:  return ""
        start = time.time()
        self.frame = cv2.imread(img_path)
        if self.frame is None:
            logger('###  None: Could not Open Image: %s!'%img_path)
            return ""
        blob = cv2.dnn.blobFromImage(self.frame, 1/255, self.image_resize, [0,0,0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self._getOutputsNames())
        result_list = self._postprocess(outs)
        result_str  = self._to_lxywh_str(result_list)
        logger("<<< Inference time: {:.2f} s... >>>".format(time.time() - start))
        return result_str


    def _postprocess(self, outs):
        # Remove the bounding boxes with low confidence using non-maxima suppression
        frameHeight = self.frame.shape[0]
        frameWidth = self.frame.shape[1]

        classIds, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        
        # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)

        result_str = ''
        result_list = []
        for i in indices:
            i = i[0]
            box = np.array(boxes[i])
            box[box < 0] = 0
            x, y, width, height = box[0], box[1], box[2], box[3]
            label = self.classnames[classIds[i]]
            if label not in self.classes_keep:    #过滤
                continue
            obj = [label, x, y, width, height]      
            #logger('< {0:8}, {1:.4f},  {2:4} {3:4} {4:4} {5:4} >'.format(obj[0], confidences[i], x, y, width, height), t=False)    
            result_list.append(obj)
            self._drawPred(classIds[i], confidences[i] * 100, x, y, x + width, y + height)
            
        #logger("<<< %d objs found... >>>"%(len(result_list)), t=False)
    
        return result_list


    def _getOutputsNames(self):
        # Get the names of the net output layers
        layersNames = self.net.getLayerNames()
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


if __name__=="__main__":
    det = Yolo()
    result = det.detect_img('./test.jpg', "SecurityControl")
    print(det.MODEL, det.task, result)
    det.show()
    det.save("yolo2.png")
