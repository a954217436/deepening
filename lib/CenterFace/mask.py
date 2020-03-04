import time
import cv2
import numpy as np

from .centerface import CenterFace

from .. import Detector
from ..logger import logger
from ..task_config import TASK_DICT


def log_softmax(input):
    '''
    # ZhangHao, replace torch.nn.functional.log_softmax()
        input  :  [ 3.1320703 -3.131977 ]
        output :  [-1.9017245e-03 -6.2659492e+00]
    '''
    x = np.array(input)
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp)
    s = x_exp / x_sum
    result = np.log(s)
    return result


class Mask(Detector.Detector):
    def __init__(self, 
                 model_dir    =  "/home/robot/gfb/models", 
                 image_resize =  (640, 360)):

        self.image_resize = image_resize    #(im_width, im_height)
        self.classify_thres = TASK_DICT['Mask']['CLASSIFY_THRES']
        self.face_thres = TASK_DICT['Mask']['FACE_THRES']
        self.classnames = TASK_DICT['Mask']['CLASSES']
        self.frame = None
        self.network = 'centerface'

        logger("Initialize Centerface, face_thres = {}, classify_thres = {}, face_thres = {}, img_size = {}".format(self.face_thres,
                                                                        self.classify_thres,
                                                                        self.face_thres,
                                                                        self.image_resize,))
        try:
            self.classifier = cv2.dnn.readNetFromONNX(model_dir + '/MASK/sbd_mask.onnx')
            self.centerface = CenterFace(height=image_resize[1], width=image_resize[0], landmarks=True, model_dir=model_dir)
        except:
            self.classifier = None
            self.centerface = None
            logger('###  Centerface Initialize Failed... Please Check the Model Dir: %s/%s/'%(model_dir, self.MODEL))



    def _classify(self, img_arr=None):
        blob = cv2.dnn.blobFromImage(img_arr, 
                                     scalefactor=1 / 255, 
                                     size=(224, 224), 
                                     mean=(0, 0, 0),
                                     swapRB=True, crop=False)
        self.classifier.setInput(blob)
        heatmap = self.classifier.forward(['349'])
        match = log_softmax(heatmap[0][0])
        index = np.argmax(match)
        return (0 if index > self.classify_thres else 1, match[0])
    
    
    def _classify_boxes(self, bboxs, w_scale, h_scale, thresh=0.5, max_size=0):
        #bboxs = bboxs[np.where(bboxs[:, -1] > thresh)[0]]
        bboxs = bboxs.astype(int)
        results = []
        for bbox in bboxs:
            bbox[0], bbox[2] = int(bbox[0]/w_scale), int(bbox[2]/w_scale)
            bbox[1], bbox[3] = int(bbox[1]/h_scale), int(bbox[3]/h_scale)
            img_bbox = self.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if img_bbox.shape[0] * img_bbox.shape[1] < max_size:
                continue
            (ftype, prob) = self._classify(img_arr=img_bbox)
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            results.append([self.classnames[ftype], x, y, w, h])
            self._drawPred(ftype, 100, x, y, bbox[2], bbox[3])
        return results


    def detect_img(self, img_path="/home/robot/ocr.jpg", taskType="Mask"):
        if taskType != "Mask":    return ""
        self.frame = cv2.imread(img_path)
        if self.frame is None:
            logger('###  None: Could not Open Image: %s!'%img_path)
            return None
        start = time.time()
        height, width = self.frame.shape[0], self.frame.shape[1]
        frame_resized = cv2.resize(self.frame, self.image_resize)
        w_scale, h_scale = self.image_resize[0] / width, self.image_resize[1] / height
        
        dets, lms = self.centerface(frame_resized, self.face_thres)
        result_list = self._classify_boxes(dets, w_scale, h_scale)
        result_str  = self._to_lxywh_str(result_list)
        logger("<<< Inference time: {:.2f} s... >>>".format(time.time() - start))
        #print(result_str)
        return result_str


'''if __name__ == "__main__":
    im_width = 640 #640
    im_height = 360 #360
    mask = Mask((im_width, im_height))
    mask.detect_img("44.jpg")
    mask.show()'''
    
