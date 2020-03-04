import cv2


class Detector(object):
    def __init__(self):
        pass
    

    def _to_lxywh_str(self, result_list):
        result_str = ''
        result_list = sorted(result_list, key=(lambda x:(x[1], x[2])))    #按照先x后y排序
        for r in result_list:
            r_str = [str(m) for m in r]
            result_str += ' '.join(r_str) + ' '
        return result_str

    
    def _drawPred(self, classId, conf, left, top, right, bottom):
        if self.frame is None:
            return
        # Draw the predicted bounding box
        h, w = self.frame.shape[:2]
        font_size = 0.5 if w / 3600 < 0.5 else w / 3600
        font_bold = 2 if font_size > 1 else 1
        label, color = self.classnames[classId], (0,0,255) #self._colors[classId]
        cv2.rectangle(self.frame, (left, top), (right, bottom), color, 4)
        text = '{0:} {1:.0f}%'.format(label, conf)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_bold)
        top = max(top, labelSize[1])
        cv2.rectangle(self.frame, (left, top - round(1.3*labelSize[1])), (left + round(labelSize[0]), top + baseLine), (0, 0, 0), cv2.FILLED)
        cv2.putText(self.frame, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, font_size, (180,180,80), font_bold)
    
    
    def save(self, save_path):
        if self.frame is None:
            return
        cv2.imwrite(save_path, self.frame)
        
    
    def show(self, winName="Result"):
        if self.frame is None:
            return
        cv2.imshow(winName, self.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

    def detect_img(self, img_path, taskType=""):
        raise NotImplementedError("Not Surpported!")
