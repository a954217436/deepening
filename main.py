#- * - coding: UTF - 8 -*-
import os
import socket
import cv2


from configs import *
from lib.logger import logger
from lib.factory import detector_factory


def detect(detector, img_path, taskType, tempPrefix):
    resultstr = detector.detect_img(img_path, taskType)
    if detector.frame is None:
        return resultstr
    if result_save:
        save_path = tempPrefix + '_result.jpg' if os.path.isfile(tempPrefix+'.jpg') else task_path + tempPrefix.split('/')[-1] + '_result.jpg'
        detector.save(save_path)
    if result_show:
        detector.show()
    return resultstr


def process_Received(info):    
    # info : 'recognize /home/robot/historyData/123/colorful/456/456_colorful OilLeakage' 
    global Detector
    resultstr = 'error'
    if info == 'serverstatus':
        logger('***  resultstr:  serverready')
        return 'serverready'   
    elif 'recognize' in info and len(info.split(' '))==3:
        tempPrefix, taskType = info.split(' ')[1], info.split(' ')[2]
        if not taskType in TASK_DICT.keys():
            logger("###  Unrecognized taskType")
            return 'empty'
        net_name = TASK_DICT[taskType]['NET']
        if Detector.network != net_name:
            logger("Switching NetWork from %s to %s"%(Detector.network, net_name))
            del Detector
            Detector = detector_factory(taskType)
            
        resultstr = detect(Detector, ocr_path, taskType, tempPrefix)
        if resultstr == "":
            resultstr = "empty"
    else:
        logger('###  Unrecognized command')

    return resultstr


def main():
    #启动Socket
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', PORT))
    server.listen(5)
    
    logger("==================Loading Success! Start Socket==================")
    while True:
        logger('===Server Listening===\n')
        conn, address = server.accept()
        info = conn.recv(2048).decode()
        logger('***  receive from {}, message : {}'.format(address[0], info))
        resultstr = process_Received(info)
        conn.sendall(resultstr.encode())
        logger("***  Sendall resultstr: " + resultstr)
        logger('- '*80 + '\n')
     
    
if __name__ == "__main__":
    Detector = detector_factory(INIT_TASK)
    main()
