#- * - coding: UTF - 8 -*-
import os
import time
import logging
from configs import *


def init_log(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y%m%d_%H%M%S'))

    logging.basicConfig(level = logging.DEBUG,
                        format = '%(asctime)s %(message)s',
                        datefmt = '%Y-%m-%d %H-%M-%S',
                        filename = os.path.join(output_dir, log_name),
                        filemode = 'w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    return logging


log = init_log(task_path)
logger = log.info
logger("Log dir : " + task_path)



'''def get_time_str():
    #获取并格式化系统时间
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())


log_name = '/' + get_time_str() + '.log'

def log(info, t=True, p=True):
    #t:是否记录时间    p:是否打印在终端
    if p:
        print(info)
    with open(task_path + log_name, 'a') as lg:
        t = (get_time_str()+":    ") if t else " "*24
        lg.write(t + info + '\n')'''