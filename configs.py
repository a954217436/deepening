#- * - coding: UTF - 8 -*-
from lib.task_config import TASK_DICT

PORT = 7788

ocr_path   = "/home/robot/ocr.jpg"    # task img 
work_dir   = "/home/robot/gfb/"
task_path  = work_dir + "log/"        # log and img_result
model_path = work_dir + "models/"

YOLO_NMS  = 0.30                      # NMS置信度阈值，过滤掉相交区域大于该值的两个目标
YOLO_SIZE = (608, 608)                # [320， 416， 608], must be 32*N
FACE_SIZE = (640, 360)                # width, height

INIT_TASK = 'Mask'                    # ['SecurityControl', 'ForeignBody', 'OilLeakage', 'EquipmentDefect', 'SubstationDefect', 'HELMET', 'Mask']

result_show = False                   # 是否显示图片
result_save = True                    # 是否保存图片（默认保存在task_path下）
save_recent = 100                     # 保存最近巡检的图片数量


