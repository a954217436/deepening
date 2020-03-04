TASK_DICT = {}
TASK_DICT['SecurityControl'] = {
    'NET'    : 'yolo',
    'MODEL'  : 'YCXW',
    'CLASSES': ['wcgz', 'wcaqm', 'xy', 'rydd'],
    'THRES'  : 0.50,}

TASK_DICT['ForeignBody'] = {
    'NET'    : 'yolo',
    'MODEL'  : 'GFB8',
    'CLASSES': ['yw_gkxfw', 'yw_nc'],
    'THRES'  : 0.25,}

TASK_DICT['OilLeakage'] = {
    'NET'    : 'yolo',
    'MODEL'  : 'GFB8',
    'CLASSES': ['sly_dmyw', 'sly_bjbmyw'],
    'THRES'  : 0.15,}

TASK_DICT['EquipmentDefect'] = {
    'NET'    : 'yolo',
    'MODEL'  : 'GFB8',
    'CLASSES': ['xmbhyc', 'bj_bpps', 'hxq_gjbs', 'bj_bpmh'],
    'THRES'  : 0.30,}

TASK_DICT['SubstationDefect'] = {
    'NET'    : 'yolo',
    'MODEL'  : 'CNQX',
    'CLASSES': [],
    'THRES'  : 0.50,}

TASK_DICT['HELMET'] = {
    'NET'    : 'yolo',
    'MODEL'  : 'AQM',
    'CLASSES': ['wcaqm'],
    'THRES'  : 0.50,}

TASK_DICT['Mask'] = {
    'NET'             : 'centerface',
    'MODEL'           : 'MASK',
    'CLASSES'         : ['mask', 'nomask'],
    'FACE_THRES'      : 0.50,
    'CLASSIFY_THRES'  : 0.50,}
