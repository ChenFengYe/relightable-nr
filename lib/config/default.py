from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

# from .models import MODEL_EXTRAS

# _C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 8
_C.AUTO_RESUME = False
_C.RANK = 0
# _C.VERBOSE = True

# Cuda related params
_C.CUDA = CN()
_C.CUDA.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'dnr_unet_512_tex1024'
_C.MODEL.IMAGE_SIZE = 512
_C.MODEL.TEX_SIZE = 1024

_C.LOSS = CN()

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'coco_kpt'
_C.DATASET.DATASET_TEST = 'coco'
_C.DATASET.MULTI_FRAME = True
_C.DATASET.LOG_DIR = ''
_C.DATASET.TRAIN = 'train2017'
_C.DATASET.TEST = 'val2017'

# training data augmentation
_C.DATASET.MAX_ROTATION = 30
_C.DATASET.MIN_SCALE = 0.75
_C.DATASET.MAX_SCALE = 1.25
_C.DATASET.SCALE_TYPE = 'short'
_C.DATASET.MAX_TRANSLATE = 40
_C.DATASET.INPUT_SIZE = 512
_C.DATASET.OUTPUT_SIZE = [128, 256, 512]
# _C.DATASET.FLIP = 0.5

# heatmap generator (default is OUTPUT_SIZE/64)
# _C.DATASET.SIGMA = -1
# _C.DATASET.SCALE_AWARE_SIGMA = False
# _C.DATASET.BASE_SIZE = 256.0
# _C.DATASET.BASE_SIGMA = 2.0
# _C.DATASET.INT_SIGMA = False

# train
_C.TRAIN = CN()

# _C.TRAIN.LR_FACTOR = 0.1
# _C.TRAIN.LR_STEP = [90, 110]
# _C.TRAIN.LR = 0.001

# _C.TRAIN.OPTIMIZER = 'adam'
# _C.TRAIN.MOMENTUM = 0.9
# _C.TRAIN.WD = 0.0001
# _C.TRAIN.NESTEROV = False
# _C.TRAIN.GAMMA1 = 0.99
# _C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 10000

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.IMAGES_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE = 3
# Test Model Epoch
#_C.TEST.SCALE_FACTOR = [1]
_C.TEST.DETECTION_THRESHOLD = 0.2
_C.TEST.TAG_THRESHOLD = 1.
_C.TEST.USE_DETECTION_VAL = True
_C.TEST.IGNORE_TOO_MUCH = False
_C.TEST.MODEL_FILE = ''
_C.TEST.IGNORE_CENTER = True
_C.TEST.NMS_KERNEL = 3
_C.TEST.NMS_PADDING = 1
_C.TEST.PROJECT2IMAGE = False

_C.TEST.WITH_HEATMAPS = (True,)
_C.TEST.WITH_AE = (True,)

_C.TEST.LOG_PROGRESS = False

# # debug
# _C.DEBUG = CN()
# _C.DEBUG.DEBUG = True
# _C.DEBUG.SAVE_BATCH_IMAGES_GT = False
# _C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
# _C.DEBUG.SAVE_HEATMAPS_GT = True
# _C.DEBUG.SAVE_HEATMAPS_PRED = True
# _C.DEBUG.SAVE_TAGMAPS_PRED = True

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if not os.path.exists(cfg.DATASET.ROOT):
        cfg.DATASET.ROOT = os.path.join(
            cfg.DATA_DIR, cfg.DATASET.ROOT
        )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    if cfg.DATASET.WITH_CENTER:
        cfg.DATASET.NUM_JOINTS += 1
        cfg.MODEL.NUM_JOINTS = cfg.DATASET.NUM_JOINTS

    if not isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)):
        cfg.DATASET.OUTPUT_SIZE = [cfg.DATASET.OUTPUT_SIZE]
    if not isinstance(cfg.LOSS.WITH_HEATMAPS_LOSS, (list, tuple)):
        cfg.LOSS.WITH_HEATMAPS_LOSS = (cfg.LOSS.WITH_HEATMAPS_LOSS)

    if not isinstance(cfg.LOSS.HEATMAPS_LOSS_FACTOR, (list, tuple)):
        cfg.LOSS.HEATMAPS_LOSS_FACTOR = (cfg.LOSS.HEATMAPS_LOSS_FACTOR)

    if not isinstance(cfg.LOSS.WITH_AE_LOSS, (list, tuple)):
        cfg.LOSS.WITH_AE_LOSS = (cfg.LOSS.WITH_AE_LOSS)

    if not isinstance(cfg.LOSS.PUSH_LOSS_FACTOR, (list, tuple)):
        cfg.LOSS.PUSH_LOSS_FACTOR = (cfg.LOSS.PUSH_LOSS_FACTOR)

    if not isinstance(cfg.LOSS.PULL_LOSS_FACTOR, (list, tuple)):
        cfg.LOSS.PULL_LOSS_FACTOR = (cfg.LOSS.PULL_LOSS_FACTOR)

    cfg.freeze()


def check_config(cfg):
    #assert cfg.LOSS.NUM_STAGES == len(cfg.LOSS.WITH_HEATMAPS_LOSS), \
    #    'LOSS.NUM_SCALE should be the same as the length of LOSS.WITH_HEATMAPS_LOSS'

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)