from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yacs

from yacs.config import CfgNode as CN

# from .models import MODEL_EXTRAS

_C = CN()
_C.AUTO_RESUME = True
_C.GPUS = 0,

_C.RANK = 0
_C.DIST_URL = 'tcp://127.0.0.1:23456'
_C.WORLD_SIZE = 1
_C.WORKERS = 16

_C.VERBOSE = True
_C.OUTPUT_DIR = ''
# _C.DATA_DIR = ''

############################################################################
_C.LOG = CN()
_C.LOG.CFG_NAME = 'config_name.yaml'
_C.LOG.LOG_DIR = 'log'
_C.LOG.LOGGING_ROOT = ''                # Path to directory where to write tensorboard logs and checkpoints
_C.LOG.PRINT_FREQ = 100
_C.LOG.CHECKPOINT_FREQ = 100

############################################################################
# DATASET related params
_C.DATASET = CN()
_C.DATASET.DATASET = 'realdome_cx'
_C.DATASET.GEN_TEX = False
_C.DATASET.TEX_INTERPOLATER = 'nearest'

_C.DATASET.ROOT = 'data/realdome_cx'    # Root folder for img_dir and mesh_dir
_C.DATASET.FRAME_RANGE = [-1,-1]
_C.DATASET.CAM_RANGE = [-1,-1]
_C.DATASET.IMG_DIR = '_/rgb0/%03d.png'
_C.DATASET.MESH_DIR = '_/mesh/%03d.obj'
_C.DATASET.GAMMA = 1.0
_C.DATASET.OUTPUT_SIZE = [512,512]      # Sidelength of generated images. Only less than native resolution of images is recommended
_C.DATASET.CALIB_PATH = '_/test_calib53/calib20200619_test_mid_53.mat'  # Path of calibration file for inference sequence                                           
_C.DATASET.CALIB_FORMAT = 'convert'
_C.DATASET.CAM_MODE = 'projection' # projection or orthogonal
# 3D computing
_C.DATASET.PRELOAD_MESHS = False
_C.DATASET.PRELOAD_VIEWS = False
_C.DATASET.TEX_PATH = ''
_C.DATASET.UV_PATH = ''                 # Preset uv for all frame

_C.DATASET.UV_CONVERTER = './data/UVTextureConverter/densepose_to_SMPL_fix.npy'                 # Preset uv for all frame

# Relight params
# _C.DATASET.LIGHTING_IDX = 0

# Training data augmentation
_C.DATASET.MAX_SHIFT = 0
_C.DATASET.MAX_ROTATION = 0
_C.DATASET.MAX_SCALE = 1.0
_C.DATASET.FLIP = 0.5

# # dataset transform parameters for densepose dataset
# _C.DATASET.PREPROCESS_MODE = '' # to-do
# _C.DATASET.CROP_SIZE = 512      # merge with _C.DATASET.OUTPUT_SIZE
# _C.DATASET.LOAD_SIZE = 512      # merge with _C.DATASET.OUTPUT_SIZE
# _C.DATASET.ASPECT_RATIO = 1.0   # merge with _C.DATASET.MAX_SCALE
# _C.DATASET.NO_FLIP = True       # merge with _C.DATASET.FLIP

# # train test split
# _C.DATASET.TEST_SET = ''
# _C.DATASET.TRAIN_SET = ''
############################################################################
# DATASET related params
_C.DATASET_FVV = CN()
_C.DATASET_FVV.DATASET = 'realdome_cx'
_C.DATASET_FVV.ROOT = 'data/realdome_cx'    # Root folder for img_dir and mesh_dir
_C.DATASET_FVV.FRAME_RANGE = [-1,-1]
_C.DATASET_FVV.CAM_RANGE = [-1,-1]
_C.DATASET_FVV.IMG_DIR = '_/rgb0/%03d.png'
_C.DATASET_FVV.MESH_DIR = '_/mesh/%03d.obj'
_C.DATASET_FVV.OUTPUT_SIZE = [512,512]
_C.DATASET_FVV.CALIB_PATH = '_/test_calib53/calib20200619_test_mid_53.mat'
_C.DATASET_FVV.CALIB_FORMAT = 'convert'
_C.DATASET_FVV.CAM_MODE = 'projection' # projection or orthogonal
# 3D computing
_C.DATASET_FVV.PRELOAD_MESHS = False
_C.DATASET_FVV.PRELOAD_VIEWS = False
_C.DATASET_FVV.TEX_PATH = ''
_C.DATASET_FVV.UV_PATH = ''                 # Preset uv for all frame
_C.DATASET_FVV.UV_CONVERTER = './data/UVTextureConverter/densepose_to_SMPL_fix.npy'                 # Preset uv for all frame
# Training data augmentation
_C.DATASET_FVV.MAX_SHIFT = 0
_C.DATASET_FVV.MAX_ROTATION = 0
_C.DATASET_FVV.MAX_SCALE = 1.0
_C.DATASET_FVV.FLIP = 0.5
############################################################################
# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.VGG_PATH = './models/vgg19-dcbb9e9d.pth'
_C.MODEL.NAME = 'RenderNet'
# _C.MODEL.SYNC_BN = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.TEX_CREATER = CN()
_C.MODEL.TEX_CREATER.NUM_CHANNELS = 3   # Num of channels for orig texture
_C.MODEL.TEX_CREATER.NUM_SIZE = 1024
_C.MODEL.TEX_MAPPER = CN()
_C.MODEL.TEX_MAPPER.NUM_CHANNELS = 24   # Num of channels for neural texture
_C.MODEL.TEX_MAPPER.NUM_SIZE = 1024
_C.MODEL.TEX_MAPPER.MIPMAP_LEVEL = 4    # Mipmap levels for neural texture
_C.MODEL.TEX_MAPPER.SH_BASIS = True     # Whether apply spherical harmonics to sampled feature maps
_C.MODEL.TEX_MAPPER.MERGE_TEX = True        # Whether merge texture different training
_C.MODEL.TEX_MAPPER.NUM_PARAMS = -1
_C.MODEL.FEATURE_MODULE = CN()
_C.MODEL.FEATURE_MODULE.NF0 = 64            # Number of features in outermost layer of U-Net architectures
_C.MODEL.FEATURE_MODULE.NUM_DOWN = 5
_C.MODEL.FEATURE_MODULE.NUM_PARAMS = -1
_C.MODEL.RENDER_MODULE = CN()
_C.MODEL.RENDER_MODULE.NF0 = 64             # Number of features in outermost layer of U-Net architectures
_C.MODEL.RENDER_MODULE.NUM_DOWN = 5
_C.MODEL.RENDER_MODULE.OUTPUT_CHANNELS =3
_C.MODEL.RENDER_MODULE.NUM_PARAMS = -1
_C.MODEL.ALIGN_MODULE = CN()
_C.MODEL.ALIGN_MODULE.MID_CHANNELS = 64

_C.MODEL.GAN = CN()
_C.MODEL.GAN.MODE = 'vanilla'
_C.MODEL.GAN.LAMBDA_L1 = 100.0              # weight for L1 loss, the training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1

_C.MODEL.NET_D = CN()
_C.MODEL.NET_D.ARCH = 'multiscale'               # Specify discriminator architecture [basic | n_layers | pixel | multiscale]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator
_C.MODEL.NET_D.INPUT_CHANNELS = 6           # 6 = 3 + 3 (fake img + real img)
# _C.MODEL.NET_D.OUTPUT_CHANNELS = 3
_C.MODEL.NET_D.N_LAYERS_D = 3               # Only used if netD==n_layers
_C.MODEL.NET_D.NDF = 64                     # Number of discrim filters in the first conv layer
_C.MODEL.NET_D.NORM = 'batch'               # Instance normalization or batch normalization [instance | batch | none]
_C.MODEL.NET_D.INIT_TYPE = 'normal'         # Network initialization [normal | xavier | kaiming | orthogonal]
_C.MODEL.NET_D.INIT_GAIN = 0.02             # Scaling factor for normal, xavier and orthogonal

############################################################################
_C.TRAIN = CN()
_C.TRAIN.EXP_NAME = 'example'
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.RESUME = False
_C.TRAIN.SAMPLING_PATTERN = 'all' # skipinv_10          # Sampling of image training sequences_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.SAMPLING_PATTERN_VAL = 'all' # skipinv_10
_C.TRAIN.SAMPLING_PAIRWISE = False
_C.TRAIN.SAMPLING_PAIRMODE = 'all'
_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.CHECKPOINT_DIR = ''
_C.TRAIN.CHECKPOINT_NAME = ''
_C.TRAIN.END_EPOCH = 10000
_C.TRAIN.GAMMA = 1.0
_C.TRAIN.SHUFFLE = True
_C.TRAIN.VAL_FREQ = -1                          # Test on validation data every X iterations
_C.TRAIN.LR = 0.001 
_C.TRAIN.LR_MODE = 'multistep'
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.OPTIMIZER = 'adam'
# _C.TRAIN.MOMENTUM = 0.9
# _C.TRAIN.WD = 0.0001

############################################################################
_C.LOSS = CN()
_C.LOSS.PERCEPTUALLOSS = "L1"
_C.LOSS.WEIGHT_GAN_G = 1.0
_C.LOSS.WEIGHT_PERCEPTUAL = 1.0
_C.LOSS.WEIGHT_HSV = 1.0
_C.LOSS.WEIGHT_ATLAS = 1.0
_C.LOSS.WEIGHT_ATLAS_REF = 0.1
_C.LOSS.WEIGHT_ATLAS_UNIFY = 0.01
_C.LOSS.WEIGHT_VIEWS = 0.01

############################################################################
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 3
_C.TEST.FRAME_RANGE = [0,100]
_C.TEST.CALIB_PATH = '_/calib.mat'
_C.TEST.CALIB_DIR = ''
_C.TEST.CALIB_NAME = ''
_C.TEST.SAMPLING_PATTERN = 'all'            # Sampling of image testing sequences
_C.TEST.MODEL_PATH = 'models/dataset_name/net_name/date_net_params.pth'    # Path to a checkpoint to load render_module weights from
_C.TEST.MODEL_DIR = ''
_C.TEST.MODEL_NAME = ''
_C.TEST.SAVE_FOLDER = 'img_test'            # Save folder for test imgs
#_C.TEST.SCALE_FACTOR = [1]
_C.TEST.LOG_PROGRESS = False

############################################################################
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_TRANSFORMED_IMG = False
_C.DEBUG.SAVE_TRANSFORMED_MASK = False
_C.DEBUG.SAVE_NEURAL_TEX = False
# _C.DEBUG.SAVE_BATCH_IMAGES_GT = False
# _C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
# _C.DEBUG.SAVE_HEATMAPS_GT = True
# _C.DEBUG.SAVE_HEATMAPS_PRED = True
# _C.DEBUG.SAVE_TAGMAPS_PRED = True

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    (cfg_path, cfg_name) = os.path.split(args.cfg)

    if cfg.TRAIN.CHECKPOINT and cfg.TRAIN.CHECKPOINT[:2] == '_/':
        cfg.TRAIN.CHECKPOINT = os.path.join(cfg.DATASET.ROOT, cfg.TRAIN.CHECKPOINT[2:])
    if cfg.TRAIN.CHECKPOINT:    
        cfg.TRAIN.CHECKPOINT_DIR = cfg.TRAIN.CHECKPOINT.split('/')[-2]
        cfg.TRAIN.CHECKPOINT_NAME = cfg.TRAIN.CHECKPOINT.split('/')[-1]

    # Add path under root folder
    if cfg.TEST.CALIB_PATH and cfg.TEST.CALIB_PATH[:2] == '_/':
        cfg.TEST.CALIB_PATH = os.path.join(cfg.DATASET.ROOT, cfg.TEST.CALIB_PATH[2:])
    cfg.TEST.FRAME_RANGE = set_range(cfg.TEST.FRAME_RANGE)
    
    # find test model
    if not os.path.exists(cfg.TEST.MODEL_PATH):
        cfg.TEST.MODEL_PATH = os.path.join(cfg_path, cfg.TEST.MODEL_PATH)

    if cfg.TEST.MODEL_PATH and cfg.TEST.MODEL_PATH[:2] == '_/':
        cfg.TEST.MODEL_PATH = os.path.join(cfg.DATASET.ROOT, cfg.TEST.MODEL_PATH[2:])
    (cfg.TEST.CALIB_DIR, cfg.TEST.CALIB_NAME) = os.path.split(cfg.TEST.CALIB_PATH)
    (cfg.TEST.MODEL_DIR, cfg.TEST.MODEL_NAME) = os.path.split(cfg.TEST.MODEL_PATH)
    if cfg.TEST.MODEL_DIR == '' or cfg.TEST.MODEL_DIR == './' :
        (cfg.TEST.MODEL_DIR, _) = os.path.split(args.cfg)        

    if cfg.DATASET.IMG_DIR[:2] == '_/':
        cfg.DATASET.IMG_DIR = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.IMG_DIR[2:])
    if cfg.DATASET.CALIB_PATH[:2] == '_/':
        cfg.DATASET.CALIB_PATH = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.CALIB_PATH[2:])
    if cfg.DATASET.MESH_DIR[:2] == '_/':
        cfg.DATASET.MESH_DIR = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.MESH_DIR[2:])
    if cfg.DATASET.CALIB_PATH[:2] == '_/':
        cfg.DATASET.CALIB_PATH = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.CALIB_PATH[2:])
        
    if not cfg.DATASET.TEX_PATH and cfg.DATASET.TEX_PATH[:2] == '_/':
        cfg.DATASET.TEX_PATH = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.TEX_PATH[2:])
        cfg.LOG.LOGGING_ROOT = os.path.join(cfg.DATASET.ROOT, 'logs')

    if not cfg.LOG.LOGGING_ROOT:
        cfg.LOG.LOGGING_ROOT = os.path.join(cfg.DATASET.ROOT, 'logs')

    if cfg.MODEL.PRETRAINED and cfg.MODEL.PRETRAINED[:2] == '_/':
        cfg.MODEL.PRETRAINED = os.path.join(cfg.DATASET.ROOT, cfg.MODEL.PRETRAINED[2:])
    if not os.path.exists(cfg.TRAIN.CHECKPOINT):
        cfg.TRAIN.CHECKPOINT = os.path.join(cfg_path, cfg.TRAIN.CHECKPOINT)

    # # Precompute save folder
    # cfg.DATASET.MESH_PATTERN = cfg.DATASET.MESH_DIR.split('/')[-1].split('.')[0]
    # cfg.DATASET.PRECOMP_DIR = os.path.join(cfg.DATASET.ROOT, 'precomp_' + cfg.DATASET.MESH_PATTERN)

    # Precompute frame camera range
    cfg.DATASET.CAM_RANGE = set_range(cfg.DATASET.CAM_RANGE)
    cfg.DATASET.FRAME_RANGE = set_range(cfg.DATASET.FRAME_RANGE)

    if not isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)):
        cfg.DATASET.OUTPUT_SIZE = [cfg.DATASET.OUTPUT_SIZE]

    ###############################################################
    # DATASET_FVV
    if cfg.DATASET_FVV.IMG_DIR[:2] == '_/':
        cfg.DATASET_FVV.IMG_DIR = os.path.join(cfg.DATASET_FVV.ROOT, cfg.DATASET_FVV.IMG_DIR[2:])
    if cfg.DATASET_FVV.CALIB_PATH[:2] == '_/':
        cfg.DATASET_FVV.CALIB_PATH = os.path.join(cfg.DATASET_FVV.ROOT, cfg.DATASET_FVV.CALIB_PATH[2:])
    if cfg.DATASET_FVV.MESH_DIR[:2] == '_/':
        cfg.DATASET_FVV.MESH_DIR = os.path.join(cfg.DATASET_FVV.ROOT, cfg.DATASET_FVV.MESH_DIR[2:])
    if cfg.DATASET_FVV.CALIB_PATH[:2] == '_/':
        cfg.DATASET_FVV.CALIB_PATH = os.path.join(cfg.DATASET_FVV.ROOT, cfg.DATASET_FVV.CALIB_PATH[2:])      
    cfg.DATASET_FVV.CAM_RANGE = set_range(cfg.DATASET_FVV.CAM_RANGE)
    cfg.DATASET_FVV.FRAME_RANGE = set_range(cfg.DATASET_FVV.FRAME_RANGE)

    cfg.freeze()

def check_config(cfg):
    pass

def set_range(range_params):
    if type(range_params) != list:
        raise ValueError('Error! range value must be list')

    if len(range_params) == 1:
        range_params = range_params
    elif len(range_params) == 2:
        range_params = list(range(range_params[0], range_params[1]))
    elif len(range_params) > 2:
        pass
        # print(range_params)
        # raise ValueError('Error! range value is not support')
    return range_params

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
