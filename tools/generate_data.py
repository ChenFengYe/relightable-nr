import argparse
import os, time

import numpy as np
import cv2
import datetime

import _init_paths

from lib.utils.util import cond_mkdir, make_gif
from lib.config import cfg, update_config
from lib.utils import vis

import torch
import torchvision

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from lib.models import metric
from lib.models.render_net import RenderNet
from lib.models.feature_net import FeatureNet
from lib.models.merge_net import MergeNet
from lib.models.gan_net import Pix2PixModel

from utils.encoding import DataParallelModel

from lib.dataset.DomeViewDataset import DomeViewDataset
from lib.dataset.DPViewDataset import DPViewDataset  

def parse_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    args = parser.parse_args()
    return args

def build_model(args):
    print('Load config...')
    update_config(cfg, args)

    # print("Setup Log ...")
    log_dir = cfg.TEST.CALIB_DIR.split('/')
    log_dir = os.path.join(cfg.TEST.MODEL_DIR, cfg.TRAIN.EXP_NAME+'_'+cfg.TEST.CALIB_NAME[:-4]+'_resol_'+str(cfg.DATASET.OUTPUT_SIZE[0])+'_'+log_dir[-2]+'_'+
                           log_dir[-1].split('_')[0] + '_' + log_dir[-1].split('_')[1] + '_' +
                           cfg.TEST.MODEL_NAME.split('-')[-1].split('.')[0])
    cond_mkdir(log_dir)
    if len(cfg.TEST.FRAME_RANGE) > 1:
        save_dir_img_ext = cfg.TEST.SAVE_FOLDER + '_' + str(cfg.TEST.FRAME_RANGE[0]) +'_'+ str(cfg.TEST.FRAME_RANGE[-1])
    else:
        save_dir_img_ext = cfg.TEST.SAVE_FOLDER
    save_dir_img = os.path.join(log_dir, save_dir_img_ext)
    save_dir_uv = save_dir_img+'_uv'
    save_dir_nimg = save_dir_img+'_nimg'
    cond_mkdir(save_dir_img)
    cond_mkdir(save_dir_uv)
    cond_mkdir(save_dir_nimg)
    
    print("Build dataloader ...")
    view_dataset = eval(cfg.DATASET.DATASET)(cfg, isTrain=False)
    print("*" * 100)

    # print('Build Network...')
    # model_net = eval(cfg.MODEL.NAME)(cfg, isTrain=False)
    # model = model_net
    # model.setup(cfg)
    
    # print('Loading Model...')
    # checkpoint_path = cfg.TEST.MODEL_PATH
    # if os.path.exists(checkpoint_path):
    #     pass
    # elif os.path.exists(os.path.join(cfg.TEST.MODEL_DIR, checkpoint_path)):
    #     checkpoint_path = os.path.join(cfg.TEST.MODEL_DIR,checkpoint_path)

    print('Start buffering data for inference...')
    view_dataloader = DataLoader(view_dataset, batch_size = cfg.TEST.BATCH_SIZE, shuffle = False, num_workers = cfg.WORKERS)
    view_dataset.buffer_all()
    i = 0
    for view_data in view_dataloader:
        print(str(i) +'/' +str(view_data.__len__()))
        i += 1

def main():
    args = parse_args()

    build_model(args)

if __name__ == '__main__':
    main()