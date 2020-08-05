import argparse
import os, time

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import cv2
import scipy.io
from collections import OrderedDict

import _init_paths

from dataset import dataio, zhen_dataset_rotate
from dataset import data_util
from utils import util

from models import network

from config import cfg
from config import update_config

from utils import camera
from utils import sph_harm
from utils import util
from shutil import copyfile


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


def main():
    print('Load config...')
    args = parse_args()
    update_config(cfg, args)

    print('Loading Dataset...')
    view_dataset = zhen_dataset_rotate.ViewDatasetZhen(cfg, is_train=False)
    # num_view = len(view_dataset)
    # view_dataset.buffer_all()
    view_dataloader = DataLoader(view_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=8)

    print('Set device...')
    # model = DataParallelModel(model, device_ids=cfg.GPUS).cuda()
    # CUDA_VISIBLE_DEVICES = 2
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPUS)
    device = torch.device('cuda')
    # device = torch.device('cuda:'+ str(cfg.GPUS))

    print('Build Network...')
    # texture mapper
    texture_mapper = network.TextureMapper(texture_size=cfg.MODEL.TEX_MAPPER.NUM_SIZE,
                                           texture_num_ch=cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                           mipmap_level=cfg.MODEL.TEX_MAPPER.MIPMAP_LEVEL,
                                           texture_init=None,
                                           fix_texture=True,
                                           apply_sh=cfg.MODEL.TEX_MAPPER.SH_BASIS)
    # rendering module
    render_module = network.RenderingModule(nf0=cfg.MODEL.RENDER_MODULE.NF0,
                                      in_channels=cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                      out_channels=cfg.MODEL.RENDER_MODULE.OUTPUT_CHANNELS,
                                      num_down_unet=5,
                                      use_gcn=False)
    # interpolater
    # interpolater = network.Interpolater()
    # Rasterizer
    # cur_obj_path = ''

    print('Loading Model...')
    # load checkpoint
    util.custom_load([texture_mapper, render_module], ['texture_mapper', 'render_module'], cfg.TEST.MODEL_PATH)

    # move to device
    texture_mapper.to(device)
    render_module.to(device)
    # interpolater.to(device)
    # rasterizer.to(device)

    # use multi-GPU
    # if cfg.GPUS != '':
    #     texture_mapper = nn.DataParallel(texture_mapper)
    #     render_module = nn.DataParallel(render_module)

    # set mode
    texture_mapper.eval()
    render_module.eval()
    # interpolater.eval()
    # rasterizer.eval()

    # texture_mapper.train()
    # render_module.train()
    # interpolater.train()
    # rasterizer.train()

    # # fix test
    # def set_bn_train(m):
    #     if type(m) == torch.nn.BatchNorm2d:
    #         m.train()
    # render_module.apply(set_bn_train)

    print('Set Log...')
    # log_dir = cfg.TEST.CALIB_DIR.split('/')
    # log_dir = os.path.join(cfg.TEST.MODEL_DIR, cfg.TEST.CALIB_NAME[:-4], 'resol_' + str(cfg.DATASET.OUTPUT_SIZE[0]),
    #                        log_dir[-2],
    #                        log_dir[-1].split('_')[0] + '_' + log_dir[-1].split('_')[1] + '_' +
    #                        cfg.TEST.MODEL_NAME.split('-')[-1].split('.')[0])
    log_dir = cfg.TEST.MODEL_DIR
    print(log_dir)
    log_dir = os.path.join(log_dir, os.path.splitext(os.path.basename(cfg.TEST.MODEL_PATH))[0])
    print(log_dir)
    data_util.cond_mkdir(log_dir)
    save_dir_img_est = os.path.join(log_dir, cfg.TEST.SAVE_FOLDER)
    print(save_dir_img_est)
    data_util.cond_mkdir(save_dir_img_est)
    # save_dir_sh_basis_map = os.path.join(cfg.TEST.CALIB_DIR, 'resol_' + str(cfg.DATASET.OUTPUT_SIZE[0]), 'precomp',
    #                                      'sh_basis_map')
    # data_util.cond_mkdir(save_dir_sh_basis_map)
    util.custom_copy(args.cfg, os.path.join(log_dir, cfg.LOG.CFG_NAME))

    print('Begin inference...')
    inter = 0
    with torch.no_grad():
        # for ithView in range(num_view):
        # view_trgt = view_dataset[ithView]
        # print("emm")
        # print(view_dataloader)
        for view_trgt in view_dataloader:
            start = time.time()

            # rasterize
            uv_map = view_trgt['uvmap'].permute(0, 2, 3, 1).to(device)
            # alpha_map = view_trgt['mask'].to(device)

            # print("after data assign")

            # sample texture
            neural_img = texture_mapper(uv_map)

            # rendering module
            outputs = render_module(neural_img, None)
            # img_max_val = 2.0
            # outputs = (outputs * 0.5 + 0.5) * img_max_val  # map to [0, img_max_val]

            # print("finish!")

            # apply alpha
            # outputs = outputs * alpha_map

            # save
            for batch_idx in range(0, outputs.shape[0]):
                cv2.imwrite(os.path.join(save_dir_img_est, str(inter).zfill(5) + '.png'),
                            outputs[batch_idx, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)
                inter = inter + 1

            end = time.time()
            print("View %07d   t_total %0.4f" % (inter, end - start))

    util.make_gif(save_dir_img_est, save_dir_img_est + '.gif')


if __name__ == '__main__':
    main()
