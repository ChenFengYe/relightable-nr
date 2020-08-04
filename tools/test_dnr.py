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

from dataset import dataio

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
    view_dataset = dataio.ViewDataset(cfg,
                                    root_dir = cfg.DATASET.ROOT,
                                    calib_path = cfg.TEST.CALIB_PATH,
                                    calib_format = cfg.DATASET.CALIB_FORMAT,
                                    sampling_pattern = cfg.TEST.SAMPLING_PATTERN,
                                    is_train = False,
                                    )
    num_view = len(view_dataset)
    view_dataset.buffer_all()
    view_dataloader = DataLoader(view_dataset, batch_size = cfg.TEST.BATCH_SIZE, shuffle = False, num_workers = 8)

    print('Set device...')
    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # CUDA_VISIBLE_DEVICES = 2
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPUS)
    device = torch.device('cuda')
    # device = torch.device('cuda:'+ str(cfg.GPUS))

    print('Build Network...')
    # texture creater
    texture_creater = network.TextureCreater(texture_size = cfg.MODEL.TEX_CREATER.NUM_SIZE,
                                            texture_num_ch = cfg.MODEL.TEX_CREATER.NUM_CHANNELS)
    # render net
    # feature_net = network.FeatureNet(nf0 = cfg.MODEL.FEATURE_NET.NF0,
    #                             in_channels = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS + cfg.MODEL.TEX_CREATER.NUM_CHANNELS,
    #                             out_channels = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
    #                             num_down_unet = 3,
    #                             use_gcn = False)

    # texture mapper
    texture_mapper = network.TextureMapper(texture_size = cfg.MODEL.TEX_MAPPER.NUM_SIZE,
                                            texture_num_ch = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                            mipmap_level = cfg.MODEL.TEX_MAPPER.MIPMAP_LEVEL,
                                            texture_init = None,
                                            fix_texture = True,
                                            apply_sh = cfg.MODEL.TEX_MAPPER.SH_BASIS)
    # rendering net
    render_net = network.RenderingNet(nf0 = cfg.MODEL.RENDER_NET.NF0,
                                    in_channels = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                    out_channels = cfg.MODEL.RENDER_NET.OUTPUT_CHANNELS,
                                    num_down_unet = 5,
                                    use_gcn = False)
    # interpolater
    # interpolater = network.Interpolater()
    # Rasterizer
    cur_obj_path = ''
    if not cfg.DATASET.LOAD_PRECOMPUTE:
        view_data = view_dataset.read_view(0)
        cur_obj_path = view_data['obj_path']
        frame_idx = view_data['f_idx']
        obj_data = view_dataset.objs[frame_idx]
        rasterizer = network.Rasterizer(cfg,
                            obj_fp = cur_obj_path, 
                            img_size = cfg.DATASET.OUTPUT_SIZE[0],
                            obj_data = obj_data,
                            camera_mode = cfg.DATASET.CAM_MODE,
                            # preset_uv_path = cfg.DATASET.UV_PATH,
                            global_RT = view_dataset.global_RT)

    print('Loading Model...')
    # load checkpoint
    util.custom_load([texture_mapper, render_net], ['texture_mapper', 'render_net'], cfg.TEST.MODEL_PATH)

    # move to device
    texture_mapper.to(device)
    # feature_net.to(device)    
    render_net.to(device)
    #interpolater.to(device)
    rasterizer.to(device)

    # use multi-GPU
    # if cfg.GPUS != '':
    #     texture_mapper = nn.DataParallel(texture_mapper)
    #     render_net = nn.DataParallel(render_net)

    # set mode
    # texture_mapper.eval()
    # render_net.eval()
    # interpolater.eval()
    # rasterizer.eval()

    texture_mapper.train()
    render_net.train()
    # feature_net.train()
    #interpolater.train()
    rasterizer.train()

    # # fix test 
    # def set_bn_train(m):
    #     if type(m) == torch.nn.BatchNorm2d:
    #         m.train()
    # render_net.apply(set_bn_train)

    print('Set Log...')
    log_dir = cfg.TEST.CALIB_DIR.split('/')
    log_dir = os.path.join(cfg.TEST.MODEL_DIR, cfg.TEST.CALIB_NAME[:-4], 'resol_' + str(cfg.DATASET.OUTPUT_SIZE[0]), log_dir[-2],
                           log_dir[-1].split('_')[0] + '_' + log_dir[-1].split('_')[1] + '_' +
                           cfg.TEST.MODEL_NAME.split('-')[-1].split('.')[0])
    util.cond_mkdir(log_dir)
    save_dir_img = os.path.join(log_dir, 
                                    cfg.TEST.SAVE_FOLDER, 
                                    str(cfg.TEST.FRAME_RANGE[0]) + str(cfg.TEST.FRAME_RANGE[1]))
    util.cond_mkdir(save_dir_img)
    
    print('Begin inference...')
    inter = 0
    with torch.no_grad():
        # for ithView in range(num_view):
        # view_trgt = view_dataset[ithView]
        for view_trgt in view_dataloader:

            start = time.time()
            
            # get view data
            frame_idxs = view_trgt[0]['f_idx'].numpy()
            # rasterize
            uv_map = []            
            alpha_map = []
            for batch_idx, frame_idx in enumerate(frame_idxs):
                obj_path = view_trgt[0]['obj_path'][batch_idx]
                if cur_obj_path != obj_path:
                    cur_obj_path = obj_path
                    obj_data = view_dataset.objs[frame_idx]
                    rasterizer.update_vs(obj_data['v_attr'])

                proj = view_trgt[0]['proj'].to(device)[batch_idx, ...]
                pose = view_trgt[0]['pose'].to(device)[batch_idx, ...]
                dist_coeffs = view_trgt[0]['dist_coeffs'].to(device)[batch_idx, ...]

                uv_map_single, alpha_map_single, _, _, _, _, _, _, _, _, _, _, _, _ = \
                    rasterizer(proj = proj[None, ...], 
                                pose = pose[None, ...], 
                                dist_coeffs = dist_coeffs[None, ...], 
                                offset = None,
                                scale = None,
                                )
                uv_map.append(uv_map_single[0, ...].clone().detach())
                alpha_map.append(alpha_map_single[0, ...].clone().detach())

            # fix alpha map
            uv_map = torch.stack(uv_map, dim = 0)
            alpha_map = torch.stack(alpha_map, dim = 0)[:, None, : , :]

            # ignore sh_basis now
            # sh_basis_map_fp = os.path.join(save_dir_sh_basis_map, str(ithView).zfill(5) + '.mat')
            # if cfg.TEST.FORCE_RECOMPUTE or not os.path.isfile(sh_basis_map_fp):
            #     print('Compute sh_basis_map...')
            #     # compute view_dir_map in world space
            #     view_dir_map, _ = camera.get_view_dir_map(uv_map.shape[1:3], proj_inv, R_inv)
            #     # SH basis value for view_dir_map
            #     sh_basis_map = sph_harm.evaluate_sh_basis(lmax = 2, directions = view_dir_map.reshape((-1, 3)).cpu().detach().numpy()).reshape((*(view_dir_map.shape[:3]), -1)).astype(np.float32) # [N, H, W, 9]                
            #     # save
            #     scipy.io.savemat(sh_basis_map_fp, {'sh_basis_map': sh_basis_map[0, :]})
            # else:
            #     sh_basis_map = scipy.io.loadmat(sh_basis_map_fp)['sh_basis_map'][None, ...]
            # sh_basis_map = torch.from_numpy(sh_basis_map).to(device)

            # sample texture
            neural_img = texture_mapper(uv_map)

            if cfg.DEBUG.DEBUG:
                neural_tex = texture_mapper.texture_i.cpu().detach().numpy();
                scipy.io.savemat('./Debug/Nerual_tex.mat', neural_tex)

            # rendering net
            outputs = render_net(neural_img, None)
            img_max_val = 2.0
            outputs = (outputs * 0.5 + 0.5) * img_max_val # map to [0, img_max_val]

            # apply alpha
            outputs = outputs * alpha_map

            # save
            for batch_idx in range(0, outputs.shape[0]):
                cv2.imwrite(os.path.join(save_dir_img, str(inter).zfill(5) + '.png'), outputs[batch_idx, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)
                inter = inter + 1

            end = time.time()
            print("View %07d   t_total %0.4f" % (inter, end - start))
    
    util.make_gif(save_dir_img, save_dir_img+'.gif')    

if __name__ == '__main__':
    main()
