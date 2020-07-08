import argparse
import os, time

import torch
from torch import nn
import numpy as np
import cv2
import scipy.io
from collections import OrderedDict

import _init_paths

from dataset import dataio
from dataset import data_util
from utils import util

from models import network
from utils import camera
from utils import sph_harm

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
    
    print('Set device...')
    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUS
    # device = torch.device('cuda:'+ cfg.GPUS)
    device = torch.device('cuda')

    print('Build Network...')
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
    interpolater = network.Interpolater()
    # rasterizer
    rasterizer = [] # load while testing

    print('Loading Model...')
    # load checkpoint
    util.custom_load([texture_mapper, render_net], ['texture_mapper', 'render_net'], cfg.MODEL.PRETRAINED)

    # move to device
    texture_mapper.to(device)
    render_net.to(device)
    interpolater.to(device)

    # use multi-GPU
    # if cfg.GPUS != '':
    #     texture_mapper = nn.DataParallel(texture_mapper)
    #     render_net = nn.DataParallel(render_net)

    # set mode
    texture_mapper.eval()
    render_net.eval()
    interpolater.eval()

    # fix test 
    # def set_bn_train(m):
    #     if type(m) == torch.nn.BatchNorm2d:
    #         m.train()
    # render_net.apply(set_bn_train)

    print('Loading Dataset...')
    view_dataset = dataio.ViewDataset(root_dir = cfg.DATASET.ROOT,
                                    calib_path = cfg.TEST.CALIB_PATH,
                                    calib_format = 'convert',
                                    img_size = cfg.DATASET.OUTPUT_SIZE,
                                    sampling_pattern = cfg.TEST.SAMPLING_PATTERN,
                                    is_train = False,
                                    load_precompute = False,
                                    )
    num_view = len(view_dataset)    
    view_dataset.buffer_all()

    print('Set Log...')
    log_dir = cfg.TEST.MODEL_DIR.split('/')
    log_dir = os.path.join(cfg.TEST.CALIB_DIR, 'resol_' + str(cfg.DATASET.OUTPUT_SIZE[0]), log_dir[-2],
                           log_dir[-1].split('_')[0] + '_' + log_dir[-1].split('_')[1] + '_' +
                           cfg.TEST.MODEL_NAME.split('-')[-1].split('.')[0])
    data_util.cond_mkdir(log_dir)
    save_dir_img_est = os.path.join(log_dir, cfg.TEST.SAVE_FOLDER)
    data_util.cond_mkdir(save_dir_img_est)
    save_dir_sh_basis_map = os.path.join(cfg.TEST.CALIB_DIR, 'resol_' + str(cfg.DATASET.OUTPUT_SIZE[0]), 'precomp', 'sh_basis_map')
    data_util.cond_mkdir(save_dir_sh_basis_map)

    print('Begin inference...')
    cur_obj_fp = ''
    with torch.no_grad():
        for ithView in range(num_view):
            start = time.time()

            # get view data
            view_trgt = view_dataset[ithView]
            proj = view_trgt[0]['proj'].to(device)
            pose = view_trgt[0]['pose'].to(device)
            proj_inv = view_trgt[0]['proj_inv'].to(device)
            R_inv = view_trgt[0]['R_inv'].to(device)
            coeffs = view_trgt[0]['dist_coeffs'].reshape((1,-1)).to(device)
            frame_idx = view_trgt[0]['f_idx']
            global_RT = view_dataset.global_RT

            proj = proj[None, :]
            pose = pose[None, :]
            proj_inv = proj_inv[None, :]
            R_inv = R_inv[None, :]

            # rasterize
            if cfg.DATASET.MESH_DIR%(frame_idx) != cur_obj_fp:
                cur_obj_fp = cfg.DATASET.MESH_DIR%(frame_idx)
                print('Reset Rasterizer to ' + cur_obj_fp)
                rasterizer = network.Rasterizer(obj_fp = cur_obj_fp, img_size = cfg.DATASET.OUTPUT_SIZE[0], global_RT = global_RT)
                rasterizer.to(device)
                #rasterizer.cuda(0) # currently, rasterizer can only be put on gpu 0
                rasterizer.eval()

            uv_map, alpha_map, face_index_map, weight_map, faces_v_idx, normal_map, normal_map_cam, faces_v, faces_vt, position_map, position_map_cam, depth, v_uvz, v_front_mask = \
                rasterizer(proj = proj,
                            pose = pose,
                            dist_coeffs = coeffs,
                            offset = None,
                            scale = None,
                            )
            
            batch_size = alpha_map.shape[0]

            sh_basis_map_fp = os.path.join(save_dir_sh_basis_map, str(ithView).zfill(5) + '.mat')
            if cfg.TEST.FORCE_RECOMPUTE or not os.path.isfile(sh_basis_map_fp):
                print('Compute sh_basis_map...')
                # compute view_dir_map in world space
                view_dir_map, _ = camera.get_view_dir_map(uv_map.shape[1:3], proj_inv, R_inv)
                # SH basis value for view_dir_map
                sh_basis_map = sph_harm.evaluate_sh_basis(lmax = 2, directions = view_dir_map.reshape((-1, 3)).cpu().detach().numpy()).reshape((*(view_dir_map.shape[:3]), -1)).astype(np.float32) # [N, H, W, 9]                
                # save
                scipy.io.savemat(sh_basis_map_fp, {'sh_basis_map': sh_basis_map[0, :]})
            else:
                sh_basis_map = scipy.io.loadmat(sh_basis_map_fp)['sh_basis_map'][None, ...]
            sh_basis_map = torch.from_numpy(sh_basis_map).to(device)

            # sample texture
            neural_img = texture_mapper(uv_map, sh_basis_map)

            # rendering net
            outputs_final = render_net(neural_img, None)
            img_max_val = 2.0
            outputs_final = (outputs_final * 0.5 + 0.5) * img_max_val # map to [0, img_max_val]

            # apply alpha
            outputs_final[0] = outputs_final[0] * alpha_map

            # debug
            # # save uvmap
            # uv_map = uv_map.cpu().detach().numpy()
            # save_dir_uv_map = '/data/NFS/new_disk/chenxin/relightable-nr/data/synthesis_gai/test_calib53_fixed/resol_512/dnr/07-01_10-01-45_32000/uv_est'
            # scipy.io.savemat(os.path.join(save_dir_uv_map, str(ithView).zfill(5) + '.mat'), {'uv_map': uv_map[0, :]})
            # # save uv_map preview
            # uv_map_img = np.concatenate((uv_map[0, :, :, :], np.zeros((*uv_map.shape[1:3], 1))), axis = 2)
            # cv2.imwrite(os.path.join(save_dir_uv_map, 'preview', str(ithView).zfill(5) + '.png'), uv_map_img[:, :, ::-1] * 255)

            # save
            cv2.imwrite(os.path.join(save_dir_img_est, str(ithView).zfill(5) + '.png'), outputs_final[0, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)

            end = time.time()
            print("View %07d   t_total %0.4f" % (ithView, end - start))

if __name__ == '__main__':
    main()
