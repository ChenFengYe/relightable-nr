import argparse
import os, time

import torch
from torch import nn
import numpy as np
import cv2
import scipy.io
from collections import OrderedDict

from dataset import dataio
from dataset import data_util
from utils import util

from models import network
from utils import camera
from utils import sph_harm


parser = argparse.ArgumentParser()

# inference sequence
parser.add_argument('--img_size', type=int, default=512,
                    help='Sidelength of generated images. Default 512. Only less than native resolution of images is recommended.')
parser.add_argument('--calib_dir', type=str, required=True,
                    help='Path of calibration file for inference sequence.')
parser.add_argument('--sampling_pattern', type=str, default='all', required=False)
# checkpoint
parser.add_argument('--checkpoint_dir', default='',
                    help='Path to a checkpoint to load render_net weights from.')
parser.add_argument('--checkpoint_name', default='',
                    help='Path to a checkpoint to load render_net weights from.')
# misc
parser.add_argument('--gpu_id', type=str, default='',
                    help='Cuda visible devices.')
parser.add_argument('--force_recompute', default = False, type = lambda x: (str(x).lower() in ['true', '1']))
parser.add_argument('--multi_frame', type=bool, default=False, help='Input dynamic frame')

opt = parser.parse_args()
checkpoint_fp = os.path.join(opt.checkpoint_dir, opt.checkpoint_name)

# load params
params_fp = os.path.join(opt.checkpoint_dir, 'params.txt')
params_file = open(params_fp, "r")
params_lines = params_file.readlines()
params = {}
for line in params_lines:
    key = line.split(':')[0]
    val = line.split(':')[1]
    if len(val) > 1:
        val = val[1:-1]
    else:
        val = None
    params[key] = val
# general
opt.data_root = params['data_root']
# mesh
opt.obj_fp = params['obj_fp']
# texture mapper
opt.texture_size = int(params['texture_size'])
opt.texture_num_ch = int(params['texture_num_ch'])
opt.mipmap_level = int(params['mipmap_level'])
opt.apply_sh = params['apply_sh'] in ['True', 'true', '1']
# rendering net
opt.nf0 = int(params['nf0'])

if opt.calib_dir[:2] == '_/':
    opt.calib_dir = os.path.join(opt.data_root, opt.calib_dir[2:])
if opt.obj_fp[:2] == '_/':
    opt.obj_fp = os.path.join(opt.data_root, opt.obj_fp[2:])
obj_name = opt.obj_fp.split('/')[-1].split('.')[0]

print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

if opt.gpu_id == '':
    device = torch.device('cpu')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device('cuda')


# load global_RT
opt.calib_fp = os.path.join(opt.calib_dir, 'calib.mat')
global_RT = torch.from_numpy(scipy.io.loadmat(opt.calib_fp)['global_RT'].astype(np.float32))

num_channel = 3

# dataset for training views
view_dataset = dataio.ViewDataset(root_dir = opt.data_root,
                                calib_path = opt.calib_fp,
                                calib_format = 'convert',
                                img_size = [opt.img_size, opt.img_size],
                                sampling_pattern = opt.sampling_pattern,
                                is_train = False,
                                load_precompute = False,
                                multi_frame = opt.multi_frame
                                )
num_view = len(view_dataset)

# texture mapper
texture_mapper = network.TextureMapper(texture_size = opt.texture_size,
                                        texture_num_ch = opt.texture_num_ch,
                                        mipmap_level = opt.mipmap_level,
                                        texture_init = None,
                                        fix_texture = True,
                                        apply_sh = opt.apply_sh)

# load checkpoint
checkpoint_dict = util.custom_load([texture_mapper], ['texture_mapper'], checkpoint_fp, strict = False)

# rendering net
new_state_dict = OrderedDict()
for k, v in checkpoint_dict['render_net'].items():
    name = k.replace("module.", "")
    new_state_dict[name] = v
render_net = network.RenderingNet(nf0 = opt.nf0,
                                in_channels = opt.texture_num_ch,
                                out_channels = num_channel,
                                num_down_unet = 5,
                                use_gcn = False)
render_net.load_state_dict(new_state_dict, strict = False)

# interpolater
interpolater = network.Interpolater()

# rasterizer
rasterizer = [] # load while testing

# move to device
texture_mapper.to(device)
render_net.to(device)
interpolater.to(device)

# use multi-GPU
# if opt.gpu_id != '':
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

def main():
    view_dataset.buffer_all()

    log_dir = opt.checkpoint_dir.split('/')
    log_dir = os.path.join(opt.calib_dir, 'resol_' + str(opt.img_size), log_dir[-2],
                           log_dir[-1].split('_')[0] + '_' + log_dir[-1].split('_')[1] + '_' +
                           opt.checkpoint_name.split('-')[-1].split('.')[0])
    data_util.cond_mkdir(log_dir)

    save_dir_img_est = os.path.join(log_dir, 'img_est')
    data_util.cond_mkdir(save_dir_img_est)

    save_dir_sh_basis_map = os.path.join(opt.calib_dir, 'resol_' + str(opt.img_size), 'precomp', 'sh_basis_map')
    data_util.cond_mkdir(save_dir_sh_basis_map)

    # Save all command line arguments into a txt file in the logging directory for later referene.
    with open(os.path.join(log_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

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

            proj = proj[None, :]
            pose = pose[None, :]
            proj_inv = proj_inv[None, :]
            R_inv = R_inv[None, :]

            # rasterize
            if opt.obj_fp%(frame_idx) != cur_obj_fp:
                cur_obj_fp = opt.obj_fp%(frame_idx)
                print('Reset Rasterizer to ' + cur_obj_fp)
                rasterizer = network.Rasterizer(obj_fp = cur_obj_fp, img_size = opt.img_size, global_RT = global_RT)
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
            if opt.force_recompute or not os.path.isfile(sh_basis_map_fp):
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
            #outputs_final[0] = outputs_final[0] * alpha_map
                
            # save
            cv2.imwrite(os.path.join(save_dir_img_est, str(ithView).zfill(5) + '.png'), outputs_final[0, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)

            end = time.time()
            print("View %07d   t_total %0.4f" % (ithView, end - start))


if __name__ == '__main__':
    main()
