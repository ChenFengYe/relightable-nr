import os
import torch
import numpy as np
# import numpy.linalg

# from PIL import Image
# import cv2
# import scipy.io

# from dataset.data_util_dome.data_util import samping_img_set, glob_imgs, load_img
from dataset.data_util_dome.transform import RandomTransform
# from dataset.data_util_densepose.uv_converter import TransferDenseposeUV, UVConverter
from dataset.DomeViewDataset import DomeViewDataset

from torch.nn import functional as F
# import neural_renderer as nr
# from lib.models import network

def get_closest_view(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, -1)
    y_vec = y.view(N, -1)
    dist = torch.sum((x_vec - y_vec) ** 2, dim=1, keepdim=True)
    min_val, min_id = torch.min(dist, dim=0)
    return min_id

# def merge_with_cfg(cfg, cfg_tar):
#     cfg.DATASET.DATASET = cfg_tar.DATASET_FVV.DATASET
#     cfg.DATASET.ROOT = cfg_tar.DATASET_FVV.ROOT
#     cfg.DATASET.FRAME_RANGE = cfg_tar.DATASET_FVV.FRAME_RANGE
#     cfg.DATASET.CAM_RANGE = cfg_tar.DATASET_FVV.CAM_RANGE
#     cfg.DATASET.MESH_DIR = cfg_tar.DATASET_FVV.MESH_DIR
#     cfg.DATASET.OUTPUT_SIZE = cfg_tar.DATASET_FVV.OUTPUT_SIZE
#     cfg.DATASET.CALIB_PATH = cfg_tar.DATASET_FVV.CALIB_PATH
#     cfg.DATASET.CALIB_FORMAT = cfg_tar.DATASET_FVV.CALIB_FORMAT
#     cfg.DATASET.CAM_MODE = cfg_tar.DATASET_FVV.CAM_MODE
#     cfg.DATASET.PRELOAD_MESHS = cfg_tar.DATASET_FVV.PRELOAD_MESHS
#     cfg.DATASET.PRELOAD_VIEWS = cfg_tar.DATASET_FVV.PRELOAD_VIEWS
#     cfg.DATASET.TEX_PATH = cfg_tar.DATASET_FVV.TEX_PATH
#     cfg.DATASET.UV_PATH = cfg_tar.DATASET_FVV.UV_PATH
#     cfg.DATASET.UV_CONVERTER = cfg_tar.DATASET_FVV.UV_CONVERTER
#     cfg.DATASET.MAX_SHIFT = cfg_tar.DATASET_FVV.MAX_SHIFT
#     cfg.DATASET.MAX_ROTATION = cfg_tar.DATASET_FVV.MAX_ROTATION
#     cfg.DATASET.MAX_SCALE = cfg_tar.DATASET_FVV.MAX_SCALE
#     cfg.DATASET.FLIP = cfg_tar.DATASET_FVV.FLIP
#     return cfg

class DomeViewDatasetFVV(DomeViewDataset):
    def __init__(self,
                 cfg,
                 isTrain = True):
        # cfg = merge_with_cfg(cfg, cfg_ref)
        
        isTrain = False
        super().__init__(cfg, isTrain)
        self.root_dir = self.cfg.DATASET.ROOT
        self.img_dir = os.path.join(self.root_dir, 'img')
        self.uvmap_dir = os.path.join(self.root_dir, 'uv')
        self.mask_dir = os.path.join(self.root_dir, 'mask')

        # load all img and uvmap
        self.frame_range = cfg.DATASET.FRAME_RANGE
        self.img_names = []
        # load with frame_range or all
        if len(self.frame_range):
            name_set = [cfg.DATASET.IMG_DIR % (frame_idx) for frame_idx in self.frame_range]
        else:
            name_set = os.listdir(self.img_dir)
        
        for img_name in name_set:
            # check all data
            img_key = os.path.splitext(img_name)[0]
            image_path = os.path.join(self.img_dir, img_name)
            uvmap_path = os.path.join(self.uvmap_dir, img_key + '_IUV.mat')
            mask_path = os.path.join(self.mask_dir, img_key + '.png')
            if os.path.exists(image_path) and os.path.exists(mask_path) and os.path.exists(uvmap_path):
                self.img_names.append(img_name)
        self.img_names = sorted(self.img_names)
        
        self.transform = RandomTransform(size = cfg.DATASET.OUTPUT_SIZE,
                                         max_shift = cfg.DATASET.MAX_SHIFT,
                                         max_scale = cfg.DATASET.MAX_SCALE,
                                         max_rotation = cfg.DATASET.MAX_ROTATION,
                                         isTrain = isTrain)
        self.view_datas = []
        self.uv_maps = []
        self.imgs = []
        self.views_num = len(self.img_names)
        for img_name in self.img_names:
            view_data = self.load_view(img_name)
            self.view_datas.append(view_data)
            self.imgs.append(view_data['img'][None,...])
            self.uv_maps.append(view_data['uv_map'][None,...])

        batch_size = 4
        cur_len = len(self.img_names)
        tar_len = cur_len + batch_size - np.mod(cur_len, batch_size)

        self.view_range_all = self.view_range[1:]
        # self.view_range = np.array(list(range(0, tar_len)), dtype=np.int) # self.views_num
        self.view_range = self.view_range_all[:tar_len]
        self.dataset_len = tar_len

        # buff mesh
        self.buffer_all()

    def __len__(self):
        return len(self.calib)

    def refresh(self):
        self.view_range = (self.view_range + self.dataset_len-1) % len(self.view_range_all) + 1
        print(self.view_range)

    def __getitem__(self, idx):
        # Generate a uv map
        view_trgt = self.get_item(idx)  
        view_data = {}
        
        # Select a tow closed views, use L1 now
        uv_map = view_trgt['uv_map'][None,...].clone().detach().cpu().repeat(self.views_num,1,1,1)
        i_ref = get_closest_view(uv_map, torch.cat(self.uv_maps, 0))
        
        # Load image / uv map / mask / tex
        view_data['img_ref'] = self.view_datas[i_ref]['img']
        view_data['uv_map_ref'] = self.view_datas[i_ref]['uv_map']
        view_data['mask_ref'] = self.view_datas[i_ref]['mask']
        view_data['tex_ref'] = self.view_datas[i_ref]['tex']

        # 3D Data
        view_data['f_idx'] = view_trgt['f_idx']
        view_data['proj'] = view_trgt['proj']
        view_data['dist_coeffs'] = view_trgt['dist_coeffs']

        # Prepare training data       
        view_data['uv_map'] = view_trgt['uv_map'].clone().detach().cpu()
        view_data['data_type'] = 'rendered'
        # 'img': view_data['img'],
        # 'mask': view_data['mask'],
        # 'tex_tar': tex_tar
        return view_data