import os
import numpy as np

from PIL import Image
import scipy.io
from dataset.data_util_densepose.data_util import get_params, get_transform, is_image_file
from dataset.data_util_densepose.transform import RandomTransform
from dataset.data_util_densepose.uv_converter import TransferDenseposeUV, UVConverter

import torch
import torchvision.transforms as T

'''
Dataloader for dataset with uv map from densepose
'''
class DPViewDataset():
    def __init__(self,
                 cfg,
                 is_train=True
                 ):
        super().__init__()

        self.cfg = cfg
        self.root_dir = self.cfg.DATASET.ROOT
        self.is_train = is_train

        if self.is_train:
            self.img_dir = os.path.join(self.root_dir, 'img')
            self.uvmap_dir = os.path.join(self.root_dir, 'uv')
            self.mask_dir = os.path.join(self.root_dir, 'mask')

            if not os.path.isdir(self.root_dir):
                raise ValueError("Error! root dir is wrong")
            if not os.path.isdir(self.img_dir):
                raise ValueError("Error! image dir is wrong")
            if not os.path.isdir(self.uvmap_dir):
                raise ValueError("Error! uvmap dir is wrong")
            if not os.path.isdir(self.mask_dir):
                raise ValueError("Error! mask dir is wrong")
    
            self.frame_range = cfg.DATASET.FRAME_RANGE

            # get frames
            self.img_names = []
            # load with frame_range or all
            if len(self.frame_range):
                for frame_idx in self.frame_range:
                    self.img_names.append(cfg.DATASET.IMG_DIR % (frame_idx))
            else:
                for x in os.listdir(self.img_dir):
                    if is_image_file(x):
                        self.img_names.append(x)

            print(" building transform for images...")
        else:
            self.frame_range = cfg.TEST.FRAME_RANGE

            self.uvmap_dir = os.path.join(self.root_dir, 'uv')
            if not os.path.isdir(self.uvmap_dir):
                raise ValueError("Error! uvmap dir is wrong")
            self.img_names = []
            # load with frame_range or all
            if len(self.frame_range):
                for frame_idx in self.frame_range:
                    self.img_names.append(cfg.DATASET.IMG_DIR % frame_idx)
            else:
                for x in os.listdir(self.uvmap_dir):
                    if(x[-4:] == '.mat'):
                    # if(x[-4:] == '.mat') and (x[9:11] == '01') :
                        self.img_names.append(x)

            self.img_names = sorted(self.img_names)
            print("image names", self.img_names)
        if cfg.VERBOSE:
            print("image names", self.img_names)
        self.transform = RandomTransform(size = cfg.DATASET.OUTPUT_SIZE,
                                         max_shift = cfg.DATASET.MAX_SHIFT,
                                         max_scale = cfg.DATASET.MAX_SCALE,
                                         max_rotation = cfg.DATASET.MAX_ROTATION,
                                         isTrain = is_train)
        self.uv_converter = UVConverter(cfg.DATASET.UV_CONVERTER)
    def __len__(self):
        return len(self.img_names)

    def get_all_view(self):
        imgs = torch.zeros(len(self.img_names), 3, 512, 512)
        uvmaps = torch.zeros(len(self.img_names), 512, 512, 2)
        for idx, img_name in enumerate(self.img_names):
            img_key = os.path.splitext(img_name)[0]

            img_path = os.path.join(self.img_dir, img_name)
            uvmap_path = os.path.join(self.uvmap_dir, img_key + '_IUV.mat')

            img = Image.open(img_path)
            img = T.functional.to_tensor(img)
            imgs[idx, ...] = img

            uvmap = scipy.io.loadmat(uvmap_path)['uv_map']
            # uvmap = TransferDenseposeUV(uvmap)
            uvmap = self.uv_converter(uvmap)
            uvmap = uvmap - np.floor(uvmap)
            uvmap = torch.tensor(uvmap.astype(np.float32))
            uvmaps[idx, ...] = uvmap
        return imgs, uvmaps        
    
    def __getitem__(self, idx):
        if self.is_train:
            img_path = os.path.join(self.img_dir, self.img_names[idx])
            img_key = os.path.splitext(self.img_names[idx])[0]
            uvmap_path = os.path.join(self.uvmap_dir, img_key + '_IUV.mat')
            mask_path = os.path.join(self.mask_dir, img_key + '.png')

            # # randomly select a pose and projection
            # select_view = np.random.randint(0, self.num_view - 1)
            # # extrinsic
            # pose = self.calib['poses'][select_view]
            # pose = np.dot(pose, self.global_RT_inv)
            # select_view = np.random.randint(0, self.num_view - 1)
            # # intrinsic
            # proj = self.calib['projs'][select_view, ...]

            img = Image.open(img_path)

            uvmap = scipy.io.loadmat(uvmap_path)['uv_map']
            # uvmap = TransferDenseposeUV(uvmap)
            uvmap = self.uv_converter(uvmap)
            # w, h = uvmap.shape[:2]
            # uvmap_3channel = np.concatenate([uvmap, np.zeros([w, h, 1])], axis=2) * 255
            # uvmap = Image.fromarray(uvmap_3channel.astype('uint8'))

            mask = Image.open(mask_path)

            # params = get_params(self.cfg, img.size)
            # transform_image = get_transform(self.cfg, params, normalize=False, toTensor=False, isTrain=self.is_train)
            # img = transform_image(img)

            # uvmap_transform = get_transform(self.cfg, params, normalize=False, toTensor=False, isTrain=self.is_train)
            # uvmap = uvmap_transform(uvmap)

            # mask_transform = get_transform(self.cfg, params, normalize=False, toTensor=False, isTrain=self.is_train)
            # mask = mask_transform(mask)

            img, mask, uvmap, _, _ = self.transform(img, mask=mask, uvmap=uvmap)
            uvmap = uvmap[:2, ...]
            uvmap[uvmap >= 1] = 1 - 1e-5
            uvmap[uvmap < 0] = 0
            uvmap = uvmap.permute(1,2,0)

            mask[mask > 0] = 1
            return {'img': img, 'uv_map': uvmap, 'mask': mask}
        else:
            img_key = os.path.splitext(self.img_names[idx])[0]
            
            # mask_path = os.path.join(self.mask_dir, img_key[:-4] + '_INDS.png')
            # print(mask_path)
            # mask = Image.open(mask_path)
            # mask = np.array(mask)
            # mask = mask[np.newaxis, ...]
            # mask[mask > 0] = 1
            # mask = torch.Tensor(mask)
            # print(mask.size())
            
            uvmap_path = os.path.join(self.uvmap_dir, img_key + '_IUV.mat')
            if not os.path.isfile(uvmap_path):
                uvmap_path = os.path.join(self.uvmap_dir, img_key + '.mat') 
            if not os.path.isfile(uvmap_path):
                raise ValueError('Not exist uvmap ...' + uvmap_path)

            uvmap = scipy.io.loadmat(uvmap_path)['uv_map']
            # uvmap = self.uv_converter(uvmap)
            uvmap = TransferDenseposeUV(uvmap)

            _, _, uvmap, _, _ = self.transform(img=None, mask=None, uvmap=uvmap)

            uvmap = uvmap - np.floor(uvmap)
            # uvmap = torch.tensor(uvmap.astype(np.float32))


            # w, h = uvmap.shape[:2]
            # uvmap_3channel = np.concatenate([uvmap, np.zeros([w, h, 1])], axis=2) * 255
            # uvmap = Image.fromarray(uvmap_3channel.astype('uint8'))

            # params = get_params(self.cfg, uvmap.size)
            # uvmap_transform = get_transform(self.cfg, params, normalize=False, toTensor=True, isTrain=self.is_train)
            # uvmap = uvmap_transform(uvmap)
            # uvmap = uvmap[:2, ...]
            return {'uv_map': uvmap}

    def buffer_all(self):
        pass