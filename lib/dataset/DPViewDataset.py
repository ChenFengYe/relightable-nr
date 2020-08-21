import os
import numpy as np
import random

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
        self.is_pairwise = cfg.TRAIN.SAMPLING_PAIRWISE

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
        else:
            self.frame_range = cfg.TEST.FRAME_RANGE

            self.img_dir = os.path.join(self.root_dir, 'img')
            self.uvmap_dir = os.path.join(self.root_dir, 'uv')
            self.mask_dir = os.path.join(self.root_dir, 'mask')
            
            if not os.path.isdir(self.uvmap_dir):
                raise ValueError("Error! uvmap dir is wrong")
            self.img_names = []
            # load with frame_range or all
            if len(self.frame_range):
                for frame_idx in self.frame_range:
                    self.img_names.append(cfg.DATASET.IMG_DIR % frame_idx)
            else:
                # train_set = [20,35,50,65,80,95,110,125,140]
                for x in os.listdir(self.uvmap_dir):
                    # if(x[-4:] == '.mat') and (int(x[:3]) not in train_set):
                    # if(x[-4:] == '.mat') and (x[9:11] == '01') :
                    if x[-4:] == '.mat':
                        self.img_names.append(x)

            self.img_names = sorted(self.img_names)

        print(" building transform for images...")
        self.transform = RandomTransform(size = cfg.DATASET.OUTPUT_SIZE,
                                         max_shift = cfg.DATASET.MAX_SHIFT,
                                         max_scale = cfg.DATASET.MAX_SCALE,
                                         max_rotation = cfg.DATASET.MAX_ROTATION,
                                         isTrain = is_train)

        self.uv_converter = UVConverter(cfg.DATASET.UV_CONVERTER)

        # create referred image set
        if self.is_pairwise:
            self.img_names_ref = []

            for i in range(len(self.img_names)):
                img_name = self.img_names[i]
                img_refs = []
                range_ref = [max(0,i-10), min(len(self.img_names)-1,i+10)]
                for ir in range(range_ref[0],range_ref[1]):
                    if i != ir and img_name[:5] == self.img_names[ir][:5]:
                        img_refs.append(self.img_names[ir])
                self.img_names_ref.append(img_refs)       

            # check ref list
            for i, img_name_ref in enumerate(self.img_names_ref):
                if len(img_name_ref) == 0:
                    self.img_names_ref[i].append(self.img_names[i])

        # reshape dataset length for multi-gpu
        # batch_size = cfg.TRAIN.BATCH_SIZE
        # batch_size = cfg.TRAIN.BATCH_SIZE * 4
        batch_size = 4
        cur_len = len(self.img_names)
        tar_len = cur_len + batch_size - np.mod(cur_len, batch_size)
        self.img_names = np.resize(self.img_names, tar_len)
        self.img_names_ref = np.resize(self.img_names_ref, tar_len)
            
        if cfg.VERBOSE:
            print("image names", self.img_names[:100], 
                  "ignore part of list, because it is too long." if len(self.img_names)>100 else "")
            print("image num ", len(self.img_names))
           

    def __len__(self):
        return len(self.img_names)

    def get_all_view(self):
        size = cfg.DATASET.OUTPUT_SIZE
        imgs = torch.zeros(len(self.img_names), 3, size, size)
        uvmaps = torch.zeros(len(self.img_names), size, size, 2)
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

    def load_view(self, img_name):
        img_key = os.path.splitext(img_name)[0]

        img_path = os.path.join(self.img_dir, img_name)
        uvmap_path = os.path.join(self.uvmap_dir, img_key + '_IUV.mat')
        mask_path = os.path.join(self.mask_dir, img_key + '.png')

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        uvmap = scipy.io.loadmat(uvmap_path)['uv_map']
        # uvmap = TransferDenseposeUV(uvmap)
        uvmap = self.uv_converter(uvmap)
        # params = get_params(self.cfg, img.size)
        # transform_image = get_transform(self.cfg, params, normalize=False, toTensor=False, isTrain=self.is_train)
        # img = transform_image(img)
        # uvmap_transform = get_transform(self.cfg, params, normalize=False, toTensor=False, isTrain=self.is_train)
        # uvmap = uvmap_transform(uvmap)
        # mask_transform = get_transform(self.cfg, params, normalize=False, toTensor=False, isTrain=self.is_train)
        # mask = mask_transform(mask)
        img, mask, uvmap, _, _ = self.transform(img, mask=mask, uvmap=uvmap)
        uvmap = uvmap[:2, ...]
        # uvmap[uvmap >= 1] = 1 - 1e-5
        uvmap[uvmap >= 1] = 0
        uvmap[uvmap < 0] = 0
        uvmap = uvmap.permute(1,2,0)

        mask[mask > 0] = 1
        mask = mask[0, ...]
        return {'img': img, 'uv_map': uvmap, 'mask': mask}

    def __getitem__(self, idx):
        if self.is_train:                    
            # get referred img key
            img_name = self.img_names[idx]
            view_data = self.load_view(img_name)
            if not self.is_pairwise:
                return {'img': view_data['img'],'uv_map': view_data['uv_map'],'mask': view_data['mask']}
            else:
                img_idx_ref = random.randint(0, len(self.img_names_ref[idx])-1)
                img_name_ref = self.img_names_ref[idx][img_idx_ref]
                view_ref_data = self.load_view(img_name_ref)
                return {'img': view_data['img'],'uv_map': view_data['uv_map'],'mask': view_data['mask'],
                        'img_ref': view_ref_data['img'],'uv_map_ref': view_ref_data['uv_map'],'mask_ref': view_ref_data['mask']}
        else:
            img_key = os.path.splitext(self.img_names[idx])[0]
            img_path = os.path.join(self.img_dir, self.img_names[idx])
            img_key = os.path.splitext(self.img_names[idx])[0]
            uvmap_path = os.path.join(self.uvmap_dir, img_key + '_IUV.mat')
            mask_path = os.path.join(self.mask_dir, img_key + '.png')

            img = Image.open(img_path) if os.path.exists(img_path) else None
            mask = Image.open(mask_path) if os.path.exists(mask_path) else None
            
            uvmap_path = os.path.join(self.uvmap_dir, img_key + '_IUV.mat')
            if not os.path.isfile(uvmap_path):
                uvmap_path = os.path.join(self.uvmap_dir, img_key + '.mat') 
            if not os.path.isfile(uvmap_path):
                raise ValueError('Not exist uvmap ...' + uvmap_path)

            uvmap = scipy.io.loadmat(uvmap_path)['uv_map']
            uvmap = self.uv_converter(uvmap)
            # uvmap = TransferDenseposeUV(uvmap)

            img, mask, uvmap, _, _ = self.transform(img=img, mask=mask, uvmap=uvmap)

            uvmap = uvmap - np.floor(uvmap)
            uvmap = uvmap.permute(1,2,0)
            if mask is not None:
                mask[mask > 0] = 1
                mask = mask[0, ...]
            # return {'img': img, 'uv_map': uvmap, 'mask': mask}
            return {'uv_map': uvmap}

    def buffer_all(self):
        pass