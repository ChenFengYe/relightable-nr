import os
import numpy as np

from PIL import Image
import scipy.io
from dataset.data_util_densepose.data_util import get_params, get_transform, TransferDenseposeUV, is_image_file
from dataset.data_util_densepose.transform import RandomTransform
import torch

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
        if not self.is_train:
            # get frames
            self.uvmap_dir = os.path.join(self.root_dir, 'uv')
            self.img_names = []
            for x in os.listdir(self.uvmap_dir):
                self.img_names.append(x)
            # self.mask_dir = os.path.join(self.root_dir, 'mask')
            self.img_names = sorted(self.img_names)
            # print("uvmap dir", self.uvmap_dir)
            # print("mask dir", self.mask_dir)
            print("image names", self.img_names)
        else:
            self.img_dir = os.path.join(self.root_dir, 'img')
            self.uvmap_dir = os.path.join(self.root_dir, 'uv')
            self.mask_dir = os.path.join(self.root_dir, 'mask')

            calib_path = cfg.DATASET.CALIB_PATH
            self.calib = scipy.io.loadmat(calib_path)
            self.num_view = self.calib['poses'].shape[0]

            self.global_RT = self.calib['global_RT']
            self.global_RT_inv = np.linalg.inv(self.global_RT)

            if not os.path.isdir(self.root_dir):
                raise ValueError("Error! root dir is wrong")
            if not os.path.isdir(self.img_dir):
                raise ValueError("Error! image dir is wrong")
            if not os.path.isdir(self.uvmap_dir):
                raise ValueError("Error! uvmap dir is wrong")
            if not os.path.isdir(self.mask_dir):
                raise ValueError("Error! mask dir is wrong")

            self.out_size = list(cfg.DATASET.OUTPUT_SIZE)

            # get frames
            self.img_names = []
            for x in os.listdir(self.img_dir):
                if is_image_file(x):
                    self.img_names.append(x)

            print(" building transform for images...")
            self.transform = RandomTransform(cfg.DATASET.OUTPUT_SIZE,
                                             cfg.DATASET.MAX_SHIFT,
                                             cfg.DATASET.MAX_SCALE,
                                             cfg.DATASET.MAX_ROTATION)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.is_train:
            img_path = os.path.join(self.img_dir, self.img_names[idx])
            img_key = os.path.splitext(self.img_names[idx])[0]
            uvmap_path = os.path.join(self.uvmap_dir, img_key + '_IUV.mat')
            mask_path = os.path.join(self.mask_dir, img_key + '.png')

            # randomly select a pose and projection
            select_view = np.random.randint(0, self.num_view - 1)
            # extrinsic
            pose = self.calib['poses'][select_view]
            pose = np.dot(pose, self.global_RT_inv)

            select_view = np.random.randint(0, self.num_view - 1)
            # intrinsic
            proj = self.calib['projs'][select_view, ...]

            img = Image.open(img_path)

            uvmap = scipy.io.loadmat(uvmap_path)['uv_map']
            uvmap = TransferDenseposeUV(uvmap)
            uvmap = uvmap - np.floor(uvmap)
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

            img, mask, uvmap, _, _ = self.transform(img, proj, pose, mask=mask, uvmap=uvmap)
            uvmap = uvmap[:2, ...]
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

            uvmap_path = os.path.join(self.uvmap_dir, img_key + '.mat')
            # print(uvmap_path)

            uvmap = scipy.io.loadmat(uvmap_path)['uv_map']
            uvmap = TransferDenseposeUV(uvmap)
            uvmap = uvmap - np.floor(uvmap)
            uvmap = torch.tensor(uvmap).permute(1,2,0)

            # w, h = uvmap.shape[:2]
            # uvmap_3channel = np.concatenate([uvmap, np.zeros([w, h, 1])], axis=2) * 255
            # uvmap = Image.fromarray(uvmap_3channel.astype('uint8'))

            # params = get_params(self.cfg, uvmap.size)
            # uvmap_transform = get_transform(self.cfg, params, normalize=False, toTensor=True, isTrain=self.is_train)
            # uvmap = uvmap_transform(uvmap)
            # uvmap = uvmap[:2, ...]

            return {'uvmap': uvmap}

    def buffer_all(self):
        pass