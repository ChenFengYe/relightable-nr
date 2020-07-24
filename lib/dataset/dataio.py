import os
import torch
import numpy as np
import numpy.linalg

from PIL import Image
import cv2
import scipy.io

from dataset import data_util
from dataset.transform import RandomTransform

import neural_renderer as nr

class ViewDataset():
    def __init__(self,
                 cfg,
                 root_dir,
                 calib_path,
                 calib_format,
                 sampling_pattern,
                 is_train = True,
                 ignore_dist_coeffs = True,
                 precomp_high_dir = None,
                 precomp_low_dir = None,
                 preset_uv_path = None,
                 ):
        super().__init__()

        self.cfg = cfg
        self.root_dir = root_dir
        self.calib_format = calib_format
        self.img_dir = cfg.DATASET.IMG_DIR
        self.img_size = list(cfg.DATASET.OUTPUT_SIZE)
        self.ignore_dist_coeffs = ignore_dist_coeffs
        self.is_train = is_train

        self.preset_uv_path = preset_uv_path
        self.precomp_high_dir = precomp_high_dir
        self.precomp_low_dir = precomp_low_dir

        self.img_gamma = cfg.DATASET.GAMMA        
        self.frame_range = cfg.DATASET.FRAME_RANGE
        self.cam_range = cfg.DATASET.CAM_RANGE
        self.cam_idxs = []
        self.frame_idxs = []
        self.frame_num = len(self.cam_range) & len(self.frame_range)

        self.views_all = []
        if not os.path.isdir(root_dir):
            raise ValueError("Error! root dir is wrong")

        # load calibration data
        if calib_format == 'convert':
            if not os.path.isfile(calib_path):
                raise ValueError("Error! calib path is wrong " + calib_path)
            self.calib = scipy.io.loadmat(calib_path)
            self.global_RT = self.calib['global_RT']
            self.num_view = self.calib['poses'].shape[0]
        else:
            raise ValueError('Unknown calib format')
        self.global_RT_inv = np.linalg.inv(self.global_RT)
        self.global_RT = torch.from_numpy(self.global_RT.astype(np.float32))

        # get frames
        self.img_fp_all = []
        self.obj_fp_all = []
        if self.is_train:
            self.frame_num =  0                
            for frame_idx in self.frame_range:
                for cam_idx in self.cam_range:
                    args_num = self.img_dir.count('%')
                    if args_num == 1:
                        img_path = self.img_dir % (frame_idx)
                    elif args_num == 2:
                        img_path = self.img_dir % (frame_idx, cam_idx)
                    else:
                        raise ValueError('Error : image pattern ' + self.img_dir + ' is not support!')
                    obj_path = cfg.DATASET.MESH_DIR %(frame_idx)
                    if not os.path.isfile(img_path):
                        raise ValueError('Not existed image path : ' + img_path)
                    if not os.path.isfile(obj_path):
                        raise ValueError('Not existed mesh path : ' + obj_path)
                    self.frame_idxs.append(frame_idx)
                    self.img_fp_all.append(img_path)
                    self.obj_fp_all.append(obj_path)
                    self.cam_idxs.append(cam_idx)
        # test
        else:
            self.img_fp_all = ['x.x'] * self.num_view
            self.cam_idxs = range(0,self.num_view)
            self.frame_idxs = np.resize(cfg.TEST.FRAME_RANGE, self.num_view)
            self.obj_fp_all = [cfg.DATASET.MESH_DIR%(frame_idx) for frame_idx in self.frame_idxs]

        # get intrinsic/extrinsic of all input images
        self.num_view = len(self.calib['poses'])
        self.poses_all = []
        img_fp_all_new = []
        for idx in range(len(self.img_fp_all)):
            img_fn = os.path.split(self.img_fp_all[idx])[-1]
            self.poses_all.append(self.calib['poses'][self.cam_idxs[idx], :, :])
            img_fp_all_new.append(self.img_fp_all[idx])

        # remove views without calibration result
        self.img_fp_all = img_fp_all_new

        # Subsample data
        self.img_fp_all, self.poses_all, self.keep_idxs = data_util.samping_img_set(self.img_fp_all, self.poses_all, sampling_pattern)
        self.cam_idxs = np.array(self.cam_idxs)
        
        if self.calib_format == 'convert':
            #cam_idx = self.keep_idxs % self.num_view            
            cam_idxs = self.cam_idxs[self.keep_idxs]
            self.calib['img_hws'] = self.calib['img_hws'][cam_idxs, ...]
            self.calib['projs'] = self.calib['projs'][cam_idxs, ...]
            self.calib['poses'] = self.calib['poses'][cam_idxs, ...]
            self.calib['dist_coeffs'] = self.calib['dist_coeffs'][cam_idxs, ...]

        # get mapping from img_fn to idx and vice versa
        # self.img_fn2idx = {}
        # self.img_idx2fn = []
        # for idx in range(len(self.img_fp_all)):
        #     img_fn = os.path.split(self.img_fp_all[idx])[-1]
        #     self.img_fn2idx[img_fn] = idx
        #     self.img_idx2fn.append(img_fn)

        print("*" * 100)
        print("Sampling pattern ", sampling_pattern)
        print("Image size ", self.img_size)
        print("*" * 100)

        print(" Buffering meshs...")
        # if preset_uv_path:
        #    self.v_attr, self.f_attr = nr.load_obj(cur_obj_fp, normalization = False)
        self.objs = {}
        if cfg.DATASET.PRELOAD_MESHS:
            for keep_idx in self.keep_idxs:
                frame_idx = self.frame_idxs[keep_idx]
                if frame_idx in self.objs:
                    continue
                cur_obj_fp = cfg.DATASET.MESH_DIR%(frame_idx)
                obj_data = {}
                obj_data['v_attr'] , obj_data['f_attr'] = nr.load_obj(cur_obj_fp, normalization = False, use_cuda = False)
                self.objs[frame_idx] = obj_data
                if cfg.VERBOSE:
                    print(' Loading mesh: ' + str(frame_idx) + ' ' + cur_obj_fp)

        print(" building transform for images...")
        self.transform = RandomTransform(cfg.DATASET.OUTPUT_SIZE, 
                                    cfg.DATASET.MAX_SHIFT, 
                                    cfg.DATASET.MAX_SCALE,
                                    cfg.DATASET.MAX_ROTATION)
        
        if cfg.DEBUG.DEBUG:
            print(self.cam_idxs)
            print(self.img_fp_all)
            print(self.frame_idxs)        

    def buffer_all(self):
        # Buffer files
        print("Buffering files...")
        self.views_all = []
        for i in range(self.__len__()):
            if not i % 50:
                print('Data', i)
            self.views_all.append(self.read_view(i))

    def buffer_one(self):
        self.views_all = []
        self.views_all.append(self.read_view(0))

    def read_view(self, idx):
        # keep_idx = self.keep_idxs[idx]
        img_fp = self.img_fp_all[idx]

        img_fn = os.path.split(img_fp)[-1]
    
        # image size
        if self.calib_format == 'convert':
            img_hw = self.calib['img_hws'][idx, :]

        # extrinsic
        pose = self.poses_all[idx]
        pose = np.dot(pose, self.global_RT_inv)

        # intrinsic
        proj = self.calib['projs'][idx, ...]

        # get view image
        if self.is_train:
            # img_gt, center_coord, center_coord_new, img_crop_size = data_util.load_img(img_fp, square_crop = True, downsampling_order = 1, target_size = self.img_size)
            # img_gt = img_gt[:, :, :3]
            # img_gt = img_gt.transpose(2,0,1)
            # img_gt = img_gt ** self.img_gamma
            img_gt = Image.open(img_fp)
            img_gt, proj, pose, mask, ROI = self.transform(img_gt, proj, pose)

            # load mask
            # if self.cfg.DEBUG.DEBUG:
            #     mask_fp = os.path.join(os.path.dirname(img_fp), 'mask/', os.path.basename(img_fp))
            #     print(mask_fp)
            #     mask_orig, center_coord, center_coord_new, img_crop_size = data_util.load_img(mask_fp, square_crop = True, downsampling_order = 1, target_size = self.img_size)
            #     mask_orig = mask_orig[:,:,None]
        else:
            min_dim = np.amin(img_hw)
            center_coord = img_hw // 2
            center_coord_new = np.array([min_dim // 2, min_dim // 2])
            img_crop_size = np.array([min_dim, min_dim])
            
            offset = np.array([center_coord_new[0] - center_coord[0], center_coord_new[1] - center_coord[1]], dtype = np.float32)
            scale = np.array([self.img_size[0] * 1.0 / (img_crop_size[0] * 1.0), self.img_size[1] * 1.0 / (img_crop_size[1] * 1.0)], dtype = np.float32)

            proj[0, -1] = (proj[0, -1] + offset[1]) * scale[1]
            proj[1, -1] = (proj[1, -1] + offset[0]) * scale[0]
            proj[0, 0] *= scale[1]
            proj[1, 1] *= scale[0]


        dist_coeffs = self.calib['dist_coeffs'][idx, :]
        dist_coeffs = dist_coeffs if not self.ignore_dist_coeffs else np.zeros(dist_coeffs.shape)

        view_dir = -pose[2, :3]
        proj_inv = numpy.linalg.inv(proj)
        R_inv = pose[:3, :3].transpose()

        frame_idx = self.frame_idxs[self.keep_idxs[idx]]
        obj_path = self.obj_fp_all[self.keep_idxs[idx]]        
        view = {'proj': torch.from_numpy(proj.astype(np.float32)),
                'pose': torch.from_numpy(pose.astype(np.float32)),
                'dist_coeffs': torch.from_numpy(dist_coeffs.astype(np.float32)),
                #'offset': torch.from_numpy(offset),
                #'scale': torch.from_numpy(scale),
                #'view_dir': torch.from_numpy(view_dir.astype(np.float32)),
                'proj_inv': torch.from_numpy(proj_inv.astype(np.float32)),
                'R_inv': torch.from_numpy(R_inv.astype(np.float32)),
                'idx': idx,
                'f_idx': frame_idx,
                'img_fn': img_fn,
                'obj_path': obj_path}

        if self.is_train:
                view['img_gt'] = img_gt
                view['ROI'] = ROI

        # load precomputed data
        if self.cfg.DATASET.LOAD_PRECOMPUTE:
            precomp_low_dir = self.precomp_low_dir % frame_idx
            precomp_high_dir = self.precomp_high_dir % frame_idx

            # cannot share across meshes
            raster = scipy.io.loadmat(os.path.join(precomp_low_dir, 'resol_' + str(self.img_size[0]), 'raster', img_fn.split('.')[0] + '.mat'))
            view['face_index_map'] = torch.from_numpy(raster['face_index_map'])
            view['weight_map'] = torch.from_numpy(raster['weight_map'])
            view['faces_v_idx'] = torch.from_numpy(raster['faces_v_idx'])
            view['v_uvz'] = torch.from_numpy(raster['v_uvz'])
            view['v_front_mask'] = torch.from_numpy(raster['v_front_mask'])[0, :]

            # can share across meshes when only changing resolution
            TBN_map = scipy.io.loadmat(os.path.join(precomp_high_dir, 'resol_' + str(self.img_size[0]), 'TBN_map', img_fn.split('.')[0] + '.mat'))['TBN_map']
            view['TBN_map'] = torch.from_numpy(TBN_map)
            uv_map = scipy.io.loadmat(os.path.join(precomp_high_dir, 'resol_' + str(self.img_size[0]), 'uv_map', img_fn.split('.')[0] + '.mat'))['uv_map']
            uv_map = uv_map - np.floor(uv_map) # keep uv in [0, 1] ?? question ?? why to - np.floor
            view['uv_map'] = torch.from_numpy(uv_map)
            normal_map = scipy.io.loadmat(os.path.join(precomp_high_dir, 'resol_' + str(self.img_size[0]), 'normal_map', img_fn.split('.')[0] + '.mat'))['normal_map']
            view['normal_map'] = torch.from_numpy(normal_map)
            view_dir_map = scipy.io.loadmat(os.path.join(precomp_high_dir, 'resol_' + str(self.img_size[0]), 'view_dir_map', img_fn.split('.')[0] + '.mat'))['view_dir_map']
            view['view_dir_map'] = torch.from_numpy(view_dir_map)
            view_dir_map_tangent = scipy.io.loadmat(os.path.join(precomp_high_dir, 'resol_' + str(self.img_size[0]), 'view_dir_map_tangent', img_fn.split('.')[0] + '.mat'))['view_dir_map_tangent']
            view['view_dir_map_tangent'] = torch.from_numpy(view_dir_map_tangent)
            sh_basis_map = scipy.io.loadmat(os.path.join(precomp_high_dir, 'resol_' + str(self.img_size[0]), 'sh_basis_map', img_fn.split('.')[0] + '.mat'))['sh_basis_map'].astype(np.float32)
            view['sh_basis_map'] = torch.from_numpy(sh_basis_map)
            reflect_dir_map = scipy.io.loadmat(os.path.join(precomp_high_dir, 'resol_' + str(self.img_size[0]), 'reflect_dir_map', img_fn.split('.')[0] + '.mat'))['reflect_dir_map']
            view['reflect_dir_map'] = torch.from_numpy(reflect_dir_map)
            alpha_map = cv2.imread(os.path.join(precomp_high_dir, 'resol_' + str(self.img_size[0]), 'alpha_map', img_fn.split('.')[0] + '.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            view['alpha_map'] = torch.from_numpy(alpha_map)

            # print('----------------------')
            # print(idx)
            # alpha_map_fp = os.path.join(precomp_high_dir, 'resol_' + str(self.img_size[0]), 'alpha_map', img_fn.split('.')[0] + '.png')
            # print(img_fp)
            # print(alpha_map_fp)

            # Data checker -  For debugging
            # image size
            # if view['img_gt'].shape[1:2] != view['uv_map'].shape[0:1]:
            #     print(view['img_gt'].shape)
            #     print(view['uv_map'].shape)
            #     raise ValueError("Image and uv map have different size!")

            # print(type((view['img_gt'] * (1 - view['alpha_map']))))
            # cv2.imread('/data/chenxin/relightable-nr/tmp/'+)
            # if not (view['img_gt'] * (1 - view['alpha_map'])).type(torch.uint8).any():
            #     raise ValueError("Alpha map not correct!")
            # if not (view['uv_map'] * (1-view['alpha_map'])).any():
            #     print(alpha_map_fp)
            #     print(uv_map_fp)
            #     raise ValueError("Alpha map not correct. \n uv_path: "+ img_fp + "\n alpha map path:"+ alpha_map_fp)

            # mask crop
            # save_dir_img_gt = './Debug/resol_512/cut_image/'
            # cv2.imwrite(os.path.join(save_dir_img_gt, '%03d_'%frame_idx + img_fn), cv2.cvtColor(img_gt.transpose(1,2,0)*255.0 * mask_orig, cv2.COLOR_BGR2RGB))
        return view

    def __len__(self):
        return len(self.img_fp_all)

    def __getitem__(self, idx):
        view_trgt = []

        # to-do fix first frame error
        if len(self.views_all) > idx:
            view_trgt.append(self.views_all[idx])
        else:
            view_trgt.append(self.read_view(idx))
            
        return view_trgt


class LightProbeDataset():
    def __init__(self,
                 data_dir,
                 img_gamma = 1.0):
        super().__init__()

        self.data_dir = data_dir
        self.img_gamma = img_gamma

        if not os.path.isdir(data_dir):
            raise ValueError("Error! data dir is wrong")

        # get path for all light probes
        self.lp_fp_all = sorted(data_util.glob_imgs(self.data_dir))

        self.lp_all = [None] * len(self.lp_fp_all)


    def buffer_one(self, idx):
        if self.lp_all[idx] is not None:
            return
            
        # get light probe
        lp_fp = self.lp_fp_all[idx]
        print(lp_fp)
        if lp_fp[-4:] == '.exr' or lp_fp[-4:] == '.hdr':
            lp_img = cv2.imread(lp_fp, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        else:
            lp_img = cv2.imread(lp_fp, cv2.IMREAD_UNCHANGED)[:, :, :3].astype(np.float32) / 255.0
        lp_img = cv2.cvtColor(lp_img, cv2.COLOR_BGR2RGB).transpose(2,0,1)
        lp_img = lp_img ** self.img_gamma

        lp = {'lp_img': torch.from_numpy(lp_img)}

        self.lp_all[idx] = lp


    def buffer_all(self):
        for idx in range(len(self.lp_fp_all)):
            self.buffer_one(idx)


    def __len__(self):
        return len(self.lp_fp_all)


    def __getitem__(self, idx):
        self.buffer_one(idx)
        return self.lp_all[idx]
