import cv2
import numpy as np
from skimage import transform
from glob import glob
import os
import math
from scipy.linalg import logm, norm
import scipy.io

def samping_img_set(img_fp_all, poses_all, sampling_pattern):
    keep_idxs = []
    if sampling_pattern == 'all':
        keep_idxs = list(range(len(img_fp_all)))
    else:
        # if sampling_pattern == 'filter':
        #     img_fp_all_new = []
        #     poses_all_new = []
        #     for idx in self.calib['keep_id'][0, :]:
        #         img_fp_all_new.append(img_fp_all[idx])
        #         poses_all_new.append(poses_all[idx])
        #         keep_idxs.append(idx)
        #     img_fp_all = img_fp_all_new
        #     poses_all = poses_all_new
        if sampling_pattern.split('_')[0] == 'first':
            first_val = int(sampling_pattern.split('_')[-1])
            img_fp_all = img_fp_all[:first_val]
            poses_all = poses_all[:first_val]
            keep_idxs = list(range(first_val))
        elif sampling_pattern.split('_')[0] == 'after':
            after_val = int(sampling_pattern.split('_')[-1])
            keep_idxs = list(range(after_val, len(img_fp_all)))
            img_fp_all = img_fp_all[after_val:]
            poses_all = poses_all[after_val:]
        elif sampling_pattern.split('_')[0] == 'skip':
            skip_val = int(sampling_pattern.split('_')[-1])
            img_fp_all_new = []
            poses_all_new = []
            for idx in range(0, len(img_fp_all), skip_val):
                img_fp_all_new.append(img_fp_all[idx])
                poses_all_new.append(poses_all[idx])
                keep_idxs.append(idx)
            img_fp_all = img_fp_all_new
            poses_all = poses_all_new
        elif sampling_pattern.split('_')[0] == 'skipinv':
            skip_val = int(sampling_pattern.split('_')[-1])
            img_fp_all_new = []
            poses_all_new = []
            for idx in range(0, len(img_fp_all)):
                if idx % skip_val == 0:
                    continue
                img_fp_all_new.append(img_fp_all[idx])
                poses_all_new.append(poses_all[idx])
                keep_idxs.append(idx)
            img_fp_all = img_fp_all_new
            poses_all = poses_all_new
        elif sampling_pattern.split('_')[0] == 'only':
            choose_idx = int(sampling_pattern.split('_')[-1])
            img_fp_all = [img_fp_all[choose_idx]]
            poses_all = [poses_all[choose_idx]]
            keep_idxs.append(choose_idx)
        else:
            raise ValueError("Unknown sampling pattern!")
        keep_idxs = np.array(keep_idxs)
    return img_fp_all, poses_all, keep_idxs

def square_img_crop(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    center_coord_new = np.array([min_dim // 2, min_dim // 2])

    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img, center_coord, center_coord_new


def load_img(filepath, target_size=None, anti_aliasing=True, downsampling_order=None, square_crop=False):
    if filepath[-4:] == '.mat':
        img = scipy.io.loadmat(filepath)['img'][:, :, ::-1]
    elif filepath[-4:] == '.exr' or filepath[-4:] == '.hdr':
        img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        # /255.
        # print('Tip!!!!!!!! 255 rescale image!! for mars images')

    if img is None:
        print("Error: Path %s invalid" % filepath)
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if square_crop:
        img, center_coord, center_coord_new = square_img_crop(img)
    else:
        center_coord = np.array(img.shape[:2]) // 2
        center_coord_new = center_coord

    img_crop_size = img.shape

    if target_size is not None:
        if downsampling_order == 1:
            img = cv2.resize(img, tuple(target_size), interpolation=cv2.INTER_AREA)
        else:
            img = transform.resize(img, target_size,
                                   order=downsampling_order,
                                   mode='reflect',
                                   clip=False, preserve_range=True,
                                   anti_aliasing=anti_aliasing)
                                   
    return img, center_coord, center_coord_new, img_crop_size


def glob_imgs(path, exts = ['*.png', '*.jpg', '*.JPEG', '*.bmp', '*.exr', '*.hdr', '*.mat']):
    imgs = []
    for ext in exts:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def get_archimedean_spiral(sphere_radius, origin = np.array([0., 0., 0.]), num_step = 1000):
    a = 300
    r = sphere_radius
    o = origin

    translations = []

    i = a / 2
    while i > 0.:
        x = r * np.cos(i) * np.cos((-np.pi / 2) + i / a * np.pi)
        y = r * np.sin(i) * np.cos((-np.pi / 2) + i / a * np.pi)
        z = r * - np.sin(-np.pi / 2 + i / a * np.pi)

        xyz = np.array((x,y,z)) + o

        translations.append(xyz)
        i -= a / (2.0 * num_step)

    return translations


def interpolate_views(pose_1, pose_2, num_steps=100):
    poses = []
    for i in np.linspace(0., 1., num_steps):
        pose_1_contrib = 1 - i
        pose_2_contrib = i

        # Interpolate the two matrices
        target_pose = pose_1_contrib * pose_1 + pose_2_contrib * pose_2

        # Renormalize the rotation matrix
        target_pose[:3,:3] /= np.linalg.norm(target_pose[:3,:3], axis=0, keepdims=True)
        poses.append(target_pose)

    return poses


def get_nn_ranking(poses):
    # Calculate the ranking of nearest neigbors
    parsed_poses = np.stack([pose[:3,2] for pose in poses], axis=0)
    parsed_poses /= np.linalg.norm(parsed_poses, axis=1, ord=2, keepdims=True)
    cos_sim_mat = parsed_poses.dot(parsed_poses.T)
    np.fill_diagonal(cos_sim_mat, -1.)
    nn_idcs = cos_sim_mat.argsort(axis=1).astype(int)  # increasing order
    cos_sim_mat.sort(axis=1)

    return nn_idcs, cos_sim_mat