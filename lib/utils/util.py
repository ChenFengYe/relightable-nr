import os
import numpy as np
from collections import OrderedDict
from shutil import copyfile

import imageio
import glob

import datetime
import sys

class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass  

def create_logger(cfg, cfg_path):
    dir_name = os.path.join(datetime.datetime.now().strftime('%m-%d') + 
                            '_' + datetime.datetime.now().strftime('%H-%M-%S') +
                            '_' + cfg.TRAIN.SAMPLING_PATTERN +
                            '_' + cfg.DATASET.ROOT.strip('/').split('/')[-1])
    # check whether to resume
    iter = 0
    epoch = cfg.TRAIN.BEGIN_EPOCH
    if cfg.TRAIN.EXP_NAME is not '':
        dir_name += '_' + cfg.TRAIN.EXP_NAME
    if cfg.AUTO_RESUME:
        checkpoint_path = ''
        if cfg.TRAIN.RESUME and cfg.TRAIN.CHECKPOINT:
             checkpoint_path = cfg.TRAIN.CHECKPOINT
             dir_name = cfg.TRAIN.CHECKPOINT_DIR
             nums = [int(s) for s in cfg.TRAIN.CHECKPOINT_NAME.split('_') if s.isdigit()]
             epoch = nums[0] + 1
             iter = nums[1] + 1
        elif cfg.MODEL.PRETRAINED:
            checkpoint_path = cfg.MODEL.PRETRAIN
            
    log_dir = os.path.join(cfg.LOG.LOGGING_ROOT, dir_name)
    cond_mkdir(log_dir)

    logfile_path = os.path.join(log_dir, dir_name+'.log')
    sys.stdout = Logger(logfile_path)
    print("Begin to reocrd ...")
    print("  writing log to " + logfile_path)

    cfgfile_path = os.path.join(log_dir, cfg.LOG.CFG_NAME)
    custom_copy(cfg_path, cfgfile_path)
    print("  backup cfg file to " + cfgfile_path)
          
    return log_dir, iter, epoch, checkpoint_path


def make_gif(input_path, save_path):
    with imageio.get_writer(save_path, mode='I') as writer:
        for filename in sorted(glob.glob(input_path +'/*.png')):
            writer.append_data(imageio.imread(filename))
            # os.remove(filename)
    writer.close()

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_network(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d" % params)
    return params

def custom_copy(filepath, targetpath, overwrite = True):
    if os.path.abspath(filepath) == os.path.abspath(targetpath):
        return
    if os.path.isfile(targetpath) and overwrite:
        os.remove(targetpath)
    copyfile(filepath, targetpath)
    
def custom_load(models, names, path, strict = True):
    import torch

    if type(models) is not list:
        models = [models]
    if type(names) is not list:
        names = [names]
    assert len(models) == len(names)

    whole_dict = torch.load(path, map_location='cpu')

    for i in range(len(models)):
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in whole_dict[names[i]].items():
            name = k
            #name = k.replace("submodule.", "")
            #name = name.replace("module.", "")
            new_state_dict[name] = v

        models[i].load_state_dict(new_state_dict, strict = strict)

    return whole_dict

def custom_save(path, parts, names):
    import torch

    if type(parts) is not list:
        parts = [parts]
    if type(names) is not list:
        names = [names]
    assert len(parts) == len(names)

    whole_dict = {}
    for i in range(len(parts)):
        if torch.is_tensor(parts[i]):
            whole_dict.update({names[i]: parts[i]})
        else:
            whole_dict.update({names[i]: parts[i].state_dict()})

    torch.save(whole_dict, path)

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    import torch
    torch.save(states, os.path.join(output_dir, filename))

    if is_best and 'state_dict' in states:
        torch.save(
            states['best_state_dict'],
            os.path.join(output_dir, 'model_best.pth.tar')
        )

##################################################
# Utility function for rotation matrices - from https://github.com/akar43/lsm/blob/b09292c6211b32b8b95043f7daf34785a26bce0a/utils.py #####
##################################################
import math

def quat2rot(q):
    '''q = [w, x, y, z]
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion'''
    eps = 1e-5
    w, x, y, z = q
    n = np.linalg.norm(q)
    s = (0 if n < eps else 2.0 / n)
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z
    R = np.array([[1 - (yy + zz), xy - wz,
                   xz + wy], [xy + wz, 1 - (xx + zz), yz - wx],
                  [xz - wy, yz + wx, 1 - (xx + yy)]])
    return R


def rot2quat(M):
    if M.shape[0] < 4 or M.shape[1] < 4:
        newM = np.zeros((4, 4))
        newM[:3, :3] = M[:3, :3]
        newM[3, 3] = 1
        M = newM

    q = np.empty((4, ))
    t = np.trace(M)
    if t > M[3, 3]:
        q[0] = t
        q[3] = M[1, 0] - M[0, 1]
        q[2] = M[0, 2] - M[2, 0]
        q[1] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
        q = q[[3, 0, 1, 2]]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def euler_to_rot(theta):
    R_x = np.array([[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]),
                     math.cos(theta[0])]])

    R_y = np.array([[math.cos(theta[1]), 0,
                     math.sin(theta[1])], [0, 1, 0],
                    [-math.sin(theta[1]), 0,
                     math.cos(theta[1])]])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]),
                     math.cos(theta[2]), 0], [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def az_el_to_rot(az, el):
    corr_mat = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    inv_corr_mat = np.linalg.inv(corr_mat)

    def R_x(theta):
        return np.array([[1, 0, 0], [0, math.cos(theta),
                                     math.sin(theta)],
                         [0, -math.sin(theta),
                          math.cos(theta)]])

    def R_y(theta):
        return np.array([[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0],
                         [math.sin(theta), 0,
                          math.cos(theta)]])

    def R_z(theta):
        return np.array([[math.cos(theta), -math.sin(theta), 0],
                         [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

    Rmat = np.matmul(R_x(-el * math.pi / 180), R_y(-az * math.pi / 180))
    return np.matmul(Rmat, inv_corr_mat)


def rand_euler_rotation_matrix(nmax=10):
    euler = (np.random.uniform(size=(3, )) - 0.5) * nmax * 2 * math.pi / 360.0
    Rmat = euler_to_rot(euler)
    return Rmat, euler * 180 / math.pi


def rot_mag(R):
    angle = (1.0 / math.sqrt(2)) * \
        norm(logm(R), 'fro') * 180 / (math.pi)
    return angle

##################################################
# camera operation for data augumentation
##################################################
def calc_center(mask):
    grid = np.mgrid[0:mask.shape[0],0:mask.shape[1]]
    grid_mask = mask[grid[0],grid[1]].astype(np.bool)
    X = grid[0,grid_mask]
    Y = grid[1,grid_mask]
    
    return np.mean(X),np.mean(Y)


def rodrigues_rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
