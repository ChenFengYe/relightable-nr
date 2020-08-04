import os
import numpy as np
import torch
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
    if cfg.TRAIN.EXP_NAME is not '':
        dir_name += '_' + cfg.TRAIN.EXP_NAME
    if cfg.AUTO_RESUME:
        checkpoint_path = ''
        if cfg.TRAIN.RESUME and cfg.TRAIN.CHECKPOINT:
             checkpoint_path = cfg.TRAIN.CHECKPOINT
             dir_name = cfg.TRAIN.CHECKPOINT_DIR
             nums = [int(s) for s in cfg.TRAIN.CHECKPOINT_NAME.split('_') if s.isdigit()]
             cfg.defrost()
             cfg.TRAIN.BEGIN_EPOCH = nums[0] + 1
             cfg.freeze()
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
          
    return log_dir, iter, checkpoint_path


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
