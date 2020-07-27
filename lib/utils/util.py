import os
import numpy as np
import torch
from collections import OrderedDict
from shutil import copyfile

import imageio
import glob

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
