import os
import numpy as np
import torch

def print_network(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d" % params)
    return params

def custom_load(models, names, path, strict = True):
    import torch
    from collections import OrderedDict

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

# def custom_save(path, parts, names):

#     if type(parts) is not list:
#         parts = [parts]
#     if type(names) is not list:
#         names = [names]
#     assert len(parts) == len(names)

#     whole_dict = {}
#     for i in range(len(parts)):
#         if torch.is_tensor(parts[i]):
#             whole_dict.update({names[i]: parts[i]})
#         else:
#             whole_dict.update({names[i]: parts[i].state_dict()})

#     torch.save(whole_dict, path)

def save_checkpoint(states, is_best, output_path):
    import torch
    torch.save(states, output_path)

    if is_best and 'state_dict' in states:
        torch.save(
            states['best_state_dict'],
            output_path[:-4] + '_model_best.pth.tar'
        )