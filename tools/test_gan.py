import argparse
import os, time

import numpy as np
import cv2
import datetime

import _init_paths

from lib.utils.util import cond_mkdir, make_gif
from lib.config import cfg, update_config
from lib.utils import vis

def parse_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    args = parser.parse_args()
    return args

def build_model(args):
    print('Load config...')
    update_config(cfg, args)

    # print("Setup Log ...")
    log_dir = cfg.TEST.CALIB_DIR.split('/')
    log_dir = os.path.join(cfg.TEST.MODEL_DIR, cfg.TRAIN.EXP_NAME+'_'+cfg.TEST.CALIB_NAME[:-4]+'_resol_'+str(cfg.DATASET.OUTPUT_SIZE[0])+'_'+log_dir[-2]+'_'+
                           log_dir[-1].split('_')[0] + '_' + log_dir[-1].split('_')[1] + '_' +
                           cfg.TEST.MODEL_NAME.split('-')[-1].split('.')[0])
    cond_mkdir(log_dir)
    if len(cfg.TEST.FRAME_RANGE) > 1:
        save_dir_img_ext = cfg.TEST.SAVE_FOLDER + '_' + str(cfg.TEST.FRAME_RANGE[0]) +'_'+ str(cfg.TEST.FRAME_RANGE[-1])
    else:
        save_dir_img_ext = cfg.TEST.SAVE_FOLDER
    save_dir_img = os.path.join(log_dir, save_dir_img_ext)
    save_dir_uv = save_dir_img+'_uv'
    save_dir_nimg = save_dir_img+'_nimg'
    cond_mkdir(save_dir_img)
    cond_mkdir(save_dir_uv)
    cond_mkdir(save_dir_nimg)
    
    import torch
    import torchvision

    from torch.utils.data import DataLoader
    from tensorboardX import SummaryWriter

    from lib.models import metric
    from lib.models.render_net import RenderNet
    from lib.models.feature_net import FeatureNet
    from lib.models.merge_net import MergeNet
    from lib.models.gan_net import Pix2PixModel

    from utils.encoding import DataParallelModel

    from lib.dataset.DomeViewDataset import DomeViewDataset
    from lib.dataset.DPViewDataset import DPViewDataset  

    print("Build dataloader ...")
    view_dataset = eval(cfg.DATASET.DATASET)(cfg, isTrain=False)
    print("*" * 100)

    print('Build Network...')
    model_net = eval(cfg.MODEL.NAME)(cfg, isTrain=False)
    model = model_net
    model.setup(cfg)
    
    print('Loading Model...')
    checkpoint_path = cfg.TEST.MODEL_PATH
    if os.path.exists(checkpoint_path):
        pass
    elif os.path.exists(os.path.join(cfg.TEST.MODEL_DIR, checkpoint_path)):
        checkpoint_path = os.path.join(cfg.TEST.MODEL_DIR,checkpoint_path)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # model_net.load_optimizer_state_dict(checkpoint['optimizer'])

        def fix_module_state_dict(state_dict):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            netG_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:len('netG.module')] == 'netG.module':
                    name = k.replace("netG.module.", "")
                    netG_state_dict[name] = v
            new_state_dict['netG'] = netG_state_dict
            return new_state_dict
        
        state_dict = fix_module_state_dict(checkpoint['state_dict'])
        model_net.load_state_dict(state_dict)

        # model_net.load_state_dict(checkpoint['state_dict'])
        print(' Load checkpoint path from %s'%(checkpoint_path))

    print('Start buffering data for inference...')
    view_dataloader = DataLoader(view_dataset, batch_size = cfg.TEST.BATCH_SIZE, shuffle = False, num_workers = cfg.WORKERS)
    view_dataset.buffer_all()

    model.eval()

    save_dict = {}
    save_dict['img'] = save_dir_img
    save_dict['nimg'] = save_dir_nimg 
    save_dict['uv'] = save_dir_uv
    return model, view_dataloader, view_dataset, save_dict

def do_test(model, view_dataloader, save_dict):
    print('Begin inference...')
    save_dir_img = save_dict['img']
    save_dir_nimg = save_dict['nimg']
    save_dir_uv = save_dict['uv']

    inter = 0
    with torch.no_grad():
        for view_data in view_dataloader:
            start = time.time()
            model.set_input(view_data)
            model.test(view_data)
            outputs = model.get_current_results()

            outputs_img = outputs['img_rs']
            neural_img = outputs['nimg_rs']
            uv_map = view_data['uv_map']

            # save
            for batch_idx in range(0, neural_img.shape[0]):
                img_rs_cv = outputs_img[batch_idx, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.
                cv2.imwrite(os.path.join(save_dir_img, str(inter).zfill(5) + '.png'), img_rs_cv)

                nimg_rs_cv = neural_img[batch_idx, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.
                cv2.imwrite(os.path.join(save_dir_nimg, str(inter).zfill(5) + '.png'), nimg_rs_cv)

                uv_map_cv = vis.tensor_2ch_to_3ch(uv_map[batch_idx, :]).cpu().detach().numpy()* 255.
                cv2.imwrite(os.path.join(save_dir_uv, str(inter).zfill(5) + '.png'), uv_map_cv)
                inter = inter + 1
                
            end = time.time()
            print("View %07d   t_total %0.4f" % (inter, end - start))

    make_gif(save_dir_img, save_dir_img+'.gif')
    make_gif(save_dir_uv, save_dir_uv+'.gif')
    make_gif(save_dir_nimg, save_dir_nimg+'.gif')

def main():
    args = parse_args()

    model, view_dataloader, _, save_dict = build_model(args)
    do_test(model, view_dataloader, save_dict)

if __name__ == '__main__':
    main()