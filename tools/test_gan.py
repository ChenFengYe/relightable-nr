import argparse
import os, time

import numpy as np
import cv2
import datetime

import _init_paths

from lib.utils.util import cond_mkdir, make_gif
from lib.config import cfg, update_config

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

def main():
    print('Load config...')
    args = parse_args()
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
    cond_mkdir(save_dir_img)
    
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
        model_net.load_state_dict(checkpoint['state_dict'], strict= False)
        print(' Load checkpoint path from %s'%(checkpoint_path))

    print('Start buffering data for inference...')
    view_dataloader = DataLoader(view_dataset, batch_size = cfg.TEST.BATCH_SIZE, shuffle = False, num_workers = cfg.WORKERS)
    view_dataset.buffer_all()

    model = model_net
    model.train()

    print('Begin inference...')
    inter = 0
    with torch.no_grad():
        for view_data in view_dataloader:
            start = time.time()
            model.set_input(view_data)
            model.test(view_data)
            outputs = model_net.fake_out.clone().detach().cpu()

            outputs_img = outputs[:, 0:3, : ,:]
            neural_img = outputs[:, 3:6, : ,:]

            # save
            for batch_idx in range(0, neural_img.shape[0]):
                cv2.imwrite(os.path.join(save_dir_img, str(inter).zfill(5) + '.png'), outputs_img[batch_idx, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)
                inter = inter + 1
                
            end = time.time()
            print("View %07d   t_total %0.4f" % (inter, end - start))

    make_gif(save_dir_img, save_dir_img+'.gif')    
if __name__ == '__main__':
    main()