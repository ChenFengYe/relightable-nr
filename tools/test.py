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
    # log_dir, iter, checkpoint_path = create_logger(cfg, args.cfg)
    # print(args)
    # print(cfg)
    # print("*" * 100)

    print('Set gpus...' + str(cfg.GPUS)[1:-1])
    print(' Batch size: '+ str(cfg.TEST.BATCH_SIZE))
    if not cfg.GPUS == 'None':        
        os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.GPUS)[1:-1]

    # import pytorch after set cuda
    import torch
    import torchvision

    from torch.utils.data import DataLoader
    from tensorboardX import SummaryWriter

    from lib.models import metric
    from lib.models.render_net import RenderNet
    from lib.models.feature_net import FeatureNet
    from lib.models.merge_net import MergeNet

    from utils.encoding import DataParallelModel

    from lib.dataset.DomeViewDataset import DomeViewDataset
    from lib.dataset.DPViewDataset import DPViewDataset  

    # device = torch.device('cuda: 2')
    # device = torch.device('cuda: '+ str(cfg.GPUS[-1]))
    # print("*" * 100)

    print("Build dataloader ...")
    # dataset for training views
    view_dataset = eval(cfg.DATASET.DATASET)(cfg = cfg, is_train=False)
    print("*" * 100)

    print('Build Network...')
    model_net = eval(cfg.MODEL.NAME)(cfg)

    print('Loading Model...')
    checkpoint_path = cfg.TEST.MODEL_PATH
    if os.path.exists(checkpoint_path):
        pass
    elif os.path.exists(os.path.join(cfg.TEST.MODEL_DIR, checkpoint_path)):
        checkpoint_path = os.path.join(cfg.TEST.MODEL_DIR,checkpoint_path)

    # model_net.load_checkpoint(checkpoint_path)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        iter_init = checkpoint['iter']
        epoch_begin = checkpoint['epoch']
        model_net.load_state_dict(checkpoint['state_dict'])
        print(' Load checkpoint path from %s'%(checkpoint_path))


    print('Start buffering data for inference...')
    view_dataloader = DataLoader(view_dataset, batch_size = cfg.TEST.BATCH_SIZE, shuffle = False, num_workers = 8)
    view_dataset.buffer_all()

    # Init Rasterizer
    if cfg.DATASET.DATASET == 'realdome_cx':
        view_data = view_dataset.read_view(0)
        cur_obj_path = view_data['obj_path']        
        frame_idx = view_data['f_idx']
        obj_data = view_dataset.objs[frame_idx]

        model_net.init_rasterizer(obj_data, view_dataset.global_RT)

    # model_net.set_parallel(cfg.GPUS)
    # model_net.set_mode(is_train = True)
    model = model_net
    # model = DataParallelModel(model_net)
    model.cuda()
    model.train()

    print('Begin inference...')
    inter = 0
    with torch.no_grad():
        for view_trgt in view_dataloader:
            start = time.time()

            ROI = None
            img_gt = None

            # get image 
            # if cfg.DATASET.DATASET == 'realdome_cx':
            #     uv_map, alpha_map, cur_obj_path = model.module.project_with_rasterizer(cur_obj_path, view_dataset.objs, view_trgt)
            # elif cfg.DATASET.DATASET == 'densepose':
            uv_map = view_trgt['uv_map'].cuda()
            # alpha_map = view_trgt['mask'][:,None,:,:].cuda()

            outputs = model.forward(uv_map = uv_map, img_gt = img_gt)
            neural_img = outputs[:, 3:6, : ,:].clamp(min = 0., max = 1.)            
            outputs = outputs[:, 0:3, : ,:]
            
            if type(outputs) == list:
                for iP in range(len(outputs)):
                    # outputs[iP] = outputs[iP].to(device)
                    outputs[iP] = outputs[iP].cuda()               
                outputs = torch.cat(outputs, dim = 0)

            # img_max_val = 2.0
            # outputs = (outputs * 0.5 + 0.5) * img_max_val # map to [0, img_max_val]

            # if alpha_map:
            #     outputs = outputs * alpha_map

            # save
            for batch_idx in range(0, outputs.shape[0]):
                cv2.imwrite(os.path.join(save_dir_img, str(inter).zfill(5) + '.png'), outputs[batch_idx, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)
                inter = inter + 1
                
            end = time.time()
            print("View %07d   t_total %0.4f" % (inter, end - start))

    make_gif(save_dir_img, save_dir_img+'.gif')    
if __name__ == '__main__':
    main()