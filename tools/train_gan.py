import argparse
import os, time

import numpy as np
import cv2
import datetime

import _init_paths

from lib.utils.util import create_logger
from lib.utils import vis
from lib.config import cfg, update_config

import scipy.io

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
    args = parser.parse_args()
    return args

def main():
    print('Load config...')
    args = parse_args()
    update_config(cfg, args)

    print("Setup Log ...")
    log_dir, iter_init, epoch_begin, checkpoint_path = create_logger(cfg, args.cfg)
    save_img_dir = os.path.join(log_dir, 'images_trainning')
    os.mkdir(save_img_dir)
    print(args)
    print(cfg)
    print("*" * 100)

    print('Set gpus...' + str(cfg.GPUS)[1:-1])
    print(' Batch size: '+ str(cfg.TRAIN.BATCH_SIZE))
    # if not cfg.GPUS == 'None':        
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.GPUS)[1:-1]
    os.environ["CUDA_VISIBLE_DEVICES"]='1'

    # import pytorch after set cuda
    import torch

    from torch.utils.data import DataLoader
    from tensorboardX import SummaryWriter

    from lib.models import metric
    from lib.models.render_net import RenderNet
    from lib.models.feature_net import FeatureNet
    from lib.models.merge_net import MergeNet
    from lib.models.feature_pair_net import FeaturePairNet
    from lib.models.gan_net import Pix2PixModel

    from lib.engine.loss import MultiLoss  

    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler
    # from utils.encoding import DataParallelModel
    # from utils.encoding import DataParallelCriterion

    from lib.dataset.DomeViewDataset import DomeViewDataset
    from lib.dataset.DomeViewDatasetFVV import DomeViewDatasetFVV
    from lib.dataset.DPViewDataset import DPViewDataset

    from lib.utils.model import save_checkpoint
    # device = torch.device('cuda: 2'+ str(cfg.GPUS[-1]))
    print("*" * 100)
    torch.__version__ # Get PyTorch and CUDA version
    torch.cuda.is_available() # Check that CUDA works
    torch.cuda.device_count() # Check how many CUDA capable devices you have

    # Print device human readable names
    torch.cuda.get_device_name(0)
    # Add more lines with +1 like get_device_name(3), get_device_name(4) if you have more devices.

    print("Build dataloader ...")
    view_dataset = eval(cfg.DATASET.DATASET)(cfg = cfg, isTrain=True)
    viewFVV_dataset = eval(cfg.DATASET_FVV.DATASET)(cfg = cfg, isTrain=True)
    if cfg.TRAIN.VAL_FREQ > 0:
        print("Build val dataloader ...")
        view_val_dataset = eval(cfg.DATASET.DATASET)(cfg = cfg, isTrain=False)
    print("*" * 100)

    print('Build Network...')
    # gpu_count = torch.cuda.device_count()
    gpu_count = len(cfg.GPUS)    
    dist.init_process_group(
        backend='nccl',
        init_method=cfg.DIST_URL,
        world_size=cfg.WORLD_SIZE,
        rank=cfg.RANK
    )    

    model_net = eval(cfg.MODEL.NAME)(cfg)
    model = model_net
    model.setup(cfg)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # iter_init = checkpoint['iter']
        # epoch_begin = checkpoint['epoch']
        # to-do try directly load optimizer state_dict with same load_state_dict
        model_net.load_optimizer_state_dict(checkpoint['optimizer'])
        model_net.load_state_dict(checkpoint['state_dict'])
        print(' Load checkpoint path from %s'%(checkpoint_path))

    print('Start buffering data for training...')
    view_dataloader = DataLoader(view_dataset,
                                 batch_size = cfg.TRAIN.BATCH_SIZE * gpu_count, 
                                 shuffle = cfg.TRAIN.SHUFFLE, 
                                 pin_memory=True,
                                 num_workers = cfg.WORKERS)
    viewFVV_dataloader = DataLoader(viewFVV_dataset,
                                 batch_size = cfg.TRAIN.BATCH_SIZE * gpu_count, 
                                 shuffle = False, 
                                 num_workers = 0)
    # view_dataloader = DataLoader(view_dataset,
    #                              batch_size = cfg.TRAIN.BATCH_SIZE * gpu_count, 
    #                             #  shuffle = cfg.TRAIN.SHUFFLE, 
    #                              pin_memory=True,
    #                              num_workers = cfg.WORKERS,
    #                              sampler=DistributedSampler(view_dataset))
    view_dataset.buffer_all()
    if cfg.TRAIN.VAL_FREQ > 0:
        print('Start buffering data for validation...')     
        view_val_dataloader = DataLoader(view_val_dataset, 
                                         batch_size = cfg.TRAIN.BATCH_SIZE,
                                         shuffle = False,
                                         num_workers = cfg.WORKERS,
                                         sampler=DistributedSampler(view_val_dataset))
        view_val_dataset.buffer_all()

    writer = SummaryWriter(log_dir)
    
    print('Begin training...  Log in ' + log_dir)
    model.train()

    start = time.time()
    iter = iter_init
    for epoch in range(epoch_begin, cfg.TRAIN.END_EPOCH + 1):
        model.update_learning_rate()
        viewFVV_dataset.refresh()

        for view_data, viewFVV_data in zip(view_dataloader, viewFVV_dataloader):
            model.optimize_parameters(view_data)

            img_gt = view_data['img']
            alpha_map = view_data['mask'][:,None,:,:]
            uv_map = view_data['uv_map']
                       
            # chcek gradiant
            if iter == iter_init:
                print('Checking gradiant in first iteration')
                for name, param in model.named_parameters(): 
                    if param.grad is None:
                        print(name, True if param.grad is not None else False)

            outputs = model_net.get_current_results()
            loss_list = model_net.get_current_losses()
            outputs_img = outputs['rs'][:,0:3,:,:].clone().detach().cpu()
            outputs_mask = outputs['rs'][:,3:4,:,:].clone().detach().cpu()
            outputs_img *= outputs_mask
            outputs['img_rs'] = outputs_img
            outputs['mask_rs'] = outputs_mask
            # neural_img = outputs['nimg_rs'].clone().detach().cpu()            
            
            # Metrics
            log_time = datetime.datetime.now().strftime('%m/%d') + '_' + datetime.datetime.now().strftime('%H:%M:%S') 
            with torch.no_grad():
                err_metrics_batch_i = metric.compute_err_metrics_batch(outputs_img * 255.0, img_gt * 255.0, alpha_map, compute_ssim = False)

            # sythnesis views
            model.optimize_parameters(viewFVV_data)
            loss_list['G_views'] = float(model_net.loss_G_Multi)
            outputs_views = model_net.get_current_results()
            outputs_views_img = outputs_views['rs'][:,0:3,:,:].clone().detach().cpu()
            outputs['nimg_rs_view'] = outputs_views_img
            outputs['img_rs_view'] = outputs_views_img
            outputs['uv_map_view'] = viewFVV_data['uv_map']
            # vis
            if not iter % cfg.LOG.PRINT_FREQ:
                vis.writer_add_image_gan(writer, iter, epoch, inputs=view_data, results=outputs, save_folder=save_img_dir)

            # Log
            end = time.time()
            iter_time = end - start
            # vis.writer_add_scalar(writer, iter, epoch, err_metrics_batch_i, loss_list, log_time, iter_time)
            vis.writer_add_scalar_gan(writer, iter, epoch, err_metrics_batch_i, loss_list, log_time, iter_time)
 
            iter += 1           
            start = time.time()

            if iter % cfg.LOG.CHECKPOINT_FREQ==0:
                final_output_dir = os.path.join(log_dir, 'model_epoch_%d_iter_%s_.pth' % (epoch, iter))
                is_best_model = False
                save_checkpoint({
                    'epoch': epoch + 1,
                    'iter': iter + 1,
                    'model': cfg.MODEL.NAME,
                    'state_dict': model_net.state_dict(),
                    # 'atlas': atlas,
                    'optimizer': model_net.optimizer_state_dict()
                    # 'best_state_dict': model.module.state_dict(),
                    # 'perf': perf_indicator,
                }, is_best_model, final_output_dir)
                # scipy.io.savemat('/data/NFS/new_disk/chenxin/relightable-nr/data/densepose_cx/logs/dnr/tmp/neural_img_epoch_%d_iter_%s_.npy'% (epoch, iter), 
                # {"neural_tex": model_net.neural_tex.cpu().clone().detach().numpy()})
                # model_net

                # validation
                # if cfg.TRAIN.VAL_FREQ > 0:
                #     if not epoch % cfg.TRAIN.VAL_FREQ:
                #         print('Begin validation...')
                #         start_val = time.time()
                #         with torch.no_grad():
                #             # error metrics
                #             metric_val = {'mae_valid':[],'mse_valid':[],'psnr_valid':[],'ssim_valid':[]}
                #             loss_list_val ={'Loss':[], 'rgb':[], 'hsv':[], 'atlas':[]}

                #             val_iter = 0
                #             for view_val_trgt in view_val_dataloader:
                #                 img_gt = view_val_trgt['img'].cuda()
                #                 # alpha_map = None
                #                 # ROI = None

                #                 # img_gt = view_val_trgt['img'].cuda()
                #                 # # if cfg.DATASET.DATASET == 'DomeViewDataset':
                #                 # #     uv_map, alpha_map, cur_obj_path = model.module.project_with_rasterizer(cur_obj_path, view_dataset.objs, view_trgt)
                #                 # #     ROI = view_trgt['ROI'].cuda()
                #                 # # elif cfg.DATASET.DATASET == 'DomeViewDataset':
                #                 # uv_map = view_val_trgt['uv_map'].cuda()
                #                 # alpha_map = view_val_trgt['mask'][:,None,:,:].cuda()
                #                 # ROI = None


                #                 outputs = model.forward(view_data, isTrain=False)
                                
                #                 outputs_img = outputs[:, 0:3, : ,:]
                #                 neural_img = outputs[:, 3:6, : ,:]
                #                 aligned_uv = outputs[:, -2:, : ,:]

                #                 # ignore loss outside alpha_map and ROI
                #                 if alpha_map is not None:
                #                     img_gt = img_gt * alpha_map
                #                     outputs = outputs * alpha_map
                #                 if ROI is not None:
                #                     img_gt = img_gt * ROI
                #                     outputs = outputs * ROI

                #                 # Metrics
                #                 loss_val = criterion(outputs, img_gt)
                #                 loss_list_val_batch = criterion.loss_list()
                #                 metric_val_batch = metric.compute_err_metrics_batch(outputs_img * 255.0,
                #                                                                             img_gt * 255.0, alpha_map,
                #                                                                             compute_ssim=True)
                #                 batch_size = outputs_img.shape[0]
                #                 for i in range(batch_size):
                #                     for key in list(metric_val.keys()):
                #                         if key in metric_val_batch.keys():
                #                             metric_val[key].append(metric_val_batch[key][i])
                #                 for key, val in loss_list_val_batch.items():
                #                     loss_list_val[key].append(val)

                #                 if val_iter == 0:
                #                     iter_id = epoch
                #                     vis.writer_add_image(writer, iter_id, epoch, img_gt, outputs_img, neural_img, uv_map, aligned_uv, atlas = None, ex_name ='Val_')
                #                 val_iter = val_iter + 1

                #             # mean error
                #             for key in list(metric_val.keys()):
                #                 if metric_val[key]:
                #                     metric_val[key] = np.vstack(metric_val[key])
                #                     metric_val[key + '_mean'] = metric_val[key].mean()
                #                 else:
                #                     metric_val[key + '_mean'] = np.nan
                #             for key in loss_list_val.keys():
                #                 loss_list_val[key] = torch.tensor(loss_list_val[key]).mean()

                #             # vis
                #             end_val = time.time()
                #             val_time = end_val - start_val
                #             log_time = datetime.datetime.now().strftime('%m/%d') + '_' + datetime.datetime.now().strftime('%H:%M:%S') 
                #             iter_id = epoch
                #             vis.writer_add_scalar(writer, iter_id, epoch, metric_val, loss=loss_list_val, log_time=log_time, iter_time=val_time, ex_name ='Val')

if __name__ == '__main__':
    main()