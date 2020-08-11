import argparse
import os, time

import numpy as np
import cv2
import datetime

import _init_paths

from lib.utils.util import create_logger
from lib.config import cfg,update_config

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
    print(args)
    print(cfg)
    print("*" * 100)

    print('Set gpus...' + str(cfg.GPUS)[1:-1])
    print(' Batch size: '+ str(cfg.TRAIN.BATCH_SIZE))
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

    from lib.engine.loss import MultiLoss  

    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler
    # from utils.encoding import DataParallelModel
    # from utils.encoding import DataParallelCriterion

    from lib.dataset.DomeViewDataset import DomeViewDataset
    from lib.dataset.DPViewDataset import DPViewDataset  

    # device = torch.device('cuda: 2'+ str(cfg.GPUS[-1]))
    print("*" * 100)

    print("Build dataloader ...")
    if cfg.DATASET.DATASET == 'realdome_cx':
        view_dataset = DomeViewDataset(cfg = cfg, 
                                       root_dir = cfg.DATASET.ROOT,
                                       calib_path = cfg.DATASET.CALIB_PATH,
                                       calib_format = cfg.DATASET.CALIB_FORMAT,
                                       sampling_pattern = cfg.TRAIN.SAMPLING_PATTERN,
                                       precomp_high_dir = cfg.DATASET.PRECOMP_DIR,
                                       precomp_low_dir = cfg.DATASET.PRECOMP_DIR,
                                       preset_uv_path = cfg.DATASET.UV_PATH)
    elif cfg.DATASET.DATASET == 'densepose':
        view_dataset = DPViewDataset(cfg = cfg)
    # view_dataset = eval(cfg.DATASET.DATASET)(cfg = cfg)
    print("*" * 100)

    print('Build Network...')
    gpu_count = torch.cuda.device_count()
    
    dist.init_process_group(
        backend='nccl',
        init_method=cfg.DIST_URL,
        world_size=cfg.WORLD_SIZE,
        rank=cfg.RANK
    )    
    model_net = eval(cfg.MODEL.NAME)(cfg)

    if gpu_count > 1:
        model_net.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model_net, device_ids=list(cfg.GPUS)
            #, find_unused_parameters = True
        )
    elif gpu_count == 1:
        model = model_net.cuda()

    # Loss
    criterion = MultiLoss(cfg)
    criterion.cuda()

    # Optimizer
    ####################################################################
    # todo
    optimizerG = torch.optim.Adam(model_net.parameters(), lr = cfg.TRAIN.LR)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizerG, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=epoch_begin
    )
    print('Loading Checkpoint...')
    ####################################################################
    # todo
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        iter_init = checkpoint['iter']
        epoch_begin = checkpoint['epoch']
        optimizerG.load_state_dict(checkpoint['optimizer'])
        model_net.load_state_dict(checkpoint['state_dict'])
        print(' Load checkpoint path from %s'%(checkpoint_path))

    print('Start buffering data for training...')
    view_dataloader = DataLoader(view_dataset,
                                 batch_size = cfg.TRAIN.BATCH_SIZE * gpu_count, 
                                #  shuffle = cfg.TRAIN.SHUFFLE, 
                                 pin_memory=True,
                                 num_workers = cfg.WORKERS,
                                 sampler=DistributedSampler(view_dataset))
    view_dataset.buffer_all()
    writer = SummaryWriter(log_dir)

    # Activate some model parts
    if cfg.DATASET.DATASET == 'realdome_cx':
        view_data = view_dataset.read_view(0)
        cur_obj_path = view_data['obj_path']        
        frame_idx = view_data['f_idx']
        obj_data = view_dataset.objs[frame_idx]
        model_net.init_rasterizer(obj_data, view_dataset.global_RT)
    if type(model_net) == MergeNet:
        imgs, uv_maps = view_dataset.get_all_view()
        model_net.init_all_atlas(imgs, uv_maps)

    
    print('Begin training...')
    model.train()
    # model_net.set_mode(is_train = True)
    # model = DataParallelModel(model_net)
    # model.cuda()

    iter = iter_init    
    for epoch in range(epoch_begin, cfg.TRAIN.END_EPOCH):
        for view_trgt in view_dataloader:
            start = time.time()

            ROI = None
            img_gt = view_trgt['img'].cuda()

            # get image 
            if cfg.DATASET.DATASET == 'realdome_cx':
                uv_map, alpha_map, cur_obj_path = model.module.project_with_rasterizer(cur_obj_path, view_dataset.objs, view_trgt)
                ROI = view_trgt['ROI'].cuda()
            elif cfg.DATASET.DATASET == 'densepose':
                uv_map = view_trgt['uv_map'].cuda()
                alpha_map = view_trgt['mask'].cuda()

            # # check per iter image
            # for batch_idx, frame_idx in enumerate(frame_idxs):
            #     if self.cfg.DEBUG.SAVE_TRANSFORMED_IMG:
            #         save_dir_img_gt = './Debug/image_mask'
            #         save_path_img_gt = os.path.join(save_dir_img_gt, '%06d_%03d.png'%(iter, frame_idx))
            #         cv2.imwrite(save_path_img_gt,  cv2.cvtColor(img_gt[0][batch_idx, ...].cpu().detach().numpy().transpose(1,2,0)*255.0, cv2.COLOR_RGB2BGR))
            #         #cv2.imwrite(os.path.join(save_dir_img_gt, '%03d_'%frame_idx + img_fn), cv2.cvtColor(img_gt*255.0, cv2.COLOR_BGR2RGB))
            #         print(' Save img: '+ save_path_img_gt)
                    
            #     if self.cfg.DEBUG.SAVE_TRANSFORMED_MASK:
            #         save_alpha_map = alpha_map.permute(0,2,3,1).cpu().detach().numpy()
            #         save_dir_mask = './Debug/image_mask'
            #         save_path_mask = os.path.join(save_dir_mask, '%06d_%03d_mask.png'%(iter, frame_idx))
            #         cv2.imwrite(save_path_mask, save_alpha_map[batch_idx, ...]*255.0)
            #         print(' Save mask: '+ save_path_mask)

            outputs = model.forward(uv_map = uv_map,
                                    img_gt = img_gt,
                                    alpha_map = alpha_map,
                                    ROI = ROI)

            # ignore loss outside alpha_map and ROI
            if img_gt is not None:
                if alpha_map is not None:
                    img_gt = img_gt * alpha_map
                if ROI is not None:
                    img_gt = img_gt * ROI

            # Loss
            loss_g = criterion(outputs, img_gt)
            # loss_g = criterion_parall(outputs, img_gt)

            loss_rn = criterion.loss_rgb
            loss_rn_hsv = criterion.loss_hsv
            loss_atlas = criterion.loss_atlas

            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()
            lr_scheduler.step()

            # chcek gradiant
            if iter == iter_init:
                for name, param in model.named_parameters(): 
                    if param.grad is None:
                        print(name, True if param.grad is not None else False)

            if type(outputs) == list:
                for iP in range(len(outputs)):
                    outputs[iP] = outputs[iP].cuda()
                outputs = torch.cat(outputs, dim = 0)

            # get output images
            neural_img = outputs[:, 3:6, : ,:].clamp(min = 0., max = 1.)
            outputs = outputs[:, 0:3, : ,:]

            # error metrics
            with torch.no_grad():
                err_metrics_batch_i = metric.compute_err_metrics_batch(outputs * 255.0, img_gt * 255.0, alpha_map, compute_ssim = False)

            # tensorboard scalar logs of training data
            writer.add_scalar("loss_g", loss_g, iter)
            writer.add_scalar("loss_rn", loss_rn, iter)
            writer.add_scalar("loss_rn_hsv", loss_rn_hsv, iter)            
            writer.add_scalar("loss_atlas", loss_atlas, iter)
            writer.add_scalar("final_mae_valid", err_metrics_batch_i['mae_valid_mean'], iter)
            writer.add_scalar("final_psnr_valid", err_metrics_batch_i['psnr_valid_mean'], iter)

            if not iter % cfg.LOG.PRINT_FREQ:
                # neural_img = model_net.get_neural_img().clamp(min = 0., max = 1.)
                atlas = model_net.get_atalas()

                output_final_vs_gt = []
                output_final_vs_gt.append(outputs.clamp(min = 0., max = 1.))
                output_final_vs_gt.append(img_gt.clamp(min = 0., max = 1.))
                output_final_vs_gt.append(neural_img)
                output_final_vs_gt.append((outputs - img_gt).abs().clamp(min = 0., max = 1.))
                output_final_vs_gt.append((outputs - img_gt).abs().clamp(min = 0., max = 1.))
                output_final_vs_gt = torch.cat(output_final_vs_gt, dim = 0)
                writer.add_image("output_final_vs_gt",
                                torchvision.utils.make_grid(output_final_vs_gt,
                                                            nrow = outputs.shape[0],
                                                            range = (0, 1),
                                                            scale_each = False,
                                                            normalize = False).cpu().detach().numpy(),
                                                            iter)

                raster_uv_maps = torch.cat((uv_map.permute(0,3,1,2),  # N H W 2 -> N 2 H W
                                    torch.zeros(uv_map.shape[0], 1, uv_map.shape[1], uv_map.shape[2], dtype=uv_map.dtype, device=uv_map.device)),
                                    dim = 1)
                writer.add_image("raster_uv_vis",
                                torchvision.utils.make_grid(raster_uv_maps,
                                                            nrow = raster_uv_maps[0].shape[0],
                                                            range = (0, 1),
                                                            scale_each = False,
                                                            normalize = False).cpu().detach().numpy()[::-1, :, :], # uv0 -> 0vu (rgb)
                                                            iter)
                writer.add_image("atlas_vis",
                                torchvision.utils.make_grid(atlas,
                                                            nrow = raster_uv_maps[0].shape[0],
                                                            range = (0, 1),
                                                            scale_each = False,
                                                            normalize = False).cpu().detach().numpy()[:, :, :],
                                                            iter)
            iter += 1

            if iter % cfg.LOG.CHECKPOINT_FREQ == 0:
                model_net.save_checkpoint(os.path.join(log_dir, 'model_epoch_%d_iter_%s_.pth' % (epoch, iter)))
                # scipy.io.savemat('/data/NFS/new_disk/chenxin/relightable-nr/data/densepose_cx/logs/dnr/tmp/neural_img_epoch_%d_iter_%s_.npy'% (epoch, iter), {"neural_tex": model_net.neural_tex.cpu().clone().detach().numpy()})
                # model_net

            end = time.time()
            log_time = datetime.datetime.now().strftime('%m/%d') + '_' + datetime.datetime.now().strftime('%H:%M:%S') 
            print("%s Iter-%07d Epoch-%03d loss_g/rgb/hsv/tex %0.4f/%0.4f/%0.4f/%0.4f mae_valid %0.4f psnr_valid %0.4f t %0.2fs" 
                  % (log_time,
                     iter,
                     epoch,
                     loss_g, 
                     loss_rn, 
                     loss_rn_hsv, 
                     loss_atlas, 
                     err_metrics_batch_i['mae_valid_mean'], 
                     err_metrics_batch_i['psnr_valid_mean'], 
                     end - start))

    final_output_dir = os.path.join(log_dir, 'model_epoch_%d_iter_%s_.pth' % (epoch, iter))
    is_best_model = False
    save_checkpoint({
        'epoch': epoch + 1,
        'iter' iter + 1,
        'model': cfg.MODEL.NAME,
        'state_dict': model.state_dict(),
        'atlas': atlas,
        'optimizer': optimizerG.state_dict(),
        # 'best_state_dict': model.module.state_dict(),
        # 'perf': perf_indicator,
    }, is_best_model, final_output_dir)

if __name__ == '__main__':
    main()