import argparse
import os, time

import numpy as np
import cv2
import datetime

import _init_paths

from lib.utils.util import create_logger
from lib.config import cfg,update_config

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

    print("Setup Log ...")
    log_dir, iter, checkpoint_path = create_logger(cfg, args.cfg)
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

    from utils.encoding import DataParallelModel
    from utils.encoding import DataParallelCriterion

    from lib.dataset.DomeViewDataset import DomeViewDataset
    from lib.dataset.DPViewDataset import DPViewDataset  
    print("*" * 100)

    print("Build dataloader ...")
    # dataset for training views
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
    model_net = eval(cfg.MODEL.NAME)(cfg)

    # Loss
    criterionL1 = torch.nn.L1Loss(reduction='mean')
    criterionL1 = DataParallelCriterion(criterionL1)
    criterionL1.cuda()
    
    # Optimizer
    optimizerG = torch.optim.Adam(model_net.parameters(), lr = cfg.TRAIN.LR)

    print('Loading Model...')
    model_net.load_checkpoint(checkpoint_path)
    # model_net.set_parallel(cfg.GPUS)
    model_net.set_mode(is_train = True)
    model = DataParallelModel(model_net)
    model.cuda()

    print('Start buffering data for training...')
    view_dataloader = DataLoader(view_dataset, batch_size = cfg.TRAIN.BATCH_SIZE, shuffle = cfg.TRAIN.SHUFFLE, num_workers = 8)
    view_dataset.buffer_all()
    writer = SummaryWriter(log_dir)

    # Init Rasterizer
    if cfg.DATASET.DATASET == 'realDome_cx':
        view_data = view_dataset.read_view(0)
        cur_obj_path = view_data['obj_path']        
        frame_idx = view_data['f_idx']
        obj_data = view_dataset.objs[frame_idx]

        model_net.init_rasterizer(obj_data, view_dataset.global_RT)

    print('Begin training...')
    # init value
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        for view_trgt in view_dataloader:
            start = time.time()

            ROI = None
            img_gt = view_trgt['img'].cuda()

            # get image 
            if cfg.DATASET.DATASET == 'realDome_cx':
                pass
                # ????? how do when using Parallel
                # uv_map, alpha_map, cur_obj_path = model.moudle.project_uv(cur_obj_path, view_dataset.objs, view_trgt)
                # ROI = view_trgt['ROI'].cuda()
            elif cfg.DATASET.DATASET == 'densepose':
                uv_map = view_trgt['uv_map'].permute(0, 2, 3, 1).cuda()
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

            # prepare gt image for loss
            if img_gt is not None:
                if alpha_map is not None:
                    img_gt = img_gt * alpha_map
                if ROI is not None:
                    img_gt = img_gt * ROI

            # Loss
            # ignore loss outside alpha_map and ROI
            loss_rn = criterionL1(outputs, img_gt)
            # loss_rn = list()
            # loss_rn.append(criterionL1(outputs.contiguous().view(-1).float(), img_gt.contiguous().view(-1).float()))
            # loss_rn = torch.stack(loss_rn, dim = 0).mean()

            # total loss for generator
            loss_g = loss_rn
            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()

            for iP in range(len(outputs)):
                outputs[iP] = outputs[iP].cuda()
            outputs = torch.cat(outputs, dim = 0)

            # img_gt = img_gt.cpu()
            # alpha_map = alpha_map.cpu()

            # error metrics
            with torch.no_grad():
                err_metrics_batch_i = metric.compute_err_metrics_batch(outputs * 255.0, img_gt * 255.0, alpha_map, compute_ssim = False)                

            # tensorboard scalar logs of training data
            writer.add_scalar("loss_g", loss_g, iter)
            writer.add_scalar("loss_rn", loss_rn, iter)
            writer.add_scalar("final_mae_valid", err_metrics_batch_i['mae_valid_mean'], iter)
            writer.add_scalar("final_psnr_valid", err_metrics_batch_i['psnr_valid_mean'], iter)

            end = time.time()
            log_time = datetime.datetime.now().strftime('%m/%d') + '_' + datetime.datetime.now().strftime('%H:%M:%S') 
            print("%s Iter %07d   Epoch %03d   loss_g %0.4f   mae_valid %0.4f   psnr_valid %0.4f   t_total %0.4f" 
                  % (log_time,
                     iter,
                     epoch,
                     loss_g, 
                     err_metrics_batch_i['mae_valid_mean'], 
                     err_metrics_batch_i['psnr_valid_mean'], 
                     end - start))

            # tensorboard figure logs of training data
            if not iter % cfg.LOG.PRINT_FREQ:
                output_final_vs_gt = []
                output_final_vs_gt.append(outputs.clamp(min = 0., max = 1.))
                output_final_vs_gt.append(img_gt.clamp(min = 0., max = 1.))
                output_final_vs_gt.append(alpha_map)                
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
                # atlas = torch.cat((texture_mapper_module.textures[0].clone().detach().cpu().permute(0,3,1,2)[:, 0:3, :, :],
                #                     orig_tex.clone().detach().cpu().permute(0,3,1,2)),
                #                     dim = 0)
                atlas = model.module.get_atalas()
                writer.add_image("atlas_vis",
                                torchvision.utils.make_grid(atlas,
                                                            nrow = raster_uv_maps[0].shape[0],
                                                            range = (0, 1),
                                                            scale_each = False,
                                                            normalize = False).cpu().detach().numpy()[:, :, :],
                                                            iter)
            iter += 1

            if iter % cfg.LOG.CHECKPOINT_FREQ == 0:
                model.moudle.save_checkpoint(os.path.join(log_dir, 'model_epoch_%d_iter_%s_.pth' % (epoch, iter)))

    model.moudle.save_checkpoint(os.path.join(log_dir, 'model_epoch_%d_iter_%s_.pth' % (epoch, iter)))

if __name__ == '__main__':
    main()