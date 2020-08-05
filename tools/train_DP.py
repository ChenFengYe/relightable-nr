import argparse
import os, time, datetime

import torch
from torch import nn
import torchvision
# import numpy as np
# import cv2

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import _init_paths

from lib.models import network
from lib.models import metric

from lib.dataset import DPViewDataset
from lib.dataset import data_util

from lib.config import cfg
from lib.config import update_config

from lib.utils import util


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

    # cfg.defrost()
    # cfg.RANK = args.ranka
    # cfg.freeze()                                    
    # device allocation

    print('Set device...')
    # print(cfg.GPUS)
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUS
    # device = torch.device('cuda')
    torch.cuda.set_device(cfg.GPUS[0])
    device = torch.device('cuda:' + str(cfg.GPUS[0]))

    print("Build dataloader ...")
    # dataset for training views
    view_dataset = zhen_dataset_rotate.ViewDatasetZhen(cfg=cfg)
    # dataset for validation views
    view_val_dataset = zhen_dataset_rotate.ViewDatasetZhen(cfg=cfg)
    # num_view_val = len(view_val_dataset)

    print('Build Network...')
    # texture mapper
    texture_mapper = network.TextureMapper(texture_size=cfg.MODEL.TEX_MAPPER.NUM_SIZE,
                                           texture_num_ch=cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                           mipmap_level=cfg.MODEL.TEX_MAPPER.MIPMAP_LEVEL,
                                           apply_sh=cfg.MODEL.TEX_MAPPER.SH_BASIS)
    # render net
    render_module = network.RenderingModule(nf0=cfg.MODEL.RENDER_MODULE.NF0,
                                      in_channels=cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                      out_channels=3,
                                      num_down_unet=5,
                                      use_gcn=False)

    texture_mapper.to(device)
    render_module.to(device)

    # L1 loss
    criterionL1 = nn.L1Loss(reduction='mean').to(device)
    # Optimizer
    optimizerG = torch.optim.Adam(list(texture_mapper.parameters()) + list(render_module.parameters()), lr=cfg.TRAIN.LR)

    print('Loading Model...')
    iter = 0
    dir_name = os.path.join(datetime.datetime.now().strftime('%m-%d') +
                            '_' + datetime.datetime.now().strftime('%H-%M-%S') +
                            '_' + cfg.TRAIN.SAMPLING_PATTERN +
                            '_' + cfg.DATASET.ROOT.strip('/').split('/')[-1])
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
    if checkpoint_path:
        print(' Checkpoint_path : %s' % (checkpoint_path))
        util.custom_load([texture_mapper, render_module], ['texture_mapper', 'render_module'], checkpoint_path)
    else:
        print(' Not load params. ')

    texture_mapper_module = texture_mapper
    render_module = render_module

    # use multi-GPU
    if len(cfg.GPUS) > 1:
        texture_mapper = nn.DataParallel(texture_mapper, device_ids=cfg.GPUS)
        render_module = nn.DataParallel(render_module, device_ids=cfg.GPUS)
        # texture_mapper = nn.DataParallel(texture_mapper)
        # render_module = nn.DataParallel(render_module)

    # set to training mode
    texture_mapper.train()
    render_module.train()

    part_list = [texture_mapper_module, render_module]  # collect all networks
    part_name_list = ['texture_mapper', 'render_module']
    print("*" * 100)
    print("Number of generator parameters:")
    cfg.defrost()
    cfg.MODEL.TEX_MAPPER.NUM_PARAMS = util.print_network(texture_mapper).item()
    cfg.MODEL.RENDER_MODULE.NUM_PARAMS = util.print_network(render_module).item()
    cfg.freeze()
    print("*" * 100)

    print("Setup Log ...")
    log_dir = os.path.join(cfg.LOG.LOGGING_ROOT, dir_name)
    data_util.cond_mkdir(log_dir)
    val_out_dir = os.path.join(log_dir, 'val_out')
    val_gt_dir = os.path.join(log_dir, 'val_gt')
    val_err_dir = os.path.join(log_dir, 'val_err')
    data_util.cond_mkdir(val_out_dir)
    data_util.cond_mkdir(val_gt_dir)
    data_util.cond_mkdir(val_err_dir)
    util.custom_copy(args.cfg, os.path.join(log_dir, cfg.LOG.CFG_NAME))

    print('Start buffering data for training and validation...')
    view_dataloader = DataLoader(view_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.SHUFFLE,
                                 num_workers=8)
    view_val_dataloader = DataLoader(view_val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=8)
    # view_dataset.buffer_all()
    # view_val_dataset.buffer_all()

    # Save all command line arguments into a txt file in the logging directory for later reference.
    writer = SummaryWriter(log_dir)
    # iter = cfg.TRAIN.BEGIN_EPOCH * len(view_dataset) # pre model is batch-1

    print('Begin training...')
    # init value
    img_h, img_w = cfg.DATASET.OUTPUT_SIZE
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        for view_trgt in view_dataloader:
            start = time.time()
            # get image 
            img_gt = view_trgt['img'].to(device)
            # get uvmap alpha
            # print("pre transpose", view_trgt['uvmap'].size())
            uv_map = view_trgt['uvmap'].permute(0, 2, 3, 1).to(device)
            # print("uvmap size", uv_map.size())
            alpha_map = view_trgt['mask'].to(device)

            # uv_map_temp = view_trgt['uv_temp'].to(device)

            # sample texture
            # neural_img = texture_mapper(uv_map, sh_basis_map)
            neural_img = texture_mapper(uv_map)

            # rendering module
            outputs = render_module(neural_img, None)
            # img_max_val = 2.0
            # outputs = (outputs * 0.5 + 0.5) * img_max_val  # map to [0, img_max_val]
            # if type(outputs) is not list:
            #     outputs = [outputs]

            # ignore loss outside ROI
            # for i in range(len(view_trgt)):
            #     outputs[i] = outputs[i] * alpha_map
            #     img_gt[i] = img_gt[i] * alpha_map

            # loss on final image
            # loss_rn = list()
            # for idx in range(len(view_trgt)):
            #     loss_rn.append(
            #         criterionL1(outputs[idx].contiguous().view(-1).float(), img_gt[idx].contiguous().view(-1).float()))
            # loss_rn = torch.stack(loss_rn, dim=0).mean()
            loss_rn = criterionL1(outputs * alpha_map, img_gt * alpha_map)

            # total loss for generator
            loss_g = loss_rn

            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()

            # error metrics
            with torch.no_grad():
                err_metrics_batch_i = metric.compute_err_metrics_batch(outputs * 255.0, img_gt * 255.0, alpha_map,
                                                                       compute_ssim=False)
                # err_metrics_batch_i = metric.compute_err_metrics_batch(outputs[0] * 255.0, img_gt[0] * 255.0, alpha_map_central, compute_ssim = False)

            # tensorboard scalar logs of training data
            writer.add_scalar("loss_g", loss_g, iter)
            writer.add_scalar("loss_rn", loss_rn, iter)
            writer.add_scalar("final_mae_valid", err_metrics_batch_i['mae_valid_mean'], iter)
            writer.add_scalar("final_psnr_valid", err_metrics_batch_i['psnr_valid_mean'], iter)

            end = time.time()
            print("Iter %07d   Epoch %03d   loss_g %0.4f   mae_valid %0.4f   psnr_valid %0.4f   t_total %0.4f" % (
                iter, epoch, loss_g, err_metrics_batch_i['mae_valid_mean'], err_metrics_batch_i['psnr_valid_mean'],
                end - start))

            # tensorboard figure logs of training data
            if not iter % cfg.LOG.PRINT_FREQ:
                output_final_vs_gt = []
                # print(view_trgt['img'].size(0))
                # for i in range(view_trgt['img'].size(0)):
                output_final_vs_gt.append(outputs.clamp(min=0., max=1.))
                output_final_vs_gt.append(img_gt.clamp(min=0., max=1.))
                output_final_vs_gt.append(alpha_map)
                output_final_vs_gt.append((outputs - img_gt).abs().clamp(min=0., max=1.))

                output_final_vs_gt = torch.cat(output_final_vs_gt, dim=0)
                raster_uv_maps = torch.cat((uv_map.permute(0, 3, 1, 2),  # N H W 2 -> N 2 H W
                                            torch.zeros(uv_map.shape[0], 1, img_h, img_w, dtype=uv_map.dtype,
                                                        device=uv_map.device)),
                                           dim=1)
                # raster_uv_maps_temp = torch.cat((uv_map_temp.permute(0, 3, 1, 2),  # N H W 2 -> N 2 H W
                #                                  torch.zeros(uv_map_temp.shape[0], 1, img_h, img_w,
                #                                              dtype=uv_map_temp.dtype,
                #                                              device=uv_map_temp.device)),
                #                                 dim=1)
                # print(output_final_vs_gt.size(), raster_uv_maps.size())
                writer.add_image("raster_uv_vis",
                                 torchvision.utils.make_grid(raster_uv_maps,
                                                             nrow=raster_uv_maps.shape[0],
                                                             range=(0, 1),
                                                             scale_each=False,
                                                             normalize=False).cpu().detach().numpy()[::-1, :, :],
                                 # uv0 -> 0vu (rgb)
                                 iter)
                writer.add_image("output_final_vs_gt",
                                 torchvision.utils.make_grid(output_final_vs_gt,
                                                             nrow=outputs.shape[0],  # 3
                                                             range=(0, 1),
                                                             scale_each=False,
                                                             normalize=False).cpu().detach().numpy(),
                                 iter)
                # writer.add_image("raster_uv_temp_vis",
                #                  torchvision.utils.make_grid(raster_uv_maps_temp,
                #                                              nrow=raster_uv_maps_temp.shape[0],
                #                                              range=(0, 1),
                #                                              scale_each=False,
                #                                              normalize=False).cpu().detach().numpy()[::-1, :, :],
                #                  # uv0 -> 0vu (rgb)
                #                  iter)

            iter += 1

            if iter % cfg.LOG.CHECKPOINT_FREQ == 0:
                util.custom_save(os.path.join(log_dir, 'model_epoch_%d_iter_%s_.pth' % (epoch, iter)),
                                 part_list,
                                 part_name_list)

    util.custom_save(os.path.join(log_dir, 'model_epoch_%d_iter_%s_.pth' % (epoch, iter)),
                     part_list,
                     part_name_list)


if __name__ == '__main__':
    main()
