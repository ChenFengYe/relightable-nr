# backuo 200709
# test_map = torch.tensor(img_gt[0][:,0,:,:][:,None,:,:] <= (2.0 * 255)).cpu().to(alpha_map.dtype).permute(0,2,3,1).numpy()
# print(test_map.shape)
# cv2.imwrite('/data/NFS/new_disk/chenxin/relightable-nr/data/realdome_cx/logs/dnr/test.png', test_map[0,:,:,:])
# cv2.imwrite('/data/NFS/new_disk/chenxin/relightable-nr/data/realdome_cx/logs/dnr/test_255.png', test_map[0,:,:,:] * 255.)

import argparse
import os, time, datetime

import torch
from torch import nn
import torchvision
import numpy as np
import cv2

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import _init_paths

from models import network
from models import metric

from dataset import dataio
from dataset import data_util

from config import cfg
from config import update_config

from utils import util
from shutil import copyfile

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
    #print(cfg.GPUS)
    #os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUS
    #device = torch.device('cuda')
    torch.cuda.set_device(cfg.GPUS[0])
    device = torch.device('cuda:'+ str(cfg.GPUS[0]))

    print("Build dataloader ...")
    # load texture
    if cfg.DATASET.TEX_PATH:
        texture_init = cv2.cvtColor(cv2.imread(cfg.DATASET.TEX_PATH), cv2.COLOR_BGR2RGB)
        texture_init_resize = cv2.resize(texture_init, (cfg.MODEL.TEX_MAPPER.NUM_SIZE, cfg.MODEL.TEX_MAPPER.NUM_SIZE), interpolation = cv2.INTER_AREA).astype(np.float32) / 255.0
        texture_init_use = torch.from_numpy(texture_init_resize).to(device)
    # dataset for training views
    view_dataset = dataio.ViewDataset(cfg = cfg, 
                                    root_dir = cfg.DATASET.ROOT,
                                    calib_path = cfg.DATASET.CALIB_PATH,
                                    calib_format = cfg.DATASET.CALIB_FORMAT,
                                    sampling_pattern = cfg.TRAIN.SAMPLING_PATTERN,
                                    precomp_high_dir = cfg.DATASET.PRECOMP_DIR,
                                    precomp_low_dir = cfg.DATASET.PRECOMP_DIR,
                                    preset_uv_path = cfg.DATASET.UV_PATH,
                                    )
    # dataset for validation views
    view_val_dataset = dataio.ViewDataset(cfg = cfg, 
                                    root_dir = cfg.DATASET.ROOT,
                                    calib_path = cfg.DATASET.CALIB_PATH,
                                    calib_format = cfg.DATASET.CALIB_FORMAT,
                                    sampling_pattern = cfg.TRAIN.SAMPLING_PATTERN_VAL,
                                    precomp_high_dir = cfg.DATASET.PRECOMP_DIR,
                                    precomp_low_dir = cfg.DATASET.PRECOMP_DIR,
                                    )
    num_view_val = len(view_val_dataset)

    print('Build Network...')
    # Rasterizer
    cur_obj_path = ''
    if not cfg.DATASET.LOAD_PRECOMPUTE:
        view_data = view_dataset.read_view(0)
        cur_obj_path = view_data['obj_path']
        frame_idx = view_data['f_idx']
        obj_data = view_dataset.objs[frame_idx]
        rasterizer = network.Rasterizer(cfg,
                            obj_fp = cur_obj_path, 
                            img_size = cfg.DATASET.OUTPUT_SIZE[0],
                            camera_mode = cfg.DATASET.CAM_MODE,
                            obj_data = obj_data,
                            # preset_uv_path = cfg.DATASET.UV_PATH,
                            global_RT = view_dataset.global_RT)
    # texture mapper
    texture_mapper = network.TextureMapper(texture_size = cfg.MODEL.TEX_MAPPER.NUM_SIZE,
                                            texture_num_ch = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                            mipmap_level = cfg.MODEL.TEX_MAPPER.MIPMAP_LEVEL,
                                            apply_sh = cfg.MODEL.TEX_MAPPER.SH_BASIS)
    # render net
    render_net = network.RenderingNet(nf0 = cfg.MODEL.RENDER_NET.NF0,
                                in_channels = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                out_channels = 3,
                                num_down_unet = 5,
                                use_gcn = False)
    # interpolater
    interpolater = network.Interpolater()

    # L1 loss
    criterionL1 = nn.L1Loss(reduction='mean').to(device)
    # Optimizer
    optimizerG = torch.optim.Adam(list(texture_mapper.parameters()) + list(render_net.parameters()), lr = cfg.TRAIN.LR)

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
        print(' Checkpoint_path : %s'%(checkpoint_path))
        util.custom_load([texture_mapper, render_net], ['texture_mapper', 'render_net'], checkpoint_path)
    else:
        print(' Not load params. ')

    texture_mapper.to(device)
    render_net.to(device)
    interpolater.to(device)
    rasterizer.to(device)

    texture_mapper_module = texture_mapper
    render_net_module = render_net
    # use multi-GPU
    if len(cfg.GPUS) > 1:
        texture_mapper = nn.DataParallel(texture_mapper, device_ids = cfg.GPUS)
        render_net = nn.DataParallel(render_net, device_ids = cfg.GPUS)
        interpolater = nn.DataParallel(interpolater, device_ids = cfg.GPUS)
        rasterizer = nn.DataParallel(rasterizer, device_ids = cfg.GPUS)
        rasterizer = rasterizer.module

    # set to training mode
    texture_mapper.train()
    render_net.train()
    interpolater.train()
    rasterizer.eval()      # not train now

    part_list = [texture_mapper_module, render_net_module]     # collect all networks
    part_name_list = ['texture_mapper', 'render_net']
    print("*" * 100)
    print("Number of generator parameters:")
    cfg.defrost()
    cfg.MODEL.TEX_MAPPER.NUM_PARAMS = util.print_network(texture_mapper).item()
    cfg.MODEL.RENDER_NET.NUM_PARAMS = util.print_network(render_net).item()
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
    copyfile(args.cfg, os.path.join(log_dir, cfg.LOG.CFG_NAME))

    print('Start buffering data for training and validation...')
    view_dataloader = DataLoader(view_dataset, batch_size = cfg.TRAIN.BATCH_SIZE, shuffle = cfg.TRAIN.SHUFFLE, num_workers = 8)
    view_val_dataloader = DataLoader(view_val_dataset, batch_size = cfg.TRAIN.BATCH_SIZE, shuffle = False, num_workers = 8)
    #view_dataset.buffer_all()
    #view_val_dataset.buffer_all()

    # Save all command line arguments into a txt file in the logging directory for later referene.
    writer = SummaryWriter(log_dir)
    # iter = cfg.TRAIN.BEGIN_EPOCH * len(view_dataset) # pre model is batch-1

    print('Begin training...')
    # init value
    val_log_batch_id = 0
    first_val = True
    img_h, img_w = cfg.DATASET.OUTPUT_SIZE
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        for view_trgt in view_dataloader:
            start = time.time()

            # get image 
            img_gt = [] 
            img_gt.append(view_trgt[0]['img_gt'].to(device))
            ROI = view_trgt[0]['ROI'].to(device)
            # get uvmap alpha
            uv_map = []            
            alpha_map = []
            if not cfg.DATASET.LOAD_PRECOMPUTE:                
                # raster module
                frame_idxs = view_trgt[0]['f_idx'].numpy()
                for batch_idx, frame_idx in enumerate(frame_idxs):
                    obj_path = view_trgt[0]['obj_path'][batch_idx]
                    if cur_obj_path != obj_path:
                        cur_obj_path = obj_path
                        obj_data = view_dataset.objs[frame_idx]
                        rasterizer.update_vs(obj_data['v_attr'])
                    proj = view_trgt[0]['proj'].to(device)[batch_idx, ...]
                    pose = view_trgt[0]['pose'].to(device)[batch_idx, ...]
                    dist_coeffs = view_trgt[0]['dist_coeffs'].to(device)[batch_idx, ...]
                    uv_map_single, alpha_map_single, _, _, _, _, _, _, _, _, _, _, _, _ = \
                        rasterizer(proj = proj[None, ...], 
                                    pose = pose[None, ...], 
                                    dist_coeffs = dist_coeffs[None, ...], 
                                    offset = None,
                                    scale = None,
                                    )                
                    uv_map.append(uv_map_single[0, ...].clone().detach())
                    alpha_map.append(alpha_map_single[0, ...].clone().detach())
                # fix alpha map
                uv_map = torch.stack(uv_map, dim = 0)
                alpha_map = torch.stack(alpha_map, dim = 0)[:, None, : , :]
                # alpha_map = alpha_map * torch.tensor(img_gt[0][:,0,:,:][:,None,:,:] <= (2.0 * 255)).permute(0,2,1,3).to(alpha_map.dtype).to(alpha_map.device)
                
                # check per iter image
                for batch_idx, frame_idx in enumerate(frame_idxs):
                    if cfg.DEBUG.SAVE_TRANSFORMED_IMG:
                        save_dir_img_gt = './Debug/image_mask'
                        save_path_img_gt = os.path.join(save_dir_img_gt, '%06d_%03d.png'%(iter, frame_idx))
                        cv2.imwrite(save_path_img_gt,  cv2.cvtColor(img_gt[0][batch_idx, ...].cpu().detach().numpy().transpose(1,2,0)*255.0, cv2.COLOR_RGB2BGR))
                        #cv2.imwrite(os.path.join(save_dir_img_gt, '%03d_'%frame_idx + img_fn), cv2.cvtColor(img_gt*255.0, cv2.COLOR_BGR2RGB))
                        print(' Save img: '+ save_path_img_gt)
                        
                    if cfg.DEBUG.SAVE_TRANSFORMED_MASK:
                        save_alpha_map = alpha_map.permute(0,2,3,1).cpu().detach().numpy()
                        save_dir_mask = './Debug/image_mask'
                        save_path_mask = os.path.join(save_dir_mask, '%06d_%03d_mask.png'%(iter, frame_idx))
                        cv2.imwrite(save_path_mask, save_alpha_map[batch_idx, ...]*255.0)
                        print(' Save mask: '+ save_path_mask)

            else:            
                # get view data
                uv_map = view_trgt[0]['uv_map'].to(device) # [N, H, W, 2]
                # sh_basis_map = view_trgt[0]['sh_basis_map'].to(device) # [N, H, W, 9]
                alpha_map = view_trgt[0]['alpha_map'][:, None, :, :].to(device) # [N, 1, H, W]

            # sample texture
            # neural_img = texture_mapper(uv_map, sh_basis_map)
            neural_img = texture_mapper(uv_map)

            # rendering net
            outputs = render_net(neural_img, None)
            img_max_val = 2.0
            outputs = (outputs * 0.5 + 0.5) * img_max_val # map to [0, img_max_val]
            if type(outputs) is not list:
                outputs = [outputs]

            # # We don't enforce a loss on the outermost 5 pixels to alleviate boundary errors, also weight loss by alpha
            # alpha_map_central = alpha_map[:, :, 5:-5, 5:-5]
            # for i in range(len(view_trgt)):
            #     outputs[i] = outputs[i][:, :, 5:-5, 5:-5] * alpha_map_central
            #     img_gt[i] = img_gt[i][:, :, 5:-5, 5:-5] * alpha_map_central

            # ignore loss outside ROI
            for i in range(len(view_trgt)):
                outputs[i] = outputs[i] * ROI * alpha_map
                img_gt[i] = img_gt[i]* ROI * alpha_map

            # loss on final image
            loss_rn = list()
            for idx in range(len(view_trgt)):
                loss_rn.append(criterionL1(outputs[idx].contiguous().view(-1).float(), img_gt[idx].contiguous().view(-1).float()))
            loss_rn = torch.stack(loss_rn, dim = 0).mean()

            # total loss for generator
            loss_g = loss_rn

            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()

            # error metrics
            with torch.no_grad():
                err_metrics_batch_i = metric.compute_err_metrics_batch(outputs[0] * 255.0, img_gt[0] * 255.0, alpha_map, compute_ssim = False)
                # err_metrics_batch_i = metric.compute_err_metrics_batch(outputs[0] * 255.0, img_gt[0] * 255.0, alpha_map_central, compute_ssim = False)

            # tensorboard scalar logs of training data
            writer.add_scalar("loss_g", loss_g, iter)
            writer.add_scalar("loss_rn", loss_rn, iter)
            writer.add_scalar("final_mae_valid", err_metrics_batch_i['mae_valid_mean'], iter)
            writer.add_scalar("final_psnr_valid", err_metrics_batch_i['psnr_valid_mean'], iter)

            end = time.time()
            print("Iter %07d   Epoch %03d   loss_g %0.4f   mae_valid %0.4f   psnr_valid %0.4f   t_total %0.4f" % (iter, epoch, loss_g, err_metrics_batch_i['mae_valid_mean'], err_metrics_batch_i['psnr_valid_mean'], end - start))

            # tensorboard figure logs of training data
            if not iter % cfg.LOG.PRINT_FREQ:
                output_final_vs_gt = []
                for i in range(len(view_trgt)):
                    output_final_vs_gt.append(outputs[i].clamp(min = 0., max = 1.))
                    output_final_vs_gt.append(img_gt[i].clamp(min = 0., max = 1.))
                    output_final_vs_gt.append((outputs[i] - img_gt[i]).abs().clamp(min = 0., max = 1.))

                output_final_vs_gt = torch.cat(output_final_vs_gt, dim = 0)
                raster_uv_maps = torch.cat((uv_map.permute(0,3,1,2),  # N H W 2 -> N 2 H W
                                    torch.zeros(uv_map.shape[0], 1, img_h, img_w, dtype=uv_map.dtype, device=uv_map.device)),
                                    dim = 1)
                writer.add_image("raster_uv_vis",
                                torchvision.utils.make_grid(raster_uv_maps,
                                                            nrow = raster_uv_maps[0].shape[0],
                                                            range = (0, 1),
                                                            scale_each = False,
                                                            normalize = False).cpu().detach().numpy()[::-1, :, :], # uv0 -> 0vu (rgb)
                                                            iter)
                writer.add_image("output_final_vs_gt",
                                torchvision.utils.make_grid(output_final_vs_gt,
                                                            nrow = 3, #outputs[0].shape[0],
                                                            range = (0, 1),
                                                            scale_each = False,
                                                            normalize = False).cpu().detach().numpy(),
                                                            iter)

            # validation
            if not iter % cfg.TRAIN.VAL_FREQ:
                start_val = time.time()
                with torch.no_grad():
                    # error metrics
                    err_metrics_val = {}
                    err_metrics_val['mae_valid'] = []
                    err_metrics_val['mse_valid'] = []
                    err_metrics_val['psnr_valid'] = []
                    err_metrics_val['ssim_valid'] = []
                    # loop over batches
                    batch_id = 0
                    for view_val_trgt in view_val_dataloader:
                        start_val_i = time.time()

                        # get image
                        img_gt = []
                        img_gt.append(view_val_trgt[0]['img_gt'].to(device))
                        ROI = view_val_trgt[0]['ROI'].to(device)
                        # get uvmap alpha
                        uv_map = []            
                        alpha_map = []
                        if not cfg.DATASET.LOAD_PRECOMPUTE:
                            # build raster module
                            frame_idxs = view_val_trgt[0]['f_idx'].numpy()
                            for batch_idx, frame_idx in enumerate(frame_idxs):
                                obj_path = view_val_trgt[0]['obj_path'][batch_idx]
                                if cur_obj_path != obj_path:
                                    cur_obj_path = obj_path
                                    obj_data = view_val_dataset.objs[frame_idx]
                                    rasterizer.update_vs(obj_data['v_attr'])
                                proj = view_val_trgt[0]['proj'].to(device)[batch_idx, ...]
                                pose = view_val_trgt[0]['pose'].to(device)[batch_idx, ...]
                                dist_coeffs = view_val_trgt[0]['dist_coeffs'].to(device)[batch_idx, ...]
                                uv_map_single, alpha_map_single, _, _, _, _, _, _, _, _, _, _, _, _ = \
                                    rasterizer(proj = proj[None, ...], 
                                                pose = pose[None, ...], 
                                                dist_coeffs = dist_coeffs[None, ...], 
                                                offset = None,
                                                scale = None,
                                                )                
                                uv_map.append(uv_map_single[0, ...].clone().detach())
                                alpha_map.append(alpha_map_single[0, ...].clone().detach())
                            # fix alpha map
                            uv_map = torch.stack(uv_map, dim = 0)
                            alpha_map = torch.stack(alpha_map, dim = 0)[:, None, : , :]
                            # alpha_map = alpha_map * torch.tensor(img_gt[0][:,0,:,:][:,None,:,:] <= (2.0 * 255)).permute(0,2,1,3).to(alpha_map.dtype).to(alpha_map.device)
                        else:                 
                            uv_map = view_val_trgt[0]['uv_map'].to(device)  # [N, H, W, 2]
                            # sh_basis_map = view_val_trgt[0]['sh_basis_map'].to(device)  # [N, H, W, 9]
                            alpha_map = view_val_trgt[0]['alpha_map'][:, None, :, :].to(device)  # [N, 1, H, W]
                            
                        view_idx = view_val_trgt[0]['idx']
                        num_view = len(view_val_trgt)
                        img_gt = []
                        for i in range(num_view):
                            img_gt.append(view_val_trgt[i]['img_gt'].to(device))

                        # sample texture
                        # neural_img = texture_mapper(uv_map, sh_basis_map)
                        neural_img = texture_mapper(uv_map)

                        # rendering net
                        outputs = render_net(neural_img, None)
                        img_max_val = 2.0
                        outputs = (outputs * 0.5 + 0.5) * img_max_val  # map to [0, img_max_val]
                        if type(outputs) is not list:
                            outputs = [outputs]

                        # apply alpha and ROI
                        for i in range(num_view):
                            outputs[i] = outputs[i] * alpha_map * ROI
                            img_gt[i] = img_gt[i] * alpha_map * ROI

                        # tensorboard figure logs of validation data
                        if batch_id == val_log_batch_id:
                            output_final_vs_gt = []
                            for i in range(num_view):
                                output_final_vs_gt.append(outputs[i].clamp(min=0., max=1.))
                                output_final_vs_gt.append(img_gt[i].clamp(min=0., max=1.))
                                output_final_vs_gt.append(
                                    (outputs[i] - img_gt[i]).abs().clamp(min=0., max=1.))
                                    
                            output_final_vs_gt = torch.cat(output_final_vs_gt, dim=0)
                            writer.add_image("output_final_vs_gt_val",
                                             torchvision.utils.make_grid(output_final_vs_gt,
                                                                         nrow=3, # outputs[0].shape[0]
                                                                         range=(0, 1),
                                                                         scale_each=False,
                                                                         normalize=False).cpu().detach().numpy(),
                                                                         iter)

                        # error metrics
                        err_metrics_batch_i_final = metric.compute_err_metrics_batch(outputs[0] * 255.0,
                                                                                     img_gt[0] * 255.0, alpha_map,
                                                                                     compute_ssim=True)
                        batch_size = view_idx.shape[0]
                        for i in range(batch_size):
                            for key in list(err_metrics_val.keys()):
                                if key in err_metrics_batch_i_final.keys():
                                    err_metrics_val[key].append(err_metrics_batch_i_final[key][i])

                        # save images
                        for i in range(batch_size):
                            cv2.imwrite(os.path.join(val_out_dir, str(iter).zfill(8) + '_' + str(
                                view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'),
                                        outputs[0][i, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :,
                                        ::-1] * 255.)
                            cv2.imwrite(os.path.join(val_err_dir, str(iter).zfill(8) + '_' + str(
                                view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'),
                                        (outputs[0] - img_gt[0]).abs().clamp(min=0., max=1.)[i, :].permute(
                                            (1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)
                            if first_val:
                                cv2.imwrite(os.path.join(val_gt_dir,
                                                         str(view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'),
                                            img_gt[0][i, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :,
                                            ::-1] * 255.)

                        end_val_i = time.time()
                        print("Val   batch %03d   mae_valid %0.4f   psnr_valid %0.4f   ssim_valid %0.4f   t_total %0.4f" % (
                            batch_id, err_metrics_batch_i_final['mae_valid_mean'],
                            err_metrics_batch_i_final['psnr_valid_mean'],
                            err_metrics_batch_i_final['ssim_valid_mean'], end_val_i - start_val_i))

                        batch_id += 1

                    for key in list(err_metrics_val.keys()):
                        if err_metrics_val[key]:
                            err_metrics_val[key] = np.vstack(err_metrics_val[key])
                            err_metrics_val[key + '_mean'] = err_metrics_val[key].mean()
                        else:
                            err_metrics_val[key + '_mean'] = np.nan

                    # tensorboard scalar logs of validation data
                    writer.add_scalar("final_mae_valid_val", err_metrics_val['mae_valid_mean'], iter)
                    writer.add_scalar("final_psnr_valid_val", err_metrics_val['psnr_valid_mean'], iter)
                    writer.add_scalar("final_ssim_valid_val", err_metrics_val['ssim_valid_mean'], iter)

                    first_val = False
                    val_log_batch_id = (val_log_batch_id + 1) % batch_id

                    end_val = time.time()
                    print("Val   mae_valid %0.4f   psnr_valid %0.4f   ssim_valid %0.4f   t_total %0.4f" % (
                    err_metrics_val['mae_valid_mean'], err_metrics_val['psnr_valid_mean'],
                    err_metrics_val['ssim_valid_mean'], end_val - start_val))

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