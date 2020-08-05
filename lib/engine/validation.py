    # dataset for validation views
    if cfg.TRAIN.VAL_FREQ > 0:
        view_val_dataset = dataio.ViewDataset(cfg = cfg, 
                                        root_dir = cfg.DATASET.ROOT,
                                        calib_path = cfg.DATASET.CALIB_PATH,
                                        calib_format = cfg.DATASET.CALIB_FORMAT,
                                        sampling_pattern = cfg.TRAIN.SAMPLING_PATTERN_VAL,
                                        precomp_high_dir = cfg.DATASET.PRECOMP_DIR,
                                        precomp_low_dir = cfg.DATASET.PRECOMP_DIR,
                                        )
        num_view_val = len(view_val_dataset)





    # validation
    if cfg.TRAIN.VAL_FREQ > 0:
        print('Start buffering data for validation...')     
        view_val_dataloader = DataLoader(view_val_dataset, batch_size = cfg.TRAIN.BATCH_SIZE, shuffle = False, num_workers = 8)
        view_val_dataset.buffer_all()






            # validation
            if cfg.TRAIN.VAL_FREQ > 0:
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

                            # rendering module
                            outputs = render_module(neural_img, None)
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
                                                                            nrow=outputs[0].shape[0], # 3
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
                            # for i in range(batch_size):
                            #     cv2.imwrite(os.path.join(val_out_dir, str(iter).zfill(8) + '_' + str(
                            #         view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'),
                            #                 outputs[0][i, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :,
                            #                 ::-1] * 255.)
                            #     cv2.imwrite(os.path.join(val_err_dir, str(iter).zfill(8) + '_' + str(
                            #         view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'),
                            #                 (outputs[0] - img_gt[0]).abs().clamp(min=0., max=1.)[i, :].permute(
                            #                     (1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)
                            #     if first_val:
                            #         cv2.imwrite(os.path.join(val_gt_dir,
                            #                                 str(view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'),
                            #                     img_gt[0][i, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :,
                            #                     ::-1] * 255.)

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
