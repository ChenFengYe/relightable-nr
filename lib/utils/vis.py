import torch
import torchvision

def writer_add_scalar(writer, iter, epoch, metrics, loss=None, log_time=0, iter_time=0, ex_name=None):
    # tensorboard scalar logs of training data
    ex_name = ex_name + '_' if ex_name else ''
    if loss:
        writer.add_scalar(ex_name+"loss_g", loss['Loss'], iter)
        writer.add_scalar(ex_name+"loss_rgb", loss['rgb'], iter)
        writer.add_scalar(ex_name+"loss_hsv", loss['hsv'], iter)            
        writer.add_scalar(ex_name+"loss_atlas", loss['atlas'], iter)
    writer.add_scalar(ex_name+"final_mae_valid", metrics['mae_valid_mean'], iter)
    writer.add_scalar(ex_name+"final_psnr_valid", metrics['psnr_valid_mean'], iter)

    print("%s %s Iter-%07d Epoch-%03d loss_g/rgb/hsv/tex %0.6f/%0.6f/%0.6f/%0.6f mae_valid %0.4f psnr_valid %0.4f t %0.2fs" 
            % (log_time,
                ex_name,
                iter,
                epoch,
                loss['Loss'], 
                loss['rgb'], 
                loss['hsv'], 
                loss['atlas'], 
                metrics['mae_valid_mean'], 
                metrics['psnr_valid_mean'], 
                iter_time))
    
def writer_add_scalar_gan(writer, num_iter, epoch, metrics, loss=None, log_time=0, iter_time=0, ex_name=None):
    # tensorboard scalar logs of training data
    ex_name = ex_name + '_' if ex_name else ''
    if loss:
        for key, val in loss.items():
            writer.add_scalar(ex_name+key, val, num_iter)
    writer.add_scalar(ex_name+"final_mae_valid", metrics['mae_valid_mean'], num_iter)
    writer.add_scalar(ex_name+"final_psnr_valid", metrics['psnr_valid_mean'], num_iter)

    it_v = iter(loss.values())  
    it_k = iter(loss.keys())  
    print("%s %s Iter-%07d Epoch-%03d \n%8s/%8s/%8s/%8s/%8s/%8s/%8s \n%0.6f/%0.6f/%0.6f/%0.6f/%0.6f/%0.6f/%0.6f mae_valid %0.4f psnr_valid %0.4f t %0.2fs" 
            % (log_time,
                ex_name,
                num_iter,
                epoch,
                next(it_k), next(it_k), next(it_k), next(it_k), next(it_k), next(it_k), next(it_k), 
                next(it_v), next(it_v), next(it_v), next(it_v), next(it_v), next(it_v), next(it_v), 
                metrics['mae_valid_mean'], 
                metrics['psnr_valid_mean'], 
                iter_time))

def writer_add_image(writer, iter, epoch, img_gt, outputs_img, neural_img, uv_map, aligned_uv, atlas=None, img_ref=None, ex_name=None):
    ex_name = ex_name + '_' if ex_name else ''
    uv_map = uv_map.permute(0,3,1,2)
    N, C, H, W = img_gt.shape

    img_vis = []
    if img_ref is not None:
        img_vis.append(img_ref.clamp(min = 0., max = 1.))
    img_vis.append(outputs_img.clamp(min = 0., max = 1.))
    img_vis.append(img_gt.clamp(min = 0., max = 1.))
    img_vis.append(neural_img)
    img_vis.append((outputs_img - img_gt).abs().clamp(min = 0., max = 1.))
    img_vis.append((neural_img - outputs_img).abs().clamp(min = 0., max = 1.))
    img_vis = torch.cat(img_vis, dim = 0)

    writer.add_image(ex_name+"img_vis",
                    torchvision.utils.make_grid(img_vis,
                                                nrow = outputs_img.shape[0],
                                                range = (0, 1),
                                                scale_each = False,
                                                normalize = False).cpu().detach().numpy(),
                                                iter)
    uv_map3 = torch.cat((uv_map, torch.zeros(N, 1, H, W, dtype=uv_map.dtype, device=uv_map.device)), dim = 1)

    if aligned_uv is not None:
        aligned_uv3 = torch.cat((aligned_uv, torch.zeros(N, 1, H, W, dtype=uv_map.dtype, device=uv_map.device)), dim = 1)
        uv_diff3 = (uv_map3 - aligned_uv3).abs().clamp(min = 0., max = 1.)
        raster_uv_maps = torch.cat((uv_map3, aligned_uv3, uv_diff3), dim = 0)
    else:
        raster_uv_maps = uv_map3
    writer.add_image(ex_name+"raster_uv_vis",
                    torchvision.utils.make_grid(raster_uv_maps,
                                                nrow = outputs_img.shape[0],
                                                range = (0, 1),
                                                scale_each = False,
                                                normalize = False).cpu().detach().numpy()[::-1, :, :], # uv0 -> 0vu (rgb)
                                                iter)
    if atlas is not None:
        writer.add_image(ex_name+"atlas_vis",
                        torchvision.utils.make_grid(atlas.clamp(min = 0., max = 1.),
                                                    nrow = outputs_img.shape[0],
                                                    range = (0, 1),
                                                    scale_each = False,
                                                    normalize = False).cpu().detach().numpy()[:, :, :],
                                                    iter)

def writer_add_image_gan(writer, iter, epoch, inputs, results, ex_name=None):
    ex_name = ex_name + '_' if ex_name else ''
    ############################################################################
    # vis img
    img_vis = []
    vis_set = ['img_ref','img_rs','img','nimg_rs']
    for vis_key in vis_set:
        if vis_key in results:
            img_vis.append(results[vis_key].clamp(min = 0., max = 1.))
        if vis_key in inputs:
            img_vis.append(inputs[vis_key].clamp(min = 0., max = 1.))
    # different
    vis_set_diff = ['img_rs','nimg_rs']
    for vis_key in vis_set_diff:
        if vis_key in inputs:
            img_vis.append((results[vis_key] - inputs['img']).abs().clamp(min = 0., max = 1.))
    img_vis = torch.cat(img_vis, dim = 0)

    writer.add_image(ex_name+"output_final_vs_gt",
                    torchvision.utils.make_grid(img_vis,
                                                nrow = results['img_rs'].shape[0],
                                                range = (0, 1),
                                                scale_each = False,
                                                normalize = False).cpu().detach().numpy(),
                                                iter)
    ############################################################################
    # vis atlas
    atlas = []
    vis_set_tex = ['tex_ref','tex_rs','tex_tar']
    for vis_key in vis_set_tex:
        if vis_key in results:
            atlas.append(results[vis_key].clamp(min = 0., max = 1.))
        if vis_key in inputs:
            atlas.append(inputs[vis_key].clamp(min = 0., max = 1.))
    # different
    # vis_set_diff = ['tex_rs','tex_ref']
    # for vis_key in vis_set_diff:
    #     if vis_key in inputs:
    #         atlas.append((results[vis_key] - inputs['tex_tar']).abs().clamp(min = 0., max = 1.))
    atlas = torch.cat(atlas, dim = 0)
    if atlas is not None:
        writer.add_image(ex_name+"atlas_vis",
                        torchvision.utils.make_grid(atlas.clamp(min = 0., max = 1.),
                                                    nrow = results['img_rs'].shape[0],
                                                    range = (0, 1),
                                                    scale_each = False,
                                                    normalize = False).cpu().detach().numpy(),
                                                    iter)
    ############################################################################
    # vis uv
    uv_maps = []
    vis_set_uv = ['uv_map','uv_map_ref']
    for vis_key in vis_set_uv:
        if vis_key in results:
            uv_map = results[vis_key]
        elif vis_key in inputs:
            uv_map = inputs[vis_key]

        uv_map = uv_map.permute(0,3,1,2)
        N, C, H, W = uv_map.shape
        uv_map3 = torch.cat((uv_map, torch.zeros(N, 1, H, W, dtype=uv_map.dtype, device=uv_map.device)), dim = 1)
        uv_maps.append(uv_map3.clamp(min = 0., max = 1.))

    # difference
    vis_set_diff = ['uv_map_align']
    for vis_key in vis_set_diff:
        if vis_key in results:
            uv_map = results[vis_key]

            uv_map = uv_map.permute(0,3,1,2)
            N, C, H, W = uv_map.shape
            uv_map_rs3 = torch.cat((uv_map, torch.zeros(N, 1, H, W, dtype=uv_map.dtype, device=uv_map.device)), dim = 1)
            uv_map_gt3 = torch.cat((inputs['uv_map'].permute(0,3,1,2), torch.zeros(N, 1, H, W, dtype=uv_map.dtype, device=uv_map.device)), dim = 1)           
            uv_maps.append((uv_map_rs3 - uv_map_gt3).abs().clamp(min = 0., max = 1.))

    uv_maps = torch.cat(uv_maps, dim = 0)
    writer.add_image(ex_name+"raster_uv_vis",
                torchvision.utils.make_grid(uv_maps,
                                            nrow = results['img_rs'].shape[0],
                                            range = (0, 1),
                                            scale_each = False,
                                            normalize = False).cpu().detach().numpy()[::-1, :, :], # uv0 -> 0vu (rgb)
                                            iter)
