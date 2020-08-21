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
    print("%s %s Iter-%07d Epoch-%03d \n%8s/%8s/%8s/%8s/%8s/%8s \n%0.6f/%0.6f/%0.6f/%0.6f/%0.6f/%0.6f mae_valid %0.4f psnr_valid %0.4f t %0.2fs" 
            % (log_time,
                ex_name,
                num_iter,
                epoch,
                next(it_k), next(it_k), next(it_k), next(it_k), next(it_k), next(it_k), 
                next(it_v), next(it_v), next(it_v), next(it_v), next(it_v), next(it_v), 
                metrics['mae_valid_mean'], 
                metrics['psnr_valid_mean'], 
                iter_time))

def writer_add_image(writer, iter, epoch, img_gt, outputs_img, neural_img, uv_map, aligned_uv, atlas=None, img_ref=None, ex_name=None):
    ex_name = ex_name + '_' if ex_name else ''
    uv_map = uv_map.permute(0,3,1,2)
    N, C, H, W = img_gt.shape

    output_final_vs_gt = []
    if img_ref is not None:
        output_final_vs_gt.append(img_ref.clamp(min = 0., max = 1.))
    output_final_vs_gt.append(outputs_img.clamp(min = 0., max = 1.))
    output_final_vs_gt.append(img_gt.clamp(min = 0., max = 1.))
    output_final_vs_gt.append(neural_img)
    output_final_vs_gt.append((outputs_img - img_gt).abs().clamp(min = 0., max = 1.))
    output_final_vs_gt.append((neural_img - outputs_img).abs().clamp(min = 0., max = 1.))
    output_final_vs_gt = torch.cat(output_final_vs_gt, dim = 0)

    writer.add_image(ex_name+"output_final_vs_gt",
                    torchvision.utils.make_grid(output_final_vs_gt,
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