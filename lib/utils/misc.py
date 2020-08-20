import numpy as np
import torch

def interpolate_bilinear_inv(img, sub_u, sub_v, texture_size):
    '''
    inverse fucntion of interpolation_bilinear
    convert data back to xy domain

    img: [N, C, H, W] 
    uv_map: [N, H, W, C] 
    texture_size: S
    return: [N, S, S, C]
    '''
    device = img.device
    batch_size = img.shape[0]
    channel = img.shape[1]

    # convert uv_map (atlas/texture to img)
    #           to
    #         tex_grid (img to atlas)    
    # N = uv_map.shape[0]
    # tex_grid = torch.ones(N, texture_size, texture_size, 2, dtype=torch.float32).to(device)
    output = torch.zeros(batch_size, channel, texture_size, texture_size).to(device)
    
    coord_nxy = torch.nonzero(sub_u)
    coord_n = coord_nxy[:,0]
    coord_x = coord_nxy[:,1]
    coord_y = coord_nxy[:,2]

    u_cur = torch.floor(sub_u[coord_n, coord_x, coord_y]).long().to(device)
    v_cur = torch.floor(sub_v[coord_n, coord_x, coord_y]).long().to(device)

    u_cur = torch.clamp(u_cur, 0, texture_size - 1)
    v_cur = torch.clamp(v_cur, 0, texture_size - 1)

    output[coord_n, :, v_cur, u_cur] = img[coord_n, :, coord_x, coord_y]

    return output.permute(0,2,3,1)

def interpolate_atlas(tex, window_rate=0.06):
    tex_size = tex.shape[2]
    window_size = window_rate * tex_size
    # to-do 0819

    return tex


def interpolate_bilinear(data, sub_x, sub_y):
    '''
    data: [H, W, C]
    sub_x: [...]
    sub_y: [...]
    return: [..., C]
    '''
    device = data.device

    mask_valid = ((sub_x >= 0) & (sub_x <= data.shape[1] - 1) & (sub_y >= 0) & (sub_y <= data.shape[0] - 1)).to(data.dtype).to(device)

    x0 = torch.floor(sub_x).long().to(device)
    x1 = x0 + 1
    
    y0 = torch.floor(sub_y).long().to(device)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, data.shape[1] - 1)
    x1 = torch.clamp(x1, 0, data.shape[1] - 1)
    y0 = torch.clamp(y0, 0, data.shape[0] - 1)
    y1 = torch.clamp(y1, 0, data.shape[0] - 1)
    
    I00 = data[y0, x0, :] # [..., C]
    I10 = data[y1, x0, :]
    I01 = data[y0, x1, :]
    I11 = data[y1, x1, :]

    # right boundary
    x0 = x0 - (x0 == x1).to(x0.dtype)
    # bottom boundary
    y0 = y0 - (y0 == y1).to(y0.dtype)

    w00 = (x1.to(data.dtype) - sub_x) * (y1.to(data.dtype) - sub_y) * mask_valid # [...]
    w10 = (x1.to(data.dtype) - sub_x) * (sub_y - y0.to(data.dtype)) * mask_valid
    w01 = (sub_x - x0.to(data.dtype)) * (y1.to(data.dtype) - sub_y) * mask_valid
    w11 = (sub_x - x0.to(data.dtype)) * (sub_y - y0.to(data.dtype)) * mask_valid

    return I00 * w00.unsqueeze_(-1) + I10 * w10.unsqueeze_(-1) + I01 * w01.unsqueeze_(-1) + I11 * w11.unsqueeze_(-1)


def interpolate_bilinear_np(data, sub_x, sub_y):
    '''
    data: [H, W, C]
    sub_x: [...]
    sub_y: [...]
    return: [..., C]
    '''
    x0 = np.floor(sub_x).astype(np.int64)
    x1 = x0 + 1
    
    y0 = np.floor(sub_y).astype(np.int64)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, data.shape[1] - 1)
    x1 = np.clip(x1, 0, data.shape[1] - 1)
    y0 = np.clip(y0, 0, data.shape[0] - 1)
    y1 = np.clip(y1, 0, data.shape[0] - 1)
    
    I00 = data[y0, x0, :] # [..., C]
    I10 = data[y1, x0, :]
    I01 = data[y0, x1, :]
    I11 = data[y1, x1, :]

    w00 = (x1 - sub_x) * (y1 - sub_y) # [...]
    w10 = (x1 - sub_x) * (sub_y - y0)
    w01 = (sub_x - x0) * (y1 - sub_y)
    w11 = (sub_x - x0) * (sub_y - y0)

    return I00 * w00[..., None] + I10 * w10[..., None] + I01 * w01[..., None] + I11 * w11[..., None]
