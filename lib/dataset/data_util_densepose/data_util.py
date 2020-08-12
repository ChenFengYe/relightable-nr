from PIL import Image
import torchvision.transforms as transforms
import random

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_params(cfg, size):
    w, h = size
    new_h = h
    new_w = w
    if cfg.DATASET.PREPROCESS_MODE == 'resize_and_crop':
        new_h = new_w = cfg.DATASET.LOAD_SIZE
    elif cfg.DATASET.PREPROCESS_MODE == 'scale_width_and_crop':
        new_w = cfg.DATASET.LOAD_SIZE
        new_h = cfg.DATASET.LOAD_SIZE * h // w
    elif cfg.DATASET.PREPROCESS_MODE == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(cfg.DATASET.LOAD_SIZE * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - cfg.DATASET.CROP_SIZE))
    y = random.randint(0, np.maximum(0, new_h - cfg.DATASET.CROP_SIZE))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(cfg, params, method=Image.BICUBIC, normalize=True, toTensor=True, isTrain=True):
    transform_list = []
    if 'resize' in cfg.DATASET.PREPROCESS_MODE:
        osize = [cfg.DATASET.LOAD_SIZE, cfg.DATASET.LOAD_SIZE]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif 'scale_width' in cfg.DATASET.PREPROCESS_MODE:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, cfg.DATASET.LOAD_SIZE, method)))
    elif 'scale_shortside' in cfg.DATASET.PREPROCESS_MODE:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, cfg.DATASET.LOAD_SIZE, method)))

    if 'crop' in cfg.DATASET.PREPROCESS_MODE:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], cfg.DATASET.CROP_SIZE)))

    if cfg.DATASET.PREPROCESS_MODE == 'none':
        base = 32
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if cfg.DATASET.PREPROCESS_MODE == 'fixed':
        w = cfg.DATASET.CROP_SIZE
        h = round(cfg.DATASET.CROP_SIZE / cfg.DATASET.ASPECT_RATIO)
        transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))

    if isTrain and not cfg.DATASET.NO_FLIP:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

# interpolate function in our network
def interpolate_bilinear_np(data, texture_size_i, uv_map):
    '''
    data: [H, W, C]
    sub_x: [...]
    sub_y: [...]
    return: [..., C]
    '''
    # H W
    uv_map_unit_texel = np.copy(uv_map)
    uv_map_unit_texel[..., 0] = (uv_map[:, :, 0] * (texture_size_i - 1.0))  # X
    uv_map_unit_texel[..., 1] = (uv_map[:, :, 1] * (texture_size_i - 1.0))
    uv_map_unit_texel[..., 1] = texture_size_i - 1 - uv_map_unit_texel[..., 1]

    sub_x = uv_map_unit_texel[..., 0]
    sub_y = uv_map_unit_texel[..., 1]

    x0 = np.floor(sub_x).astype(np.int64)
    x1 = x0 + 1

    y0 = np.floor(sub_y).astype(np.int64)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, data.shape[1] - 1)
    x1 = np.clip(x1, 0, data.shape[1] - 1)
    y0 = np.clip(y0, 0, data.shape[0] - 1)
    y1 = np.clip(y1, 0, data.shape[0] - 1)

    I00 = data[y0, x0, :]  # [..., C]
    I10 = data[y1, x0, :]
    I01 = data[y0, x1, :]
    I11 = data[y1, x1, :]

    w00 = (x1 - sub_x) * (y1 - sub_y)  # [...]
    w10 = (x1 - sub_x) * (sub_y - y0)
    w01 = (sub_x - x0) * (y1 - sub_y)
    w11 = (sub_x - x0) * (sub_y - y0)

    return I00 * w00[..., None] + I10 * w10[..., None] + I01 * w01[..., None] + I11 * w11[..., None]


# IUV to uv_map
# IUV = scipy.io.loadmat(r'D:\Desktop\NeuralTShow\Dataset\200729_serverl_images\chenxin\uv\020_017_IUV.mat')['uv_map']
# uv_shape = IUV.shape
# uv_map = TransferDenseposeUV(IUV)
# # uv_map = uv_map.astype(np.uint8)
#
# # vis uv
# uv_img = np.zeros((uv_shape[0], uv_shape[1], 3), dtype=np.float64)
# uv_img[:, :, 0:2] = uv_map
# plt.imshow(uv_img[:, :, :]);
# plt.axis('off');
# plt.show()
#
# # check with atlas
# Tex_Atlas = cv2.imread(r'D:\Desktop\DensePose\DensePoseData\demo_data\texture_from_SURREAL - Copy.png')[:, :, ::-1]
# plt.imshow(Tex_Atlas);
# plt.axis('off');
# plt.show()
# texture_size_i = 1200
# rs = interpolate_bilinear_np(Tex_Atlas, texture_size_i, uv_map)  # texture_size_i
# plt.imshow(rs / 255.);
# plt.axis('off');
# plt.show()
# cv2.imwrite(r'D:\Desktop\DensePose\DensePoseData\demo_data\rs.png', rs[:, :, ::-1])