import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
import cv2

from gcn.dense import BasicConv, GraphConv4D, ResDynBlock4D, DenseDynBlock4D, DenseDilatedKnnGraph
from torch.nn import Sequential as Seq

from pytorch_prototyping.pytorch_prototyping import *

import neural_renderer as nr
from utils import sph_harm
from utils import misc
from utils import render
from utils import camera
from utils.util import euler_to_rot

class TextureCreater(nn.Module):
    def __init__(self,
                texture_size,
                texture_num_ch,
                fix_texture = True):
        super(TextureCreater, self).__init__()

        self.texture_size = texture_size
        self.texture_num_ch = texture_num_ch
        
        # texture = torch.ones(1, texture_size, texture_size, self.texture_num_ch, dtype = torch.float32)
        # self.texture = texture

        # fix_texture:
        # if fix_texture:
        #     self.texture.requires_grad = False
        
    def forward(self, img, uv_map, interpolater='nearset'):
        '''
        img: [N, C, H, W], C == 3
        uv_map: [N, H, W, C], C == 2
        texture_size: S
        return: [N, S, S, C], C == 3
        '''        
        texture_size = self.texture_size

        # upsize img uv_map
        if False:
            uv_map_HR = torch.nn.functional.interpolate(uv_map.permute(0,3,1,2),  # [N, H, W, C] to [N, C, H, W]
                                            size = (4*texture_size, 4*texture_size),
                                            mode = 'bilinear',
                                            align_corners = True).permute(0,2,3,1) # grid need [N, H, W, C]
            img_HR = torch.nn.functional.interpolate(img,
                                            size = (4*texture_size, 4*texture_size),
                                            mode = 'bilinear',
                                            align_corners = True)
        else:
            uv_map_HR = uv_map
            img_HR = img

        uv_map_unit_texel = (uv_map_HR * (texture_size - 1))
        uv_map_unit_texel[..., -1] = texture_size - 1 - uv_map_unit_texel[..., -1]
       
        # creat texture
        sub_u = uv_map_unit_texel[..., 0]
        sub_v = uv_map_unit_texel[..., 1]
        texture = misc.interpolate_bilinear_inv(img_HR, sub_u, sub_v, texture_size)

        # fix hole
        texture = misc.interpolate_atlas_batch(texture, interpolater_mode = interpolater)
        
        return texture

class TextureMapper(nn.Module):
    def __init__(self,
                texture_size,
                texture_num_ch,
                mipmap_level, 
                texture_merge = False,
                texture_init = None,
                fix_texture = False,
                apply_sh = False):
        '''
        texture_size: [1]
        texture_num_ch: [1]
        mipmap_level: [1]
        texture_init: torch.FloatTensor, [H, W, C]
        apply_sh: bool, [1]
        '''
        super(TextureMapper, self).__init__()

        self.register_buffer('texture_size', torch.tensor(texture_size))
        self.register_buffer('texture_num_ch', torch.tensor(texture_num_ch))
        self.register_buffer('mipmap_level', torch.tensor(mipmap_level))
        self.register_buffer('apply_sh', torch.tensor(apply_sh))
        self.register_buffer('texture_merge', torch.tensor(texture_merge))

        if texture_merge and mipmap_level != 1:
            raise ValueError('Texture merge not support mipmap now!')

        # create textures as images
        self.textures = nn.ParameterList([])
        self.textures_size = []
        for ithLevel in range(self.mipmap_level):
            texture_size_i = np.round(self.texture_size.numpy() / (2.0 ** ithLevel)).astype(np.int)
            # texture_i = torch.ones(1, texture_size_i, texture_size_i, self.texture_num_ch, dtype = torch.float32)
            texture_i = torch.zeros(1, texture_size_i, texture_size_i, self.texture_num_ch, dtype = torch.float32)
            if ithLevel != 0:
                texture_i = texture_i * 0.01
            # initialize texture
            if texture_init is not None and ithLevel == 0:
                print('Initialize neural texture with reconstructed texture')
                texture_i[..., :texture_init.shape[-1]] = texture_init[None, :]
                texture_i[..., texture_init.shape[-1]:texture_init.shape[-1] * 2] = texture_init[None, :]
            self.textures_size.append(texture_size_i)
            self.textures.append(nn.Parameter(texture_i))

        tex_flatten_mipmap_init = self.flatten_mipmap(start_ch = 0, end_ch = 6)
        tex_flatten_mipmap_init = torch.nn.functional.relu(tex_flatten_mipmap_init)
        self.register_buffer('tex_flatten_mipmap_init', tex_flatten_mipmap_init)

        if fix_texture:
            print('Fix neural textures.')
            for i in range(self.mipmap_level):
                self.textures[i].requires_grad = False

    def forward(self, uv_map, neural_tex = None, sh_basis_map = None, no_grad = False, sh_start_ch = 3):
        '''
        uv_map: [N, H, W, C]
        neural_tex: [1, C, H, W]
        sh_basis_map: [N, H, W, 9]
        return: [N, C, H, W]

        self.textures[0]: [1, H, W, C]
        '''
        if self.texture_merge and neural_tex is not None:
            if neural_tex.shape[2:4] != self.textures[0].shape[1:3]:
                print(neural_tex.shape)
                print(self.textures[0].shape)
                raise ValueError('Input nerual tex shape is not equal to max size of textures')
            self.textures[0] = torch.nn.Parameter(neural_tex.permute((0,2,3,1)))
            # self.textures[0].data = neural_tex.permute((0,2,3,1))

        for ithLevel in range(self.mipmap_level):
            texture_size_i = self.textures_size[ithLevel]
            if no_grad:
                texture_i = self.textures[ithLevel].clone().detach()
            else:
                texture_i = self.textures[ithLevel]

            ############################################################################
            # vertex texcoords map in [-1, 1]
            grid_uv_map = uv_map * 2. - 1.
            grid_uv_map[..., -1] = -grid_uv_map[..., -1] # flip v

            # sample from texture (bilinear)
            texture_batch = [texture_i for ib in range(grid_uv_map.shape[0])]
            texture_batch = torch.cat(tuple(texture_batch), dim = 0)

            if ithLevel == 0:
                output = torch.nn.functional.grid_sample(texture_batch.permute(0, 3, 1, 2), grid_uv_map, mode='bilinear', padding_mode='zeros') # , align_corners=False
            else:
                output = output + torch.nn.functional.grid_sample(texture_i.permute(0, 3, 1, 2), grid_uv_map, mode='bilinear', padding_mode='zeros') # , align_corners=False

            # Add uv_map to mask sure background is not mask
            uv_map = uv_map.permute(3, 0, 1, 2)
            mask = uv_map[0, :] == 0
            output = output.permute(1,0,2,3) # to [C N H W]
            output[:, mask] = 0
            output = output.permute(1,0,2,3) # to [N C H W]

            # [N, C, H, W]
            ############################################################################
            # # vertex texcoords map in unit of texel
            # uv_map_unit_texel = (uv_map * (texture_size_i - 1))
            # uv_map_unit_texel[..., -1] = texture_size_i - 1 - uv_map_unit_texel[..., -1]

            # # sample from texture (bilinear)
            # if ithLevel == 0:
            #     output = misc.interpolate_bilinear(texture_i[0, :], uv_map_unit_texel[..., 0], uv_map_unit_texel[..., 1]).permute((0, 3, 1, 2)) # [N, C, H, W]
            # else:
            #     output = output + misc.interpolate_bilinear(texture_i[0, :], uv_map_unit_texel[..., 0], uv_map_unit_texel[..., 1]).permute((0, 3, 1, 2)) # [N, C, H, W]
            ############################################################################

        # apply spherical harmonics
        if self.apply_sh and sh_basis_map is not None:
            output[:, sh_start_ch:sh_start_ch + 9, :, :] = output[:, sh_start_ch:sh_start_ch + 9, :, :] * sh_basis_map.permute((0, 3, 1, 2))
        
        return output

    def flatten_mipmap(self, start_ch, end_ch):
        for ithLevel in range(self.mipmap_level):
            if ithLevel == 0:
                out = self.textures[ithLevel][..., start_ch:end_ch]
            else:
                out = out + torch.nn.functional.interpolate(self.textures[ithLevel][..., start_ch:end_ch].permute(0, 3, 1, 2),
                    size = (self.textures_size[0], self.textures_size[0]),
                    mode = 'bilinear',
                    align_corners = True,
                    ).permute(0, 2, 3, 1)
        return out

class TextureNoGradMapper(nn.Module):
    def __init__(self):
        '''
        texture_size: [1]
        texture_num_ch: [1]
        '''
        super(TextureNoGradMapper, self).__init__()

    def forward(self, uv_map, neural_tex):
        '''
        uv_map: [N, H, W, C]
        neural_tex: [1, C, H, W]
        return: [N, C, H, W]
        '''
        # vertex texcoords map in [-1, 1]
        grid_uv_map = uv_map * 2. - 1.
        grid_uv_map[..., -1] = -grid_uv_map[..., -1] # flip v

        # # sample from texture (bilinear)
        # texture_batch = [neural_tex for ib in range(grid_uv_map.shape[0])]
        # texture_batch = torch.cat(tuple(texture_batch), dim = 0)

        output = torch.nn.functional.grid_sample(neural_tex, grid_uv_map, mode='bilinear', padding_mode='zeros') # , align_corners=False

        # Add uv_map to mask sure background is not mask
        uv_map = uv_map.permute(3, 0, 1, 2)
        mask = uv_map[0, :] == 0
        output = output.permute(1,0,2,3) # to [C N H W]
        output[:, mask] = 0
        output = output.permute(1,0,2,3) # to [N C H W]
        return output

class Rasterizer(nn.Module):
    def __init__(self,
                cfg, 
                obj_fp = None, 
                obj_data = None,
                preset_uv_path = None,
                global_RT = None):

        super(Rasterizer, self).__init__()

        img_size = cfg.DATASET.OUTPUT_SIZE[0]
        camera_mode = cfg.DATASET.CAM_MODE

        # load obj
        #v_attr, f_attr = []
        if obj_data != None:
            v_attr, f_attr = obj_data['v_attr'] , obj_data['f_attr']
        elif obj_fp != None:
            v_attr, f_attr = nr.load_obj(obj_fp, normalization = False)           
        else:
            raise ValueError('Not input obj data!')

        if preset_uv_path != None:
            ref_v_attr, ref_f_attr = nr.load_obj(preset_uv_path, normalization = False)
            if v_attr['v'].shape[0] != ref_v_attr['v'].shape[0]:
                raise ValueError('Refered uv mesh and cur frame mesh have no same vertices length!')
            else:
                f_attr = ref_f_attr

        vertices = v_attr['v'].cuda()
        faces = f_attr['f_v_idx'].cuda()
        vertices_texcoords = v_attr['vt'].cuda()
        faces_vt_idx = f_attr['f_vt_idx'].cuda()
        vertices_normals = v_attr['vn'].cuda()
        faces_vn_idx = f_attr['f_vn_idx'].cuda()
        self.num_vertex = vertices.shape[0]
        self.num_face = faces.shape[0]
        if cfg.DEBUG.DEBUG:
            print('vertices shape:', vertices.shape)
            print('faces shape:', faces.shape)
            print('vertices_texcoords shape:', vertices_texcoords.shape)
            print('faces_vt_idx shape:', faces_vt_idx.shape)
            print('vertices_normals shape:', vertices_normals.shape)
            print('faces_vn_idx shape:', faces_vn_idx.shape)
        self.img_size = img_size
        self.global_RT = global_RT

        # apply global_RT
        if global_RT is not None:
            vertices = torch.matmul(global_RT.to(vertices.device), torch.cat((vertices, torch.ones(self.num_vertex, 1).to(vertices.device)), dim = 1).transpose(1, 0)).transpose(1, 0)[:, :3]
            vertices_normals = torch.nn.functional.normalize(torch.matmul(global_RT[:3, :3].to(vertices.device), vertices_normals.transpose(1, 0)).transpose(1, 0), dim = 1)

        self.register_buffer('vertices', vertices[None, :, :]) # [1, num_vertex, 3]
        self.register_buffer('faces', faces[None, :, :]) # [1, num_vertex, 3]
        self.register_buffer('vertices_texcoords', vertices_texcoords[None, :, :])
        self.register_buffer('faces_vt_idx', faces_vt_idx[None, :, :])
        self.register_buffer('vertices_normals', vertices_normals[None, :, :])
        self.register_buffer('faces_vn_idx', faces_vn_idx[None, :, :])

        self.mesh_span = (self.vertices[0, :].max(dim = 0)[0] - self.vertices[0, :].min(dim = 0)[0]).max()

        # create textures
        texture_size = 4
        textures = torch.zeros(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.textures = nn.Parameter(textures)

        # setup renderer
        renderer = nr.Renderer(image_size = img_size, 
                               camera_mode = camera_mode, # 'projection' or 'orthogonal'
                               orig_size = img_size,
                               near = 0.0,
                               far = 1e5)
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        renderer.anti_aliasing = False
        renderer.fill_back = False
        self.renderer = renderer

    def update_vs(self, v_attr):
        vertices = v_attr['v'].cuda()
        # vertices_normals = v_attr['vn'].cuda()
        # apply global_RT
        if self.global_RT is not None:
            vertices = torch.matmul(self.global_RT.to(vertices.device), torch.cat((vertices, torch.ones(self.num_vertex, 1).to(vertices.device)), dim = 1).transpose(1, 0)).transpose(1, 0)[:, :3]
            # vertices_normals = torch.nn.functional.normalize(torch.matmul(self.global_RT[:3, :3].to(vertices.device), vertices_normals.transpose(1, 0)).transpose(1, 0), dim = 1)
        self.register_buffer('vertices', vertices[None, :, :]) # [1, num_vertex, 3]
        self.mesh_span = (self.vertices[0, :].max(dim = 0)[0] - self.vertices[0, :].min(dim = 0)[0]).max()

    def forward(self, proj, pose, dist_coeffs, offset, scale):
        _, depth, alpha, face_index_map, weight_map, v_uvz, faces_v_uvz, faces_v_idx = self.renderer(self.vertices, 
                                                                                            self.faces, 
                                                                                            torch.tanh(self.textures), 
                                                                                            K = proj, 
                                                                                            R = pose[:, :3, :3], 
                                                                                            t = pose[:, :3, -1, None].permute(0, 2, 1),
                                                                                            dist_coeffs = dist_coeffs,
                                                                                            offset = offset,
                                                                                            scale = scale)
        batch_size = face_index_map.shape[0]
        image_size = face_index_map.shape[1]

        # find indices of vertices on frontal face
        v_uvz[..., 0] = (v_uvz[..., 0] * 0.5 + 0.5) * depth.shape[2] # [1, num_vertex]
        v_uvz[..., 1] = (1 - (v_uvz[..., 1] * 0.5 + 0.5)) * depth.shape[1] # [1, num_vertex]
        v_depth = misc.interpolate_bilinear(depth[0, :, :, None], v_uvz[..., 0], v_uvz[..., 1]) # [1, num_vertex, 1]
        v_front_mask = ((v_uvz[0, :, 2] - v_depth[0, :, 0]) < self.mesh_span * 5e-3)[None, :] # [1, num_vertex]

        # perspective correct weight
        faces_v_z_inv_map = torch.cuda.FloatTensor(batch_size, image_size, image_size, 3).fill_(0.0)
        for i in range(batch_size):
            faces_v_z_inv_map[i, ...] = 1 / faces_v_uvz[i, face_index_map[i, ...].long()][..., -1]
        weight_map = (faces_v_z_inv_map * weight_map) * depth.unsqueeze_(-1) # [batch_size, image_size, image_size, 3]
        weight_map = weight_map.unsqueeze_(-1) # [batch_size, image_size, image_size, 3, 1]

        # uv map
        if self.renderer.fill_back:
            faces_vt_idx = torch.cat((self.faces_vt_idx, self.faces_vt_idx[:, :, list(reversed(range(self.faces_vt_idx.shape[-1])))]), dim=1).detach()
        else:
            faces_vt_idx = self.faces_vt_idx.detach()
        faces_vt = nr.vertex_attrs_to_faces(self.vertices_texcoords, faces_vt_idx) # [1, num_face, 3, 2]
        uv_map = faces_vt[:, face_index_map.long()].squeeze_(0) # [batch_size, image_size, image_size, 3, 2], before weighted combination
        uv_map = (uv_map * weight_map).sum(-2) # [batch_size, image_size, image_size, 2], after weighted combination
        uv_map = uv_map - uv_map.floor() # handle uv_map wrapping, keep uv in [0, 1]

        # normal map in world space
        if self.renderer.fill_back:
            faces_vn_idx = torch.cat((self.faces_vn_idx, self.faces_vn_idx[:, :, list(reversed(range(self.faces_vn_idx.shape[-1])))]), dim=1).detach()
        else:
            faces_vn_idx = self.faces_vn_idx.detach()
        faces_vn = nr.vertex_attrs_to_faces(self.vertices_normals, faces_vn_idx) # [1, num_face, 3, 3]
        normal_map = faces_vn[:, face_index_map.long()].squeeze_(0) # [batch_size, image_size, image_size, 3, 3], before weighted combination
        normal_map = (normal_map * weight_map).sum(-2) # [batch_size, image_size, image_size, 3], after weighted combination
        normal_map = torch.nn.functional.normalize(normal_map, dim = -1)

        # normal_map in camera space
        normal_map_flat = normal_map.flatten(start_dim = 1, end_dim = 2).permute((0, 2, 1))
        normal_map_cam = pose[:, :3, :3].matmul(normal_map_flat).permute((0, 2, 1)).reshape(normal_map.shape)
        normal_map_cam = torch.nn.functional.normalize(normal_map_cam, dim = -1)

        # position_map in world space
        faces_v = nr.vertex_attrs_to_faces(self.vertices, faces_v_idx) # [1, num_face, 3, 3]
        position_map = faces_v[0, face_index_map.long()] # [batch_size, image_size, image_size, 3, 3], before weighted combination
        position_map = (position_map * weight_map).sum(-2) # [batch_size, image_size, image_size, 3], after weighted combination

        # position_map in camera space
        position_map_flat = position_map.flatten(start_dim = 1, end_dim = 2).permute((0, 2, 1))
        position_map_cam = pose[:, :3, :3].matmul(position_map_flat).permute((0, 2, 1)).reshape(position_map.shape) + pose[:, :3, -1][:, None, None, :]

        return uv_map, alpha, face_index_map, weight_map, faces_v_idx, normal_map, normal_map_cam, faces_v, faces_vt, position_map, position_map_cam, depth, v_uvz, v_front_mask

class RenderingModule(nn.Module):
    def __init__(self,
                 nf0,
                 in_channels,
                 out_channels,
                 num_down_unet = 5,
                 out_channels_gcn = 512,
                 use_gcn = True,
                 outermost_highway_mode = 'concat'):
        super().__init__()

        self.register_buffer('nf0', torch.tensor(nf0))
        self.register_buffer('in_channels', torch.tensor(in_channels))
        self.register_buffer('out_channels', torch.tensor(out_channels))
        self.register_buffer('num_down_unet', torch.tensor(num_down_unet))
        self.register_buffer('out_channels_gcn', torch.tensor(out_channels_gcn))

        self.net = Unet(in_channels = in_channels,
                 out_channels = out_channels,
                 outermost_linear = True,
                 use_dropout = False,
                #  use_dropout = True,
                 dropout_prob = 0.1,
                 nf0 = nf0,
                 norm = nn.InstanceNorm2d,
                #  norm = nn.BatchNorm2d,# chenxin 200803 temporary change for debug
                 max_channels = 8 * nf0,
                 num_down = num_down_unet,
                 out_channels_gcn = out_channels_gcn,
                 use_gcn = use_gcn,
                 outermost_highway_mode = outermost_highway_mode)

        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, v_fea = None):
        x = self.net(input, v_fea)
        # return self.tanh(x)
        return self.sigmoid(x).clone()

class FeatureModule(nn.Module):
    def __init__(self,
                 nf0,
                 in_channels,
                 out_channels,
                 num_down_unet = 5,
                 out_channels_gcn = 512,
                 use_gcn = True,
                 outermost_highway_mode = 'concat',
                 backbone = 'Unet'):
        super().__init__()

        self.register_buffer('nf0', torch.tensor(nf0))
        self.register_buffer('in_channels', torch.tensor(in_channels))
        self.register_buffer('out_channels', torch.tensor(out_channels))
        self.register_buffer('num_down_unet', torch.tensor(num_down_unet))
        self.register_buffer('out_channels_gcn', torch.tensor(out_channels_gcn))

        self.net = eval(backbone)(in_channels = in_channels,
                 out_channels = out_channels,
                 outermost_linear = True,
                 use_dropout = False,
                 dropout_prob = 0.1,
                 nf0 = nf0,
                 norm = nn.InstanceNorm2d,
                 max_channels = 8 * nf0,
                 num_down = num_down_unet,
                 out_channels_gcn = out_channels_gcn,
                 use_gcn = use_gcn,
                 outermost_highway_mode = outermost_highway_mode)

        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, orig_texs, neural_tex = None, v_fea = None):
        '''
        orig_tex: [N, H, W, 3]
        neural_tex: [1, H, W, C]       
        return: [N, C, H, W]
        '''
        # cat neural tex for each batch
        if neural_tex is not None:
            repeat_size = (int(orig_texs.shape[0]/neural_tex.shape[0]),1,1,1)
            neural_texs = neural_tex.repeat(repeat_size)
            cat_texs = torch.cat((orig_texs, neural_texs), 3).permute(0,3,1,2)
        else:
            cat_texs = orig_texs

        # unet
        x = self.net(cat_texs, v_fea)
        # # average each batch
        # x_mean = torch.mean(x, dim=0, keepdim=True)
        # # return self.tanh(x_mean)
        return self.sigmoid(x)        

class AttentionFeatureModule(FeatureModule):
    def __init__(self,
                 nf0,
                 in_channels,
                 out_channels,
                 num_down_unet = 5,
                 out_channels_gcn = 512,
                 use_gcn = True,
                 outermost_highway_mode = 'concat'):
        backbone = 'AttentionUnet'
        super().__init__(nf0,
                         in_channels,
                         out_channels,
                         num_down_unet,
                         out_channels_gcn,
                         use_gcn,
                         outermost_highway_mode,
                         backbone)

        self.sigmoid = nn.Sigmoid()

    def forward(self, orig_texs, neural_tex = None, v_fea = None):
        '''
        orig_tex: [N, H, W, 3]
        neural_tex: [1, H, W, C]       
        return: [N, C, H, W]
        '''
        # cat neural tex for each batch
        if neural_tex is not None:
            repeat_size = (int(orig_texs.shape[0]/neural_tex.shape[0]),1,1,1)
            neural_texs = neural_tex.repeat(repeat_size)
            cat_texs = torch.cat((orig_texs, neural_texs), 3).permute(0,3,1,2)
        else:
            cat_texs = orig_texs

        # unet
        x = self.net(cat_texs, v_fea)

        feature_ch = x[:,:-1,:,:]
        attention_ch = x[:,-1,:,:][:,None,:,:]
        attention_ch = self.sigmoid(attention_ch)
        return feature_ch, attention_ch

class DenseDeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DenseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.kernel_size
        act = opt.act_type
        norm = opt.norm_type
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv_type
        c_growth = channels
        self.n_blocks = opt.n_blocks
        num_v = opt.num_v_gcn
        out_channels = opt.out_channels_gcn

        self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv4D(opt.in_channels, channels, conv, act, norm, bias)

        if opt.block_type.lower() == 'res':
            self.backbone = Seq(*[ResDynBlock4D(channels, k, 1+i, conv, act, norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
        elif opt.block_type.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock4D(channels+c_growth*i, c_growth, k, 1+i, conv, act,
                                                  norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
        else:
            raise NotImplementedError('{} is not implemented. Please check.\n'.format(opt.block))
        self.fusion_block = BasicConv([channels+c_growth*(self.n_blocks-1), 1024], act, None, bias)
        self.prediction = Seq(*[BasicConv([1+channels+c_growth*(self.n_blocks-1), 512, 256], act, None, bias),
                                BasicConv([256, 64], act, None, bias)])
        self.linear = Seq(*[utils.spectral_norm(nn.Linear(num_v,2048)), utils.spectral_norm(nn.Linear(2048, out_channels))])

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
            elif isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
            elif isinstance(m,torch.nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, inputs):
        data = torch.cat((inputs.pos,inputs.x),1).unsqueeze(0).unsqueeze(-1)
        feats = [self.head(data.transpose(2,1), self.knn(data[:,:, 0:3]))]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1]))
        feats = torch.cat(feats, 1)
        fusion, _ = torch.max(self.fusion_block(feats), 1, keepdim=True)

        fea = self.linear(fusion.view(-1)).unsqueeze(0)
        return fea

class Interpolater(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, sub_x, sub_y):
        '''
        data: [N, H, W, C] or [1, H, W, C]
        sub_x: [N, ...]
        sub_y: [N, ...]
        return: [N, ..., C]
        '''
        if data.shape[0] == 1:
            return misc.interpolate_bilinear(data[0, :], sub_x, sub_y) # [N, ..., C]
        elif data.shape[0] == sub_x.shape[0]:
            out = []
            for i in range(data.shape[0]):
                out.append(misc.interpolate_bilinear(data[i, :], sub_x[i, :], sub_y[i, :])) # [..., C]
            return torch.stack(out) # [N, ..., C]
        else:
            raise ValueError('data.shape[0] should be 1 or batch size')

class InterpolaterVertexAttr(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v_attr, faces_v_idx, face_index_map, weight_map):
        '''
        v_attr: [N, num_vertex, num_attr] or [1, num_vertex, num_attr]
        faces_v_idx: [N, num_face, 3]
        face_index_map: [N, H, W]
        weight_map: [N, H, W, 3, 1]
        return: [N, H, W, num_attr]
        '''
        return render.interp_vertex_attr(v_attr, faces_v_idx, face_index_map, weight_map)

class Mesh(nn.Module):
    def __init__(self, obj_fp, global_RT = None):
        super().__init__()
        
        # load obj
        v_attr, f_attr = nr.load_obj(obj_fp, normalization = False)
        v = v_attr['v'].cpu() # [num_vertex, 3]
        vn = v_attr['vn'].cpu() # [num_vertex, 3]
        self.num_vertex = v.shape[0]

        # compute useful infomation
        self.v_orig = v.clone()
        self.vn_orig = vn.clone()
        self.span_orig = v.max(dim = 0)[0] - v.min(dim = 0)[0]
        self.span_max_orig = self.span_orig.max()
        self.center_orig = v.mean(dim = 0)

        # apply global_RT
        if global_RT is not None:
            v = torch.matmul(global_RT.to(v.device), torch.cat((v, torch.ones(self.num_vertex, 1).to(v.device)), dim = 1).transpose(1, 0)).transpose(1, 0)[:, :3]
            vn = torch.nn.functional.normalize(torch.matmul(global_RT[:3, :3].to(vn.device), vn.transpose(1, 0)).transpose(1, 0), dim = 1)
        
        self.register_buffer('v', v)
        self.register_buffer('vn', vn)
        print('v shape:', self.v.shape)
        print('vn shape:', self.vn.shape)

        # compute useful infomation
        self.span = v.max(dim = 0)[0] - v.min(dim = 0)[0]
        self.span_max = self.span.max()
        self.center = v.mean(dim = 0)

    def forward(self):
        pass

class RaysLTChromLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rays_lt, alpha_map, img = None):
        '''
        rays_lt: [N, num_ray, C, H, W]
        alpha_map: [N, 1, H, W]
        img: [N, C, H, W]
        return: [1]
        '''
        rays_lt_chrom = torch.nn.functional.normalize(rays_lt, dim = 2) # [N, num_ray, C, H, W]
        rays_lt_chrom_mean = rays_lt_chrom.mean(dim = 1)[:, None, :, :, :] # [N, 1, C, H, W]
        rays_lt_chrom_mean = torch.nn.functional.normalize(rays_lt_chrom_mean, dim = 2) # [N, 1, C, H, W]
        rays_lt_chrom_diff = (1 - (rays_lt_chrom * rays_lt_chrom_mean).sum(2)) * alpha_map # [N, num_ray, H, W]
        if img is not None:
            # weight by image intensity
            weight = (img.norm(dim = 1, keepdim = True) * 20).clamp(max = 1.0)
            rays_lt_chrom_diff = rays_lt_chrom_diff * weight # [N, num_ray, H, W]
        loss_rays_lt_chrom = rays_lt_chrom_diff.sum() / alpha_map.sum() / rays_lt_chrom_diff.shape[1]
        return loss_rays_lt_chrom, rays_lt_chrom, rays_lt_chrom_mean, rays_lt_chrom_diff

####################################################################################################################################
################################################## Modules for Ray Based Renderer ##################################################
####################################################################################################################################
class RaySampler(nn.Module):
    def __init__(self, num_azi, num_polar, interval_polar = 5, mode = 'reflect'):
        super().__init__()

        self.register_buffer('num_azi', torch.tensor(num_azi))
        self.register_buffer('num_polar', torch.tensor(num_polar))
        self.register_buffer('interval_polar', torch.tensor(interval_polar))
        self.mode = mode

        roty_rad = np.arange(1, num_polar + 1) * interval_polar * np.pi / 180.0
        rotz_rad = np.arange(num_azi) * 2 * np.pi / num_azi
        roty_rad, rotz_rad = np.meshgrid(roty_rad, rotz_rad, sparse = False)
        roty_rad = roty_rad.flatten()
        rotz_rad = rotz_rad.flatten()
        rotx_rad = np.zeros_like(roty_rad)
        self.rot_rad = np.vstack((rotx_rad, roty_rad, rotz_rad)) # [3, num_ray]
        self.num_ray = self.rot_rad.shape[1] + 1

        Rs = np.zeros((self.num_ray, 3, 3), dtype = np.float32)
        Rs[0, :, :] = np.eye(3)
        for i in range(self.num_ray - 1):
            Rs[i + 1, :, :] = euler_to_rot(self.rot_rad[:, i])
        self.register_buffer('Rs', torch.from_numpy(Rs)) # [num_ray, 3, 3]
        
        # pivots in tangent space
        pivots_dir = torch.matmul(self.Rs, torch.FloatTensor([0, 0, 1], device = self.Rs.device)[:, None])[..., 0].permute((1, 0)) # [3, num_ray]
        self.register_buffer('pivots_dir', pivots_dir)

    def forward(self, TBN_matrices, view_dir_map_tangent, alpha_map):
        '''
        TBN_matrices: [N, ..., 3, 3]
        view_dir_map_tangent: [N, ..., 3]
        alpha_map: [N, ..., 1]
        return0: [N, ..., 3, num_ray]
        return1: [N, ..., 2, num_ray]
        return2: [N, ..., 3, num_ray]
        '''
        if self.mode == 'reflect':
            # reflect view directions around pivots
            rays_dir_tangent = camera.get_reflect_dir(view_dir_map_tangent[..., None], self.pivots_dir, dim = -2) * alpha_map[..., None] # [N, ..., 3, num_ray]
            # transform to world space
            num_ray = rays_dir_tangent.shape[-1]
            rays_dir = torch.matmul(TBN_matrices.reshape((-1, 3, 3)), rays_dir_tangent.reshape((-1, 3, num_ray))).reshape((*(TBN_matrices.shape[:-1]), -1)) # [N, ..., 3, num_ray]
        else:
            rays_dir_tangent = self.pivots_dir # [3, num_ray]
            # transform to world space
            num_ray = rays_dir_tangent.shape[-1]
            rays_dir = torch.matmul(TBN_matrices.reshape((-1, 3, 3)), rays_dir_tangent).reshape((*(TBN_matrices.shape[:-1]), -1)) # [N, ..., 3, num_ray]
        
        rays_dir = torch.nn.functional.normalize(rays_dir, dim = -2)

        # get rays uv on light probe
        rays_uv = render.spherical_mapping_batch(rays_dir.transpose(1, -2)).transpose(1, -2) # [N, ..., 2, num_ray]
        rays_uv = rays_uv * alpha_map[..., None] - (alpha_map[..., None] == 0).to(rays_dir.dtype) # [N, ..., 2, num_ray]

        return rays_dir, rays_uv, rays_dir_tangent

class RayRenderer(nn.Module):
    def __init__(self, lighting_model, interpolater):
        super().__init__()
        self.lighting_model = lighting_model
        self.interpolater = interpolater

    def forward(self, albedo_specular, rays_uv, rays_lt, lighting_idx = None, lp = None, albedo_diffuse = None, num_ray_diffuse = 0, no_albedo = False, seperate_albedo = False, lp_scale_factor = 1):
        '''
        rays_uv: [N, H, W, 2, num_ray]
        rays_lt: [N, num_ray, C, H, W]
        albedo_specular: [N, C, H, W]
        albedo_diffuse: [N, C, H, W]
        return: [N, C, H, W]
        '''
        num_ray = rays_uv.shape[-1] - num_ray_diffuse

        # get light probe
        if lp is None:
            lp = self.lighting_model(lighting_idx, is_lp = True) # [N, H, W, C]
        lp = lp * lp_scale_factor

        # get rays color
        rays_color = self.interpolater(lp, (rays_uv[..., 0, :] * float(lp.shape[2])).clamp(max = lp.shape[2] - 1), (rays_uv[..., 1, :] * float(lp.shape[1])).clamp(max = lp.shape[1] - 1)).permute((0, -2, -1, 1, 2)) # [N, num_ray, C, H, W]

        # get specular light transport map
        ltt_specular_map = (rays_lt[:, :num_ray, ...] * rays_color[:, :num_ray, ...]).sum(1) / num_ray # [N, C, H, W]
        # get specular component
        if no_albedo:
            out_specular = ltt_specular_map
        else:
            out_specular = albedo_specular * ltt_specular_map

        if num_ray_diffuse > 0:
            # get diffuse light transport map
            ltt_diffuse_map = (rays_lt[:, num_ray:, ...] * rays_color[:, num_ray:, ...]).sum(1) / num_ray_diffuse # [N, C, H, W]
            # get diffuse component
            if no_albedo:
                out_diffuse = ltt_diffuse_map
            else:
                if seperate_albedo:
                    out_diffuse = albedo_diffuse * ltt_diffuse_map
                else:
                    out_diffuse = albedo_specular * ltt_diffuse_map
        else:
            ltt_diffuse_map = torch.zeros_like(ltt_specular_map)
            out_diffuse = torch.zeros_like(out_specular)

        if out_diffuse is not None:
            out = out_specular + out_diffuse
        else:
            out = out_specular

        return out, out_specular, out_diffuse, ltt_specular_map, ltt_diffuse_map, rays_color, lp

##########################################################################################################################
################################################## Modules for Lighting ##################################################
##########################################################################################################################
# Spherical Harmonics model
class LightingSH(nn.Module):
    def __init__(self, l_dir, lmax, num_lighting = 1, num_channel = 3, init_coeff = None, fix_params = False, lp_recon_h = 100, lp_recon_w = 200):
        '''
        l_dir: torch.Tensor, [3, num_sample], sampled light directions
        lmax: int, maximum SH degree
        num_lighting: int, number of lighting
        num_channel: int, number of color channels
        init_coeff: torch.Tensor, [num_lighting, num_basis, num_channel] or [num_basis, num_channel], initial coefficients
        fix_params: bool, whether fix parameters
        '''
        super().__init__()

        self.num_sample = l_dir.shape[1]
        self.lmax = lmax
        self.num_basis = (lmax + 1) ** 2
        self.num_lighting = num_lighting
        self.num_channel = num_channel
        self.fix_params = fix_params
        self.lp_recon_h = lp_recon_h
        self.lp_recon_w = lp_recon_w

        # get basis value on sampled directions
        print('LightingSH.__init__: Computing SH basis value on sampled directions...')
        basis_val = torch.from_numpy(sph_harm.evaluate_sh_basis(lmax = lmax, directions = l_dir.cpu().detach().numpy().transpose())).to(l_dir.dtype).to(l_dir.device)
        self.register_buffer('basis_val', basis_val) # [num_sample, num_basis]

        # basis coefficients as learnable parameters
        self.coeff = nn.Parameter(torch.zeros((num_lighting, self.num_basis, num_channel), dtype = torch.float32)) # [num_lighting, num_basis, num_channel]
        # initialize basis coeffients
        if init_coeff is not None:
            if init_coeff.dim == 2:
                init_coeff = init_coeff[None, :].repeat((num_lighting, 1, 1))
            self.coeff.data = init_coeff
        # change to non-learnable
        if self.fix_params:
            self.coeff.requires_grad_(False)
        
        # precompute light samples
        l_samples = sph_harm.reconstruct_sh(self.coeff.data, self.basis_val)
        self.register_buffer('l_samples', l_samples) # [num_lighting, num_sample, num_channel]

        # precompute SH basis value for reconstructing light probe
        lp_samples_recon_v, lp_samples_recon_u = torch.meshgrid([torch.arange(start = 0, end = self.lp_recon_h, step = 1, dtype = torch.float32) / (self.lp_recon_h - 1), 
                                                                torch.arange(start = 0, end = self.lp_recon_w, step = 1, dtype = torch.float32) / (self.lp_recon_w - 1)])
        lp_samples_recon_uv = torch.stack([lp_samples_recon_u, lp_samples_recon_v]).flatten(start_dim = 1, end_dim = -1)
        lp_samples_recon_dir = render.spherical_mapping_inv(lp_samples_recon_uv).permute((1, 0)).cpu().detach().numpy()

        basis_val_recon = torch.from_numpy(sph_harm.evaluate_sh_basis(lmax = self.lmax, directions = lp_samples_recon_dir)).to(l_dir.dtype).to(l_dir.device)
        self.register_buffer('basis_val_recon', basis_val_recon) # [num_lp_pixel, num_basis]

    def forward(self, lighting_idx = None, coeff = None, is_lp = None):
        '''
        coeff: torch.Tensor, [num_lighting, num_basis, num_channel]
        return: [1, num_lighting, num_sample, num_channel] or [1, num_sample, num_channel]
        '''
        if coeff is not None:
            if is_lp:
                out = self.reconstruct_lp(coeff)[None, :] # [1, num_lighting, H, W, C]
            else:
                out = sph_harm.reconstruct_sh(coeff, self.basis_val)[None, :]
        elif lighting_idx is not None:
            if is_lp:
                out = self.reconstruct_lp(self.coeff[lighting_idx, :])[None, :] # [1, H, W, C]
            else:
                if self.fix_params:
                    out = self.l_samples[lighting_idx, ...][None, :]
                else:
                    out = sph_harm.reconstruct_sh(self.coeff[lighting_idx, ...][None, :], self.basis_val)
        else:
            if is_lp:
                out = self.reconstruct_lp(self.coeff)[None, :] # [1, num_lighting, H, W, C]
            else:
                if self.fix_params:
                    out = self.l_samples[None, :]
                else:
                    out = sph_harm.reconstruct_sh(self.coeff, self.basis_val)[None, :]

        return out

    def get_lighting_params(self, lighting_idx):
        return self.coeff[lighting_idx, :] # [num_sample, num_channel]

    def normalize_lighting(self, lighting_ref_idx):
        lighting_ref_norm = self.coeff[lighting_ref_idx, :].norm('fro')
        norm_scale_factor = lighting_ref_norm / self.coeff.norm('fro', dim = [1, 2])
        norm_scale_factor[lighting_ref_idx] = 1.0
        self.coeff *= norm_scale_factor[:, None, None]

    def reconstruct_lp(self, coeff):
        '''
        coeff: [num_basis, C] or [num_lighting, num_basis, C]
        '''
        lp_recon = sph_harm.reconstruct_sh(coeff, self.basis_val_recon).reshape((int(self.lp_recon_h), int(self.lp_recon_w), -1)) # [H, W, C] or [num_lighting, H, W, C]
        return lp_recon

# Light Probe model
class LightingLP(nn.Module):
    def __init__(self, l_dir, num_lighting = 1, num_channel = 3, lp_dataloader = None, fix_params = False, lp_img_h = 1600, lp_img_w = 3200):
        '''
        l_dir: torch.FloatTensor, [3, num_sample], sampled light directions
        num_lighting: int, number of lighting
        num_channel: int, number of color channels
        lp_dataloader: dataloader for light probes (if not None, num_lighting is ignored)
        fix_params: bool, whether fix parameters
        '''
        super().__init__()

        self.register_buffer('l_dir', l_dir) # [3, num_sample]
        self.num_sample = l_dir.shape[1]
        self.num_lighting = num_lighting
        self.num_channel = num_channel
        self.fix_params = fix_params
        self.lp_img_h = lp_img_h
        self.lp_img_w = lp_img_w
        
        if lp_dataloader is not None:
            self.num_lighting = len(lp_dataloader)

        # spherical mapping to get light probe uv
        l_samples_uv = render.spherical_mapping(l_dir)
        self.register_buffer('l_samples_uv', l_samples_uv) # [2, num_sample]

        # light samples as learnable parameters
        self.l_samples = nn.Parameter(torch.zeros((self.num_lighting, self.num_sample, self.num_channel), dtype = torch.float32)) # [num_lighting, num_sample, num_channel]
        
        # initialize light samples from light probes
        if lp_dataloader is not None:
            self.num_lighting = len(lp_dataloader)
            lp_idx = 0
            lps = []
            for lp in lp_dataloader:
                lp_img = lp['lp_img'][0, :].permute((1, 2, 0))
                lps.append(torch.from_numpy(cv2.resize(lp_img.cpu().detach().numpy(), (lp_img_w, lp_img_h), interpolation = cv2.INTER_AREA))) # [H, W, C]
                lp_img = lps[-1]
                self.l_samples.data[lp_idx, :] = misc.interpolate_bilinear(lp_img.to(self.l_samples_uv.device), (self.l_samples_uv[None, 0, :] * float(lp_img.shape[1])).clamp(max = lp_img.shape[1] - 1), (self.l_samples_uv[None, 1, :] * float(lp_img.shape[0])).clamp(max = lp_img.shape[0] - 1))[0, :]
                lp_idx += 1

            lps = torch.stack(lps)
            self.register_buffer('lps', lps) # [num_lighting, H, W, C]

        # change to non-learnable
        if self.fix_params:
            self.l_samples.requires_grad_(False)

    def forward(self, lighting_idx = None, is_lp = False):
        '''
        return: [1, num_lighting, num_sample, num_channel] or [1, num_sample, num_channel]
        '''
        if is_lp:
            if lighting_idx is None:
                return self.lps[None, :]
            else:
                return self.lps[lighting_idx, :][None, :]
        else:
            if lighting_idx is None:
                return self.l_samples[None, :]
            else:
                return self.l_samples[lighting_idx, :][None, :]

    def fit_sh(self, lmax):
        print('LightingLP.fit_sh: Computing SH basis value on sampled directions...')
        basis_val = torch.from_numpy(sph_harm.evaluate_sh_basis(lmax = lmax, directions = self.l_dir.cpu().detach().numpy().transpose())).to(self.l_dir.dtype).to(self.l_dir.device) # [num_sample, num_basis]
        sh_coeff = sph_harm.fit_sh_coeff(samples = self.l_samples.to(self.l_dir.device), sh_basis_val = basis_val) # [num_lighting, num_basis, num_channel]
        self.register_buffer('sh_coeff', sh_coeff)
        return

##########################################################################################################################
################################################## Modules for Aligning ##################################################
##########################################################################################################################

class AlignModule(nn.Module):
    def __init__(self, input_channels, ref_channels, mid_channels, out_channels):
        super(AlignModule, self).__init__()
        self.down_h = nn.Conv2d(input_channels, mid_channels, 1, bias=False)
        self.down_l = nn.Conv2d(ref_channels, mid_channels, 1, bias=False)
        # self.down_m = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.flow_make = nn.Conv2d(mid_channels*2, 2, kernel_size=3, padding=1, bias=False)

    # def forward(self, input, ref):
    #     '''
    #     NCHW
    #     '''
    #     input_orign = input
    #     h, w = ref.size()[2:]
    #     size = (h, w)
    #     ref = self.down_l(ref)
    #     input = self.down_h(input)
    #     # input = F.interpolate(input,size=size, mode="bilinear", align_corners=False)
    #     flow = self.flow_make(torch.cat([input, ref], 1))
    #     input = self.flow_warp(input_orign, flow, size=size)
    #     # input = self.down_m(input)
    #     return input

    def forward(self, input, ref):
        '''
        NCHW
        '''
        input_clone = input.clone()
        h, w = ref.size()[2:]
        size = (h, w)
        ref = self.down_l(ref)
        input_clone = self.down_h(input_clone)
        flow = self.flow_make(torch.cat([input_clone, ref], 1))
        input = self.flow_warp(input, flow, size=size)
        # input = self.down_m(input)
        return input

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input)
        grid = grid + flow.permute(0, 2, 3, 1) / norm * 10

        output = F.grid_sample(input, grid, mode='nearest', padding_mode='zeros')
        # output = F.grid_sample(input, grid, mode='bilinear')
        return output

##########################################################################################################################
################################################## Modules for GAN ##################################################
##########################################################################################################################
'''
From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
'''
from torch.nn import init
import functools

class Identity(nn.Module):
    def forward(self, x):
        return x

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net        

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator_HD(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_HD(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator_HD, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'multiscale':
        use_sigmoid = True
        num_D = 3
        getIntermFeat = False
        net = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

##########################################################################################################################
################################################## Utils ##################################################
##########################################################################################################################
from torch.optim import lr_scheduler

def get_scheduler(optimizer, cfg):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              cfg.TRAIN.LR_MODE is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if cfg.TRAIN.LR_MODE == 'multistep': # same as linear 
        scheduler = lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR)
    # elif cfg.TRAIN.LR_MODE == 'step':
    #     scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    # elif cfg.TRAIN.LR_MODE == 'plateau':
    #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    # elif cfg.TRAIN.LR_MODE == 'cosine':
    #     scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', cfg.TRAIN.LR_MODE)
    return scheduler
