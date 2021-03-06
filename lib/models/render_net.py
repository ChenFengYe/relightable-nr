import torch

from lib.models import network
from lib.utils.model import custom_load

from pytorch_prototyping.pytorch_prototyping import *
from collections import OrderedDict

class RenderNet(torch.nn.Module):
    def __init__(self, cfg):
        super(RenderNet, self).__init__()

        self.cfg = cfg
        # texture mapper
        self.texture_mapper = network.TextureMapper(texture_size = cfg.MODEL.TEX_MAPPER.NUM_SIZE,
                                                texture_num_ch = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                                texture_merge = cfg.MODEL.TEX_MAPPER.MERGE_TEX,
                                                mipmap_level = cfg.MODEL.TEX_MAPPER.MIPMAP_LEVEL,
                                                apply_sh = cfg.MODEL.TEX_MAPPER.SH_BASIS)
        # align
        self.align_module = network.AlignModule(input_channels = 2,
                                    ref_channels = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                    mid_channels = cfg.MODEL.ALIGN_MODULE.MID_CHANNELS,
                                    out_channels = 2)

        # render net
        self.render_module = network.RenderingModule(nf0 = cfg.MODEL.RENDER_MODULE.NF0,
                                    in_channels = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                    out_channels = 3,
                                    num_down_unet = cfg.MODEL.RENDER_MODULE.NUM_DOWN,
                                    use_gcn = False)
        texture_mapper_module = self.texture_mapper
        render_module = self.render_module
        align_module = self.align_module

        self.part_list = [texture_mapper_module, render_module, align_module]     # collect all networks
        self.part_name_list = ['texture_mapper', 'render_module', 'align_module']

    def init_rasterizer(self, obj_data, global_RT):
        self.rasterizer = network.Rasterizer(self.cfg,
                            obj_data = obj_data,
                            # preset_uv_path = cfg.DATASET.UV_PATH,
                            global_RT = global_RT)

    def load_checkpoint(self, checkpoint_path = None):
        if checkpoint_path:
            print(' Checkpoint_path : %s'%(checkpoint_path))
            custom_load(self.part_list, self.part_name_list, checkpoint_path)
        else:
            print(' Not load params. ')

    def project_with_rasterizer(self, cur_obj_path, objs, view_trgt):
        # get uvmap alpha
        uv_map = []            
        alpha_map = []
        # raster module
        frame_idxs = view_trgt['f_idx'].numpy()
        for batch_idx, frame_idx in enumerate(frame_idxs):
            obj_path = view_trgt['obj_path'][batch_idx]
            if cur_obj_path != obj_path:
                cur_obj_path = obj_path
                obj_data = objs[frame_idx]
                self.rasterizer.update_vs(obj_data['v_attr'])

            proj = view_trgt['proj'].cuda()[batch_idx, ...]
            pose = view_trgt['pose'].cuda()[batch_idx, ...]
            dist_coeffs = view_trgt['dist_coeffs'].cuda()[batch_idx, ...]
            uv_map_single, alpha_map_single, _, _, _, _, _, _, _, _, _, _, _, _ = \
                self.rasterizer(proj = proj[None, ...], 
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
        return uv_map, alpha_map, cur_obj_path

    # def set_parallel(self, gpus):       
    #     self.texture_mapper.cuda()
    #     self.render_module.cuda()
    #     #interpolater.cuda()
    #     if hasattr(self, 'rasterizer'):
    #         self.rasterizer.cuda()

    # def set_mode(self, is_train = True):
    #     if is_train:
    #         # set to training mode
    #         self.texture_mapper.train()
    #         self.render_module.train()
    #         if hasattr(self, 'rasterizer'):
    #             self.rasterizer.eval()      # not train now

    #         print(" number of generator parameters:")
    #         self.cfg.defrost()
    #         self.cfg.MODEL.TEX_MAPPER.NUM_PARAMS = util.print_network(self.texture_mapper).item()
    #         # cfg.MODEL.FEATURE_MODULE.NUM_PARAMS = util.print_network(feature_module).item()
    #         self.cfg.MODEL.RENDER_MODULE.NUM_PARAMS = util.print_network(self.render_module).item()
    #         self.cfg.freeze()
    #         print("*" * 100)
    #     else:
    #         pass
    
    def get_atalas(self):
        # atlas = self.texture_mapper.textures[0].clone().detach().cpu().permute(0,3,1,2)
        # atlas_set = []
        # for ic in atlas.shape[1]/3:
        #     channel_range = [3*ic, 3*(ic+1)]
        #     atlas_set.append(atlas[1,channel_range[0]:channel_range[1],:,:])
        # atlas_set = torch.cat(atlas_set, dim = 0)
        # return atlas_set
        return self.texture_mapper.textures[0].clone().detach().cpu().permute(0,3,1,2)[:, 0:3, :, :]

    def forward(self, uv_map, img_gt=None, alpha_map=None, ROI=None):
        # first sample texture
        neural_img = self.texture_mapper.forward(uv_map = uv_map.clone().detach(), no_grad = True)

        # align uvmap
        # to-do
        # 1 change bone to res-net generater for 
        #             1.1 fix lighting change
        # 2 SPADE
        #
        #
        #
        aligned_uv = self.align_module.forward(input = uv_map.permute(0,3,1,2), ref = neural_img).permute(0,2,3,1)

        # clamp uv [-1, 1]
        aligned_uv = torch.clamp(aligned_uv, 0, 1)

        # re-sample texture
        neural_img = self.texture_mapper.forward(uv_map = aligned_uv)

        # rendering module
        outputs_img = self.render_module.forward(neural_img.clone(), None)
        
        outputs = torch.cat((outputs_img, neural_img, aligned_uv.permute(0,3,1,2)), dim = 1)
        return outputs

    def parameters(self):
        params = []
        for part in self.part_list:
            params.extend(list(part.parameters()))
        return params
        # return list(self.texture_mapper.parameters())+list(self.render_module.parameters())+list(self.align_module.parameters())