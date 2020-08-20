import torch

from lib.models import network
from lib.utils import util

from utils.encoding import DataParallelModel

class FeaturePairNet(torch.nn.Module):
    def __init__(self, cfg):
        super(FeaturePairNet, self).__init__()

        self.cfg = cfg
        # texture creater
        self.texture_creater = network.TextureCreater(texture_size = cfg.MODEL.TEX_CREATER.NUM_SIZE,
                                                texture_num_ch = cfg.MODEL.TEX_CREATER.NUM_CHANNELS)
        # feature module
        self.feature_module = network.FeatureModule(nf0 = cfg.MODEL.FEATURE_MODULE.NF0,
                                    in_channels = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS + cfg.MODEL.TEX_CREATER.NUM_CHANNELS,
                                    out_channels = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                    num_down_unet = cfg.MODEL.FEATURE_MODULE.NUM_DOWN,
                                    use_gcn = False)
        # texture mapper
        self.texture_no_grad_mapper = network.TextureNoGradMapper(texture_size = cfg.MODEL.TEX_MAPPER.NUM_SIZE,
                                                texture_num_ch = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS)
        # render net
        self.render_module = network.RenderingModule(nf0 = cfg.MODEL.RENDER_MODULE.NF0,
                                    in_channels = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                    out_channels = 3,
                                    num_down_unet = cfg.MODEL.RENDER_MODULE.NUM_DOWN,
                                    use_gcn = False)

        tex_size = cfg.MODEL.TEX_CREATER.NUM_SIZE
        tex_num_ch = cfg.MODEL.TEX_CREATER.NUM_CHANNELS

        self.orig_tex = torch.ones(1, tex_size, tex_size, tex_num_ch, dtype = torch.float32)
        self.part_list = [self.feature_module, self.texture_no_grad_mapper, self.render_module]     # collect all networks
        self.part_name_list = ['feature_module', 'texture_no_grad_mapper', 'render_module']
   
    def forward(self, view_data, is_train=True):
    # def forward(self, uv_map, uv_map_ref=None, img_gt=None, img_ref=None, alpha_map=None, ROI=None, is_train=True):
        uv_map = view_data['uv_map'].cuda()
        img_ref = view_data['img_ref'].cuda()
        uv_map_ref = view_data['uv_map_ref'].cuda()
        if is_train:
            # create texture
            self.ref_tex = self.texture_creater(img_ref, uv_map_ref)
            
            # rendering module
            #   in last layer use all feature to generate attention_map
            #   feature => conv => sigmoad => attention
            # to-do
            neural_tex, attention_map = self.feature_module(self.ref_tex)          

            #   change nerual_tex with attention_map
            neural_tex = neural_tex * attention_map

            # sample texture
            neural_img = self.texture_no_grad_mapper(uv_map=uv_map, neural_tex=neural_tex)

            # rendering module
            outputs_img = self.render_module(self.neural_img)
        else:
            pass

        outputs = torch.cat((outputs_img, neural_img), dim = 1)
        return outputs

    def parameters(self):
        params = []
        for part in self.part_list:
            params.extend(list(part.parameters()))
        return params
