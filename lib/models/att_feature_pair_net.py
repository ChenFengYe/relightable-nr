import torch

from lib.models import network
from lib.utils import util

class AttFeaturePairNet(torch.nn.Module):
    def __init__(self, cfg):
        super(AttFeaturePairNet, self).__init__()

        self.cfg = cfg
        # texture creater
        # self.texture_creater = network.TextureCreater(texture_size = cfg.MODEL.TEX_CREATER.NUM_SIZE,
        #                                         texture_num_ch = cfg.MODEL.TEX_CREATER.NUM_CHANNELS)
        # feature module
        self.att_feature_module = network.AttentionFeatureModule(nf0 = cfg.MODEL.FEATURE_MODULE.NF0,
                                    in_channels = 3,
                                    out_channels = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                    num_down_unet = cfg.MODEL.FEATURE_MODULE.NUM_DOWN,
                                    use_gcn = False)
        # texture mapper
        self.texture_no_grad_mapper = network.TextureNoGradMapper()
        # render net
        self.render_module = network.RenderingModule(nf0 = cfg.MODEL.RENDER_MODULE.NF0,
                                    in_channels = cfg.MODEL.TEX_MAPPER.NUM_CHANNELS,
                                    out_channels = 3,
                                    num_down_unet = cfg.MODEL.RENDER_MODULE.NUM_DOWN,
                                    use_gcn = False)

    def my_device(self):
        devices = ({param.device for param in self.parameters()} |
                {buf.device for buf in self.buffers()})
        if len(devices) != 1:
            raise RuntimeError('Cannot determine device: {} different devices found'
                            .format(len(devices)))
        return next(iter(devices))

    # def get_atalas(self):
    #     atalas = torch.cat((self.ref_tex.permute(0,3,1,2)[:, 0:3, :, :], 
    #                         self.neural_tex[:, 0:3, :, :],
    #                         self.att_neural_tex[:, 0:3, :, :],), dim=1)
    #     return atalas.clone().detach().cpu()

    def get_atalas(self, ref_tex, neural_tex, att_neural_tex, attention_map):
        atalas = torch.cat((ref_tex[:, 0:3, :, :], 
                            neural_tex[:, 0:3, :, :],
                            att_neural_tex[:, 0:3, :, :],
                            attention_map), dim=1)
        # return atalas.clone().detach().cpu()
        return atalas

    def forward(self, view_data, is_train=True):
        # setup data
        uv_map = view_data['uv_map'].to(self.my_device())
        # img_ref = view_data['img_ref'].to(self.my_device())
        # uv_map_ref = view_data['uv_map_ref'].to(self.my_device())
        ref_tex = view_data['tex_ref'].to(self.my_device())

        # forward
        if 1: #is_train
            # ref_tex = self.texture_creater(img_ref, uv_map_ref)            
            neural_tex, attention_map = self.att_feature_module(ref_tex)
            att_neural_tex = neural_tex * attention_map

            neural_img = self.texture_no_grad_mapper(uv_map=uv_map, neural_tex=att_neural_tex)
            outputs_img = self.render_module(neural_img)            
        else:
            pass

        atlas = self.get_atalas(ref_tex, neural_tex, att_neural_tex, attention_map)
        outputs = {'img_rs':outputs_img, 'tex_rs':neural_tex[:,0:3,:,:], 'nimg_rs':neural_img[:,0:3,:,:],
                   'atlas':atlas, 'ref_tex':ref_tex}
        return outputs

    def parameters(self):
        # params = []
        # for part in self.part_list:
        #     params.extend(list(part.parameters()))
        # return params
        return list(self.att_feature_module.parameters())+list(self.render_module.parameters())
