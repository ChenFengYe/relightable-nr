import torch

from lib.engine.perceptual_loss import PerceptualLoss
from lib.engine.contextual_loss import ContextualLoss

from torch import nn
from torch.nn import functional as F

import torchvision.models.vgg as vgg

import random

# refer to https://chsasank.github.io/vision/_modules/torchvision/models/vgg.html
# can change it to torchvision.models.vgg.vgg19 
def vgg19(vgg_path, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    cfg_E = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    model = vgg.VGG(vgg.make_layers(cfg_E), **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
        vgg_model = torch.load(vgg_path, map_location='cpu')
        model.load_state_dict(vgg_model)
    return model

class MultiLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiLoss, self).__init__()

        # weights
        self.w_per = cfg.LOSS.WEIGHT_PERCEPTUAL
        self.w_atlas = cfg.LOSS.WEIGHT_ATLAS
        self.w_atlas_ref = cfg.LOSS.WEIGHT_ATLAS_REF
        self.w_atlas_unity = cfg.LOSS.WEIGHT_ATLAS_UNIFY
        self.w_hsv = cfg.LOSS.WEIGHT_HSV                        
        self.w_views = cfg.LOSS.WEIGHT_VIEWS

        # loss
        self.loss = torch.tensor([0.0])
        self.loss_rgb = torch.tensor([0.0])
        # self.loss_hsv = torch.tensor([0.0])

        self.loss_atlas = torch.tensor([0.0])
        self.loss_atlas_ref = torch.tensor([0.0])
        self.loss_atlas_tar = torch.tensor([0.0])
        self.loss_atlas_unify = torch.tensor([0.0])

        # loss functions
        # create vgg for loss
        vgg_pretrained_features = vgg19(cfg.MODEL.VGG_PATH, pretrained=True).features
        # vgg_pretrained_features = vgg.vgg19(pretrained=True).features

        # loss - image
        self.Perceptual_L1 = PerceptualLoss(cfg, vgg_pretrained_features)
        self.Perceptual_L1 = self.Perceptual_L1.cuda()
        # loss - atalas
        # self.Contextural_cosine = ContextualLoss(use_vgg=True, vgg_model=vgg_pretrained_features, vgg_layer='relu5_4',loss_type='cosine')
        # self.Contextural_cosine = self.Contextural_cosine.cuda()
        # self.AtlasLoss = AtlasLoss(self.w_atlas_ref, self.w_atlas_unity, self.Contextural_cosine)

        self.AtlasLoss = AtlasLoss(self.w_atlas_ref, self.w_atlas_unity)
        # loss - hsv
        # self.HSV_L1 = HSVLoss()
        # loss - hsv


    def forward(self, input, target):                
        # to-do-0910 set loss for rendered view
        data_type = target['data_type']
        if data_type == 'viewed':
            rs = input['rs']
            gt = target['gt']
            mask_gt = gt[:,3:4,:,:].clone().repeat(1,3,1,1)
            rs[:,0:3,:,:] = rs[:,0:3,:,:].clone() * mask_gt
            gt[:,0:3,:,:] = gt[:,0:3,:,:].clone() * mask_gt

            # loss 1 persepetual
            self.loss_rgb = self.w_per * self.Perceptual_L1(rs, gt)

            # loss hsv
            # self.loss_hsv = self.w_hsv * self.HSV_L1(img_rs, target)         

            # all
            self.loss_atlas = self.w_atlas * self.AtlasLoss(input, target)

            # summary
            self.loss_atlas_ref = self.AtlasLoss.loss_atlas_ref
            self.loss_atlas_tar = self.AtlasLoss.loss_atlas_tar
            self.loss_atlas_unify = self.AtlasLoss.loss_atlas_unify
            self.loss = self.loss_rgb + self.loss_atlas

        elif data_type == 'rendered':
            batch_size = input['rs'].shape[0]
            idxs_ref = list(range(0, batch_size))
            idxs_ref.insert(0, idxs_ref.pop())  # shift 1 element

            rs = input['rs']            
            self.loss_views = F.l1_loss(rs, rs[idxs_ref,...].clone().detach()) + F.l1_loss(rs[idxs_ref,...], rs.clone().detach())
            self.loss_views = self.w_views * self.loss_views

        return self.loss
    
    # def loss_list(self): 
    #     loss_list ={'Loss':self.loss,
    #                 'rgb':self.loss_rgb,
    #                 'hsv':self.loss_hsv,
    #                 'atlas':self.loss_atlas,
    #                 'atlas_ref':self.loss_atlas_ref,
    #                 'atlas_tar':self.loss_atlas_tar}
    #     return loss_list

class AtlasLoss(nn.Module):
    def __init__(self, w_ref, w_unify, loss_fun = F.l1_loss):
        super(AtlasLoss, self).__init__()
        self.loss_fun = loss_fun
        self.w_unify = w_unify
        self.w_ref = w_ref
        self.loss_atlas_ref = torch.tensor([0.0])
        self.loss_atlas_tar = torch.tensor([0.0])
        self.loss_atlas_unify = torch.tensor([0.0])

    def forward(self, input, target):

        # loss 2 Atlas
        atlas_rgb = input['tex_rs'] 
        gt_tex_ref = target['tex_ref']
        gt_tex_tar = target['tex']

        mask_ref = (gt_tex_ref != 0).int().float()
        mask_tar = (gt_tex_tar != 0).int().float()

        # loss_L1(tex1, tex2)+loss_L1(tex2, tex1)
        
        self.loss_atlas_ref = self.w_ref * self.loss_fun(atlas_rgb*mask_ref, gt_tex_ref)
        self.loss_atlas_tar = self.loss_fun(atlas_rgb*mask_tar, gt_tex_tar)

        # atalas unify
        batch_size = atlas_rgb.shape[0]
        batch_idxs = list(range(0,batch_size))
        random.shuffle(batch_idxs)
        self.loss_atlas_unify = self.w_unify*(self.loss_fun(atlas_rgb, atlas_rgb[batch_idxs, ...].clone().detach()) + \
                                self.loss_fun(atlas_rgb[batch_idxs, ...], atlas_rgb.clone().detach()))

        self.loss_atlas = self.loss_atlas_tar + self.loss_atlas_ref + self.loss_atlas_unify
        
        return self.loss_atlas

class HSVLoss(nn.Module):
    def __init__(self, h=0, s=1, v=0.7, eps=1e-7, threshold_h=0.03, threshold_sv=0.1):
        super(HSVLoss, self).__init__()
        self.hsv = [h, s, v]
        # self.loss = nn.L1Loss(reduction='none')
        self.loss = torch.nn.L1Loss(reduction='mean')
        
        self.eps = eps

        # since Hue is a circle (where value 0 is equal to value 1 that are both "red"), 
        # we need a threshold to prevent the gradient explod effect
        # the smaller the threshold, the optimal hue can to more close to target hue
        self.threshold_h = threshold_h
        # since Hue and (Value and Satur) are conflict when generated image' hue is not the target Hue, 
        # we have to condition V to prevent it from interfering the Hue loss
        # the larger the threshold, the ealier to activate V loss
        self.threshold_sv = threshold_sv

    def get_hsv(self, im):
        img = im * 0.5 + 0.5
        hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)

        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6

        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + self.eps )
        saturation[ img.max(1)[0]==0 ] = 0

        value = img.max(1)[0]
        return hue, saturation, value

    def get_rgb_from_hsv(self):
        C = self.hsv[2] * self.hsv[1]
        X = C * ( 1 - abs( (self.hsv[0]*6)%2 - 1 ) )
        m = self.hsv[2] - C

        if self.hsv[0] < 1/6:
            R_hat, G_hat, B_hat = C, X, 0
        elif self.hsv[0] < 2/6:
            R_hat, G_hat, B_hat = X, C, 0
        elif self.hsv[0] < 3/6:
            R_hat, G_hat, B_hat = 0, C, X
        elif self.hsv[0] < 4/6:
            R_hat, G_hat, B_hat = 0, X, C
        elif self.hsv[0] < 5/6:
            R_hat, G_hat, B_hat = X, 0, C
        elif self.hsv[0] <= 6/6:
            R_hat, G_hat, B_hat = C, 0, X

        R, G, B = (R_hat+m), (G_hat+m), (B_hat+m)
        
        return R, G, B
    
    
    def forward(self, input, target):
        h, s, v = self.get_hsv(input)
        target_h, target_s, target_v = self.get_hsv(target)

        # target_h = torch.Tensor(h.shape).fill_(t_h).to(input.device).type_as(h)
        # target_s = torch.Tensor(s.shape).fill_(t_h).to(input.device).type_as(s)
        # target_v = torch.Tensor(v.shape).fill_(t_h).to(input.device).type_as(v)

        loss_h = self.loss(h, target_h)
        loss_h[loss_h<self.threshold_h] = 0.0
        loss_h = loss_h.mean()

        if loss_h < self.threshold_h*3:
            loss_h = torch.Tensor([0]).to(input.device)
        
        loss_s = self.loss(s, target_s).mean()
        if loss_h.item() > self.threshold_sv:   
            loss_s = torch.Tensor([0]).to(input.device)

        loss_v = self.loss(v, target_v).mean()
        if loss_h.item() > self.threshold_sv:   
            loss_v = torch.Tensor([0]).to(input.device)

        return loss_h + 4e-1*loss_s + 4e-1*loss_v

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            # target_tensor = self.get_target_tensor(prediction, target_is_real)
            # loss = self.loss(prediction, target_tensor)

            if isinstance(prediction, list):
                loss = 0
                for pred in prediction:
                    target_tensor = self.get_target_tensor(pred[-1], target_is_real)
                    loss += self.loss(pred[-1], target_tensor)
                return loss
            else:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                return self.loss(prediction, target_tensor)

        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
