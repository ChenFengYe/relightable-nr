import torch
from torch import nn
from torch.nn import functional as F

class MultiLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiLoss, self).__init__()
        self.w_atlas = cfg.LOSS.WEIGHT_ATLAS
        self.w_hsv = cfg.LOSS.WEIGHT_HSV        

        self.loss_rgb = torch.tensor([0])
        self.loss_hsv = torch.tensor([0])
        self.loss_atlas = torch.tensor([0])

        # loss 1 - rgb
        self.criterionL1 = torch.nn.L1Loss(reduction='mean')
        # loss 2 - hsv
        self.criterionL1_hsv = HSVLoss()
        # loss 3 - atalas
        self.criterionL1_atlas = TexRGBLoss()

    def forward(self, input, target):
        output_img = input[:,0:3,:,:]
        self.loss_rgb = self.criterionL1(output_img, target)
        self.loss_hsv = self.w_hsv * self.criterionL1_hsv(output_img, target)         

        if input.shape[1] > 3:
            atlas_rgb_layer = input[:,3:6,:,:] 

            self.loss_atlas = self.w_atlas * self.criterionL1_atlas(atlas_rgb_layer, target)         
            loss = self.loss_rgb + self.loss_hsv + self.loss_atlas

        else:
            loss = self.loss_rgb + self.loss_hsv

        return loss

'''
Create a loss to force the first 3 channel of texture to learn rgb values
'''
class TexRGBLoss(nn.Module):
    def __init__(self,):
        super(TexRGBLoss, self).__init__()

        self.L1Loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, input, target):
        loss = self.L1Loss(input, target)
        return loss

class HSVLoss(nn.Module):
    def __init__(self, h=0, s=1, v=0.7, eps=1e-7, threshold_h=0.03, threshold_sv=0.1):
        super(HSVLoss, self).__init__()
        self.hsv = [h, s, v]
        self.loss = nn.L1Loss(reduction='none')
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