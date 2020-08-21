import torch

from lib.models import network
from lib.engine import loss
from lib.utils import util

from lib.models.feature_pair_net import FeaturePairNet
from lib.engine.loss import MultiLoss

from collections import OrderedDict

class Pix2PixModel(torch.nn.Module):
    def __init__(self, cfg, isTrain=True):
        super(Pix2PixModel, self).__init__()

        self.cfg = cfg
        self.device = torch.device('cuda:{}'.format(cfg.GPUS[0]))
        self.isTrain = cfg
        self.model_names = []
        self.visual_names = []
        self.optimizers = []

        self.loss_names = ['G_GAN', 'G_Multi', 'G_per', 'G_atlas', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']        
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # Generator
        self.netG = FeaturePairNet(cfg=cfg)

        # Discriminator
        if self.isTrain: 
            self.netD = network.define_D(input_nc = cfg.MODEL.NET_D.INPUT_CHANNELS,
                                            ndf = cfg.MODEL.NET_D.NDF,
                                            netD = cfg.MODEL.NET_D.ARCH,
                                            n_layers_D = cfg.MODEL.NET_D.N_LAYERS_D,
                                            norm = cfg.MODEL.NET_D.NORM,
                                            init_type = cfg.MODEL.NET_D.INIT_TYPE,
                                            init_gain = cfg.MODEL.NET_D.INIT_GAIN,
                                            gpu_ids = cfg.GPUS)
        if self.isTrain:
            # define loss functions
            self.criterionGAN = loss.GANLoss(cfg.MODEL.GAN.MODE).to(self.device)
            self.criterionMulti = MultiLoss(cfg)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=cfg.TRAIN.LR, betas=(cfg.TRAIN.LR_FACTOR, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=cfg.TRAIN.LR, betas=(cfg.TRAIN.LR_FACTOR, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    # to-do 0821
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']            

        self.real_A = input['img_ref'].to(self.device)
        self.real_B = input['img'].to(self.device)
        
    def devices(self):
        devices = ({param.device for param in self.parameters()} |
                {buf.device for buf in self.buffers()})
        if len(devices) != 1:
            raise RuntimeError('Cannot determine device: {} different devices found'
                            .format(len(devices)))
        return next(iter(devices))

    def get_atalas(self):
        return self.netG.get_atalas()

    # def get_atalas(self, ref_tex, neural_tex, att_neural_tex):
    #     atalas = torch.cat((ref_tex.permute(0,3,1,2)[:, 0:3, :, :], 
    #                         neural_tex[:, 0:3, :, :],
    #                         att_neural_tex[:, 0:3, :, :],), dim=0)
    #     return atalas.clone().detach().cpu()

    def forward(self, input):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_out = self.netG(input)  # G(A)

        if self.isTrain:
            alpha_map = input['mask'][:,None,:,:].cuda()
            self.fake_out = self.fake_out * alpha_map
            self.real_B = self.real_B * alpha_map
                
        self.fake_B = self.fake_out[:, 0:3, : ,:]
        self.fake_tex = self.fake_out[:, 3:6, : ,:]

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        ##################################################################
        # to-do 0821 give a better design
        self.loss_G_Multi = self.criterionMulti(self.fake_out, self.real_B)
        # for vis
        self.loss_G_per = self.criterionMulti.loss_rgb
        self.loss_G_atlas = self.criterionMulti.loss_atlas
        ##################################################################
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_Multi
        self.loss_G.backward()

    def optimize_parameters(self, view_data):
        self.set_input(view_data)

        self.forward(view_data)          # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights        

    def setup(self, cfg):
        if self.isTrain:
            self.schedulers = [network.get_scheduler(optimizer, cfg) for optimizer in self.optimizers]
        # to-do 0821
        # if not self.isTrain or cfg.continue_train:
        #     load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
        #     self.load_networks(load_suffix)
        self.print_networks(cfg.VERBOSE)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
        
    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def optimizer_state_dict(self):
        state_dicts = []
        for optimizer in self.optimizers:
            state_dicts.append(optimizer.state_dict())
        return state_dicts

    def load_optimizer_state_dict(self, state_dicts):
        for optimizer, state_dict in zip(self.optimizers, state_dicts):
            optimizer.load_state_dict(state_dict)        

# from base_model.py
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.cfg.TRAIN.LR_MODE == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad        