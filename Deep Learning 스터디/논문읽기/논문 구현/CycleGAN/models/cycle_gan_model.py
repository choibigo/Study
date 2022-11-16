import torch
import itertools
from . import image_pool
from . import networks
from torch.optim import lr_scheduler

class CycleGANModel():

    def __init__(self,
                lr,
                input_nc,
                output_nc,
                device,
                lambda_identity = 0.5,
                lambda_A = 10.0,
                lambda_B = 10.0,
                beta1 = 0.5,
                pool_size = 50,
                ngf=64,
                ndf=64,
                netG='unet_256',
                netD='basic',
                n_layers_D = 3,
                norm='instance',
                use_dropout=False,
                init_type = 'normal',
                init_gain=0.02,
                gan_mode = 'lsgan',
                ):

        self.device = device
        self.lambda_identity = lambda_identity
        self.optimizers = []
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B

        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']

        self.netG_A = networks.define_G(input_nc, output_nc, ngf, netG, norm,
                                        use_dropout, init_type, init_gain, device=self.device)
        self.netG_B = networks.define_G(output_nc, input_nc, ngf, netG, norm,
                                        use_dropout, init_type, init_gain, device=self.device)

        self.netD_A = networks.define_D(output_nc, ndf, netD,
                                        n_layers_D, norm, init_type, init_gain, device=self.device)
        self.netD_B = networks.define_D(input_nc, ndf, netD,
                                        n_layers_D, norm, init_type, init_gain, device=self.device)

        if self.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
            assert(input_nc == output_nc)
        self.fake_A_pool = image_pool.ImagePool(pool_size)  # create image buffer to store previously generated images
        self.fake_B_pool = image_pool.ImagePool(pool_size)  # create image buffer to store previously generated images
        # define loss functions
        self.criterionGAN = networks.GANLoss(gan_mode).to(self.device)  # define GAN loss.
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=lr, betas=(beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def setup(self, lr_policy, niter=300, niter_decay=30, lr_decay_iters=100):
        self.schedulers = list()
        for optimizer in self.optimizers:
            if lr_policy == 'linear':
                def lambda_rule(epoch):
                    lr_l = 1.0 - max(0, epoch+1 - niter) / float(niter_decay + 1)
                    return lr_l
                scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            elif lr_policy == 'step':
                scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
            elif lr_policy == 'plateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
            elif lr_policy == 'cosine':
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=niter, eta_min=0)
            else:
                return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
            self.schedulers.append(scheduler)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()