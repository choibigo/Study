import torch
from . import networks
from torch.optim import lr_scheduler

class BiCycleGANModel():
    def __init__(self,
                lr,
                batch_size,
                input_nc,
                output_nc,
                nz,
                device,
                gan_mode='lsgan',
                beta1=0.5,
                lambda_GAN=1.0,
                lambda_GAN2=1.0,
                use_same_D=False,
                conditional_D=False,
                lambda_z=0.5,
                lambda_kl=0.01,
                lambda_L1=10.0,
                ngf=64,
                ndf=64,
                nef=64,
                netG='unet_256',
                netD='basic_256_multi', 
                netD2='basic_256_multi',
                netE='resnet_256',
                norm='instance',
                nl='relu',
                use_dropout=False,
                init_type='xavier',
                init_gain=0.02,
                where_add='all',
                upsample='basic',
                num_Ds=2):

        assert batch_size % 2 == 0
        
        self.device = device
        self.lambda_GAN = lambda_GAN
        self.lambda_GAN2 = lambda_GAN2
        self.use_same_D = use_same_D
        self.batch_size = batch_size
        self.nz = nz
        self.conditional_D = conditional_D
        self.lambda_z = lambda_z
        self.lambda_kl = lambda_kl
        self.lambda_L1 = lambda_L1
        
        use_D = self.lambda_GAN > 0.0
        use_D2 = self.lambda_GAN2 > 0.0 and not self.use_same_D
        self.model_names = ['G']
        self.netG = networks.define_G(input_nc, output_nc, self.nz, ngf, netG=netG,
                                      norm=norm, nl=nl, use_dropout=use_dropout, init_type=init_type, init_gain=init_gain,
                                      device=self.device, where_add=where_add, upsample=upsample)
        D_output_nc = input_nc + output_nc if conditional_D else output_nc
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, ndf, netD=netD, norm=norm, nl=nl,
                                          init_type=init_type, init_gain=init_gain, num_Ds=num_Ds, device=self.device)
        if use_D2:
            self.model_names += ['D2']
            self.netD2 = networks.define_D(D_output_nc, ndf, netD=netD2, norm=norm, nl=nl,
                                           init_type=init_type, init_gain=init_gain, num_Ds=num_Ds, device=self.device)
        else:
            self.netD2 = None
        self.model_names += ['E']
        self.netE = networks.define_E(output_nc, self.nz, nef, netE=netE, norm=norm, nl=nl,
                                        init_type=init_type, init_gain=init_gain, device=self.device, vaeLike=True)

        self.criterionGAN = networks.GANLoss(gan_mode=gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionZ = torch.nn.L1Loss()
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizers.append(self.optimizer_E)

        if use_D:
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizers.append(self.optimizer_D)
        if use_D2:
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizers.append(self.optimizer_D2)
    
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

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.detach().to(self.device)

    def encode(self, input_image):
        mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    def test(self, z0=None, encode=False):
        with torch.no_grad():
            if encode:
                z0, _ = self.netE(self.real_B)
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.nz)
            self.fake_B = self.netG(self.real_A, z0)
            return self.real_A, self.fake_B, self.real_B

    def forward(self):
        # get real images
        half_size = self.batch_size // 2
        # A1, B1 for encoded; A2, B2 for random
        self.real_A_encoded = self.real_A[0:half_size]
        self.real_B_encoded = self.real_B[0:half_size]
        self.real_A_random = self.real_A[half_size:]
        self.real_B_random = self.real_B[half_size:]
        # get encoded z
        self.z_encoded, self.mu, self.logvar = self.encode(self.real_B_encoded)
        # get random z
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.nz)
        # generate fake_B_encoded
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded)
        # generate fake_B_random
        self.fake_B_random = self.netG(self.real_A_encoded, self.z_random)
        if self.conditional_D:   # tedious conditoinal data
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
            self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
            self.real_data_random = torch.cat([self.real_A_random, self.real_B_random], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded
            self.fake_data_random = self.fake_B_random
            self.real_data_encoded = self.real_B_encoded
            self.real_data_random = self.real_B_random

        # compute z_predict
        if self.lambda_z > 0.0:
            self.mu2, logvar2 = self.netE(self.fake_B_random)  # mu2 is a point estimate

    def backward_D(self, netD, real, fake):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake.detach())
        # real
        pred_real = netD(real)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, self.lambda_GAN)
        if self.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, self.lambda_GAN2)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, self.lambda_GAN2)
        # 2. KL loss
        if self.lambda_kl > 0.0:
            self.loss_kl = torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * (-0.5 * self.lambda_kl)
        else:
            self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl
        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        self.set_requires_grad([self.netD, self.netD2], True)
        # update D1
        if self.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
            if self.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random)
            self.optimizer_D.step()

        if self.lambda_GAN2 > 0.0 and not self.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random)
            self.optimizer_D2.step()

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.lambda_z > 0.0:
            self.loss_z_L1 = self.criterionZ(self.mu2, self.z_random) * self.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad([self.netD, self.netD2], False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()

        # update G alone
        if self.lambda_z > 0.0:
            self.set_requires_grad([self.netE], False)
            self.backward_G_alone()
            self.set_requires_grad([self.netE], True)

        self.optimizer_E.step()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_D()

    