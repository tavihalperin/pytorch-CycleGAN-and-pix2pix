import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn.utils import spectral_norm #import SpectralNorm

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


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
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


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
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=True,
             init_type='normal', init_gain=0.02, gpu_ids=[], res=True):
    norm_layer = get_norm_layer(norm_type=norm)

    net = DilatedResnetGeneratorG(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                  use_dropout=use_dropout, padding_type='reflect', res=res)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_F(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=True,
             init_type='normal', init_gain=0.02, gpu_ids=[], res=True):
    norm_layer = get_norm_layer(norm_type=norm)

    net = DilatedResnetGeneratorF(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                  use_dropout=use_dropout, padding_type='reflect', res=res)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='instance', init_type='normal',
             init_gain=0.02, gpu_ids=[], with_specnorm=False):
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

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    norm_layer = get_norm_layer(norm_type=norm)

    net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer,
                              with_specnorm=with_specnorm)

    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
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
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class DilatedResnetGeneratorF(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=128, norm_layer=nn.BatchNorm2d, use_dropout=True,
                 padding_type='reflect', res=True):

        super(DilatedResnetGeneratorF, self).__init__()

        use_bias = isinstance(norm_layer(0), Identity)
        self.no_res = not res
        self.layer1 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        self.layer1 = nn.Sequential(*self.layer1)
        self.model = DilatedResnetGenerator(ngf, output_nc, ngf, norm_layer, use_dropout,
                                            padding_type)

    def forward(self, input):
        x = self.layer1(input)
        if self.no_res:
            return self.model(x)
        return input + self.model(x)

class DilatedResnetGeneratorG(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=128, norm_layer=nn.BatchNorm2d, use_dropout=True,
                 padding_type='reflect', res=True):

        super(DilatedResnetGeneratorG, self).__init__()

        use_bias = isinstance(norm_layer, Identity)

        self.no_res = not res
        self.layer1a = [nn.ReflectionPad2d(3),
                        nn.Conv2d(input_nc, ngf//2, kernel_size=7, padding=0, bias=use_bias),
                        norm_layer(ngf//2),
                        nn.ReLU(True)]
        self.layer1b = [nn.ReflectionPad2d(3),
                        nn.Conv2d(input_nc, ngf//2, kernel_size=7, padding=0, bias=use_bias),
                        norm_layer(ngf//2),
                        nn.ReLU(True)]
        self.layer1a = nn.Sequential(*self.layer1a)
        self.layer1b = nn.Sequential(*self.layer1b)
        self.model = DilatedResnetGenerator(ngf, output_nc, ngf, norm_layer, use_dropout,
                                            padding_type)

    def forward(self, source, reference):
        x = self.layer1a(source)
        y = self.layer1b(reference)
        x = torch.cat([x,y], dim=1)
        if self.no_res:
            return self.model(x)
        return source + self.model(x)


class DilatedResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=128, norm_layer=nn.BatchNorm2d, use_dropout=True,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        super(DilatedResnetGenerator, self).__init__()
        use_bias = isinstance(norm_layer, Identity)
        model = []
        dilations = [1, 2, 4]
        for dilation in dilations:
            model += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, dilation=dilation)]

        model += [nn.ReflectionPad2d(2)]
        model += [nn.Conv2d(ngf, ngf//2, kernel_size=3, padding=0, dilation=2)]
        model += [norm_layer(ngf//2)]
        model += [nn.ReLU(True)]

        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(ngf//2, output_nc, kernel_size=3, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, dilation):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout,
                                                use_bias, dilation)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, dilation):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            padding_layer = nn.ReflectionPad2d(dilation)
        elif padding_type == 'replicate':
            padding_layer = nn.ReplicationPad2d(dilation)
        elif padding_type == 'zero':
            padding_layer = None
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if padding_layer:
            conv_block += [padding_layer]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias,
                                 dilation=dilation),
                       norm_layer(dim),
                       nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        if padding_layer:
            conv_block += [padding_layer]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias,
                                 dilation=dilation),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d,
                 with_specnorm=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d

        use_bias = False
        kw = 4
        padw = 0#1
        def conv_layer(input_nc, ndf, kernel_size, stride, padding, bias=True, with_specnorm=False):
            conv = nn.Conv2d(input_nc, ndf, kernel_size, stride=stride, padding=padding, bias=bias)
            if with_specnorm:
                conv = spectral_norm(conv)
                # conv = SpectralNorm()(conv)
            return conv

        sequence = [
            conv_layer(input_nc, ndf, kw, stride=2, padding=padw, with_specnorm=with_specnorm),
            # nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        # nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                conv_layer(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=padw,
                           bias=use_bias, with_specnorm=with_specnorm),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            conv_layer(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=padw,
                       bias=use_bias, with_specnorm=with_specnorm),
            # nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [conv_layer(ndf * nf_mult, 1, 3, stride=1, padding=padw, with_specnorm=False)]
                # nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
#         print(input.shape)
#         print(self.model(input).shape)
#         exit()
        return self.model(input)

