import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P

# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
# taken from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
class Attention(nn.Module):
    def __init__(self, ch, which_conv=nn.Conv2d, name='attention'):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

# https://github.com/facebookresearch/video-nonlocal-net/blob/master/lib/models/nonlocal_helper.py
class Attention3d(nn.Module):
    def __init__(self, ch, which_conv=nn.Conv3d, name='attention'):
        super(Attention3d, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        batch_size = x.size(0)

        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool3d(self.phi(x), [1, 2, 2])
        g = F.max_pool3d(self.g(x), [1, 2, 2])    
        # Perform reshapes
        theta = theta.view(batch_size, self. ch // 8, -1)
        phi = phi.view(batch_size, self. ch // 8, -1)
        g = g.view(batch_size, self. ch // 2, -1)

        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(batch_size, -1, x.shape[2], x.shape[3], x.shape[4]))
        return self.gamma * o + x

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class ResidualBlock(nn.Module):

    def __init__(self, inner_module=None, identity_map=None):
        super().__init__()
        self.inner_module = inner_module
        self.identity_map = identity_map
        if identity_map is None:
            self.identity_map = Identity()

        def tag_residual(x):
            x.is_residual = True

        # apply sqrt(2) factor to residual path
        self.inner_module.apply(tag_residual)


    def forward(self, x):
        identity = self.identity_map(x)
        x = self.inner_module(x)
        return identity + x

class Subsample(nn.Module):

    def __init__(self, sn=2, st=2):
        super().__init__()
        self.sn = sn
        self.st = st

    # BxTxCxWxH
    def forward(self, x, bt=None):
        if bt is None:
            bt = torch.randint(self.st, (1,))

        x = x[::self.sn, :, bt::self.st]
        return x, bt

class SubsampleRect(nn.Module):

    def __init__(self, width=None, height=None, depth=None, subsample_batch=False):
        super().__init__()
        self.width = torch.tensor(width)
        self.height = width
        if height is not None:
            self.height = torch.tensor(height)
        self.depth = None
        if depth is not None:
            self.depth = torch.tensor(depth)
        self.subsample_batch = subsample_batch

    # BxCxWxH or BxTxCxWxH
    def forward(self, x):
        # note: torch.randint generates [low, high)
        if self.depth is None:
            w, h = x.size(2), x.size(3)
            max_width = int(w - self.width) + 1
            max_height = int(h - self.height) + 1
            px = int(torch.randint(max_width, (1,)))
            py = int(torch.randint(max_height, (1,)))
            return x[:, :, py:py+self.height, px:px+self.width]
        else:
            c_idx = 0 if self.subsample_batch else 2
            c, w, h = x.size(c_idx), x.size(3), x.size(4)
            max_width = int(w - self.width) + 1
            max_height = int(h - self.height) + 1
            max_channels = int(c - self.depth) + 1
            px = int(torch.randint(max_width, (1,)))
            py = int(torch.randint(max_height, (1,)))
            pz = int(torch.randint(max_channels, (1,)))
            if self.subsample_batch:
                return x[pz:pz+self.depth, :, :, py:py+self.height, px:px+self.width]
            else:
                return x[:, :, pz:pz+self.depth, py:py+self.height, px:px+self.width]


# CxWxH => Cx(2W)x(2H)
class UpBlock(nn.Module):

    # TODO: unpool layer
    def __init__(self, in_channels=128, out_channels=None, which_bn=nn.BatchNorm2d, which_conv=nn.Conv2d, upsample_instead=True, which_unpool=nn.ConvTranspose2d, wide=False, with_non_local=False):
        super().__init__()

        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        mid_ch = self.in_channels if wide else self.out_channels

        # not supporting the other way around for now
        assert(upsample_instead)

        unpool1 = nn.Upsample(scale_factor=2)

        main_path = nn.Sequential(
            which_bn(in_channels),
            nn.ReLU(inplace=False),
            unpool1,
            which_conv(in_channels, mid_ch, 3, 1, padding=1),
            which_bn(mid_ch),
            nn.ReLU(inplace=False),
            which_conv(mid_ch, out_channels, 3, 1, padding=1)
        )

        identity_map = nn.Upsample(scale_factor=2)

        if in_channels != out_channels:
            identity_map = nn.Sequential(identity_map, which_conv(in_channels, out_channels, 1))

        self.main = ResidualBlock(inner_module=main_path, identity_map=identity_map)

        self.with_non_local = with_non_local
        if with_non_local:
            self.attn = Attention(out_channels)

    def forward(self, x):
        x = self.main(x)
        if self.with_non_local:
            x = self.attn(x)
        return x

class DownSample(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        kernel = [1, 1, 1]
        stride = [1, 1, 1]
        padding = [0, 0, 0]

        for i in range(3):
            size = x.size(i + 2)
            if size == 1:
                continue

            kernel[i] = 2
            stride[i] = 2
            if size % 2 != 0:
                padding[i] = 1

        return F.avg_pool3d(x, kernel_size=kernel, stride=stride, padding=padding)

class DownBlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=None, which_conv=nn.Conv3d, wide=True):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        mid_ch = out_channels if wide else in_channels

        main_path = nn.Sequential(
            nn.ReLU(inplace=False),
            which_conv(in_channels, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            which_conv(mid_ch, out_channels, kernel_size=3, padding=1),
            DownSample()
        )
        identity_map = nn.Sequential(
            which_conv(in_channels, out_channels, 1),
            DownSample()
        )
        self.main = ResidualBlock(inner_module=main_path, identity_map=identity_map)

    def forward(self, x):
        return self.main(x)

class RenderBlock(nn.Module):

    def __init__(self, in_channels=128, out_channels=3, which_bn=nn.BatchNorm2d, which_conv=nn.Conv2d):
        super().__init__()
        self.bn = which_bn(in_channels)
        self.activation = nn.ReLU()
        self.conv = which_conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.final = nn.Tanh()

    def forward(self, x):
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.final(x)
        return x

if __name__ == '__main__':
    #x = torch.randn(2, 3, 5, 5)
    #subsample = SubsampleRect(width=3, height=2)
    #print(subsample)
    #print(x)
    #x = subsample(x)
    #print(x.size())
    #print(x)

    #print()
    #print()

    #up = UpBlock(in_channels=3, out_channels=10)
    #print(up)
    #x = up(x)
    #print(x.size())

    #print()
    #print()

    #x = torch.randn(10, 3, 1, 4, 4)
    #ds = DownSample()
    #print('before ds=', x.size())
    #x = ds(x)
    #print('after ds=', x.size())

    #print()
    #print()

    x = torch.randn(10, 3, 16, 100, 100).cuda()
    down = DownBlock(in_channels=3, out_channels=128).cuda()
    from txt2vid.util.torch.init import init

    init(down, 'xavier')

    print(down)
    x = down(x)
    print(x.size())

    print("num params=", sum(p.numel() for p in down.parameters()))
    #print()
    #print()

    #x = torch.randn(64, 3, 16, 128, 128)
    #ss = Subsample()
    #print(ss)
    #print('before ss=', x.size())
    #x = ss(x)
    #print('after ss=', x.size())

