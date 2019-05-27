import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.apply(tag_residual)


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
    def __init__(self, in_channels=128, out_channels=None, which_bn=nn.BatchNorm2d, which_conv=nn.Conv2d, which_unpool=nn.ConvTranspose2d):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        main_path = nn.Sequential(
            which_bn(in_channels),
            nn.ReLU(True),
            which_unpool(in_channels, in_channels, 2, 2),
            which_conv(in_channels, in_channels, 3, 1, padding=1, bias=False),
            which_bn(in_channels),
            nn.ReLU(True),
            which_conv(in_channels, out_channels, 3, 1, padding=1, bias=False)
        )

        # note: not using unpool layer
        identity_map = which_unpool(in_channels, in_channels, 2, 2)

        if out_channels != in_channels:
            identity_map = nn.Sequential(identity_map, which_unpool(in_channels, out_channels, 1))

        self.main = ResidualBlock(inner_module=main_path, identity_map=identity_map)

    def forward(self, x):
        return self.main(x)

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

    def __init__(self, in_channels=3, out_channels=None, which_conv=nn.Conv3d):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        main_path = nn.Sequential(
            which_conv(in_channels, in_channels, 1, bias=False),
            nn.ReLU(True),
            which_conv(in_channels, out_channels, 1, bias=False),
            DownSample()
        )
        identity_map = nn.Sequential(
            which_conv(in_channels, out_channels, 1, bias=False),
            DownSample()
        )
        self.main = ResidualBlock(inner_module=main_path, identity_map=identity_map)

    def forward(self, x):
        return self.main(x)

class RenderBlock(nn.Module):

    def __init__(self, in_channels=128, out_channels=3, which_bn=nn.BatchNorm2d, which_conv=nn.Conv2d):
        super().__init__()
        self.bn = which_bn(in_channels)
        self.activation = nn.ReLU(True)
        self.conv = which_conv(in_channels, out_channels, 3, 1, 1, bias=False)
        self.final = nn.Tanh()

    def forward(self, x):
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.final(x)
        return x

if __name__ == '__main__':
    x = torch.randn(2, 3, 5, 5)
    subsample = SubsampleRect(width=3, height=2)
    print(subsample)
    print(x)
    x = subsample(x)
    print(x.size())
    print(x)

    print()
    print()

    up = UpBlock(in_channels=3, out_channels=10)
    print(up)
    x = up(x)
    print(x.size())

    print()
    print()

    x = torch.randn(10, 3, 1, 4, 4)
    ds = DownSample()
    print('before ds=', x.size())
    x = ds(x)
    print('after ds=', x.size())

    print()
    print()

    x = torch.randn(10, 3, 16, 100, 100)
    down = DownBlock(in_channels=3, out_channels=128)
    print(down)
    x = down(x)
    print(x.size())

    print()
    print()

    x = torch.randn(64, 3, 16, 128, 128)
    ss = Subsample()
    print(ss)
    print('before ss=', x.size())
    x = ss(x)
    print('after ss=', x.size())
