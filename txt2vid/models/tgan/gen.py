import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

# https://github.com/proceduralia/tgan-pytorch/blob/master/models/temporal_generator.py

class VideoFrameGenerator(nn.Module):
    def __init__(self, z_slow_dim, z_fast_dim, out_channels=3, bottom_width=4, conv_ch=512):
        super().__init__()
        self.ch = conv_ch
        self.bottom_width = bottom_width
        slow_mid_dim = bottom_width * bottom_width * conv_ch //2
        fast_mid_dim = bottom_width * bottom_width * conv_ch //2

        self.l0s = nn.Linear(z_slow_dim, slow_mid_dim)
        self.l0f = nn.Linear(z_fast_dim, fast_mid_dim)
        self.dc1 = nn.ConvTranspose2d(conv_ch, conv_ch // 2, 4, 2, 1)
        self.dc2 = nn.ConvTranspose2d(conv_ch // 2, conv_ch // 4, 4, 2, 1)
        self.dc3 = nn.ConvTranspose2d(conv_ch // 4, conv_ch // 8, 4, 2, 1)
        self.dc4 = nn.ConvTranspose2d(conv_ch // 8, conv_ch // 16, 4, 2, 1)
        self.dc5 = nn.ConvTranspose2d(conv_ch // 16, out_channels, 3, 1, 1)

        self.bn0s = nn.BatchNorm1d(slow_mid_dim)
        self.bn0f = nn.BatchNorm1d(fast_mid_dim)
        self.bn1 = nn.BatchNorm2d(conv_ch // 2)
        self.bn2 = nn.BatchNorm2d(conv_ch // 4)
        self.bn3 = nn.BatchNorm2d(conv_ch // 8)
        self.bn4 = nn.BatchNorm2d(conv_ch // 16)

    def forward(self, z_slow, z_fast):
        n = z_slow.size(0)
        h_slow = (F.relu(self.bn0s(self.l0s(z_slow))).view(n, self.ch // 2, self.bottom_width, self.bottom_width))
        h_fast = (F.relu(self.bn0f(self.l0f(z_fast)))).view(n, self.ch // 2, self.bottom_width, self.bottom_width)
        h = torch.cat((h_slow, h_fast), 1)
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.relu(self.bn4(self.dc4(h)))
        x = F.tanh(self.dc5(h))
        return x

class Gen(nn.Module):
    def __init__(self, z_slow_dim=256, z_fast_dim=256, cond_dim=0, out_channels=3, bottom_width=4, conv_ch=512):
        from txt2vid.models.tgan.temporal_gen import FrameSeedGenerator
        super().__init__()
        self.z_slow_plus_cond_dim = z_slow_dim + cond_dim
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim
        self.out_channels = out_channels
        self._fsgen = FrameSeedGenerator(self.z_slow_plus_cond_dim, z_fast_dim)
        self._vgen = VideoFrameGenerator(self.z_slow_plus_cond_dim, z_fast_dim, out_channels, bottom_width,conv_ch)

    def forward(self, z_slow, cond=None):
        if cond is not None:
            z_slow = torch.cat((z_slow, cond), dim=-1)

        z_fast = self._fsgen(z_slow)
        print("z_fast=", z_fast.size())

        B, n_z_fast, n_frames = z_fast.size()
        z_fast = z_fast.permute(0, 2, 1).contiguous().view(B * n_frames, n_z_fast) #squash time dimension in batch dimension

        B, n_z_slow = z_slow.size()
        z_slow = z_slow.unsqueeze(1).repeat(1, n_frames, 1)
        z_slow = z_slow.contiguous().view(B * n_frames, n_z_slow)

        print(z_slow.size())
        print(z_fast.size())
        out = self._vgen(z_slow, z_fast)
        out = out.view(B, n_frames, self.out_channels, 64, 64)
        return out.permute(0, 2, 1, 3, 4)

    @property
    def latent_size(self):
        return self.z_slow_dim

if __name__ == "__main__":
    # The number of frames in a video is fixed at 16
    batch_size = 64
    num_channels = 3
    z_size = 100
    cond_size = 256

    gen = Gen(z_slow_dim=100, cond_dim=256)
    z = torch.randn(batch_size, z_size)
    cond = torch.randn(batch_size, cond_size)
    out = gen(z, cond=cond)
    print("Output video generator:", out.size())

    from txt2vid.util.misc import count_params
    print("Num params = %d" % count_params(gen))
