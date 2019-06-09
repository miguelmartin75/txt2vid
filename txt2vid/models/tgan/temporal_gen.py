import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

# https://github.com/proceduralia/tgan-pytorch/blob/master/models/temporal_generator.py

class FrameSeedGenerator(nn.Module):
    #Generate exactly 16 latent vectors starting from 1
    def __init__(self, z_slow_dim, z_fast_dim):
        super().__init__()
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim

        self.dc0 = nn.ConvTranspose1d(z_slow_dim, 512, 1, 1, 0)
        self.dc1 = nn.ConvTranspose1d(512, 256, 4, 2, 1)
        self.dc2 = nn.ConvTranspose1d(256, 128, 4, 2, 1)
        self.dc3 = nn.ConvTranspose1d(128, 128, 4, 2, 1)
        self.dc4 = nn.ConvTranspose1d(128, z_fast_dim, 4, 2, 1)
        self.bn0 = nn.BatchNorm1d(512)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, z_slow):
        h = z_slow.view(z_slow.size(0),-1, 1)
        print("h=", h.size())
        h = F.relu(self.bn0(self.dc0(h)))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        z_fast = F.tanh(self.dc4(h))
        return z_fast

if __name__ == '__main__':
    INPUT_SIZE = 256
    OUTPUT_SIZE = 256
    fs = FrameSeedGenerator(INPUT_SIZE, OUTPUT_SIZE)
    z = torch.randn(16, INPUT_SIZE)
    all_frames = fs(z)
    print(all_frames.size())
