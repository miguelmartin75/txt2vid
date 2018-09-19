import torch
import torch.nn as nn
import torch.nn.init as init

class Discrim(nn.Module):
    def __init__(self, num_filters=64, num_channels=3, leaky=0.1):
        super().__init__()

        self.seq = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_channels, num_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_filters) x 32 x 32
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_filters*2) x 16 x 16
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_filters*4) x 8 x 8
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_filters*8) x 4 x 4
            nn.Conv2d(num_filters * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            #nn.Linear(, 1)
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.seq(input)

class Generator(nn.Module):
    def __init__(self, latent_size=100, num_filters=64, num_channels=3):
        super().__init__()

        self.latent_size = latent_size
        
        self.seq = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_size, num_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(True),
            # state size. (num_filters*8) x 4 x 4
            nn.ConvTranspose2d(num_filters* 8, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(True),
            # state size. (num_filters*4) x 8 x 8
            nn.ConvTranspose2d(num_filters* 4, num_filters* 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(True),
            # state size. (num_filters*2) x 16 x 16
            nn.ConvTranspose2d(num_filters* 2, num_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(num_filters, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.seq(input)

def weights_init(layer):
    name = layer.__class__.__name__
    if 'Conv' in name or 'Linear' in name:
        init.xavier_normal_(layer.weight.data)
        if 'Linear' in name:
            layer.bias.data.fill_(0.0)
    elif 'BatchNorm' in name:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0.0)

