import torch
import torch.nn as nn

class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True,  stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)

    def forward(self, input):
        output = self.conv(input)
        return output

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        return output

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = input
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        output = self.conv(output)
        return output

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size,input_height,output_width,output_depth) for t_t in spl]
        output = torch.stack(stacks,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size,output_height,output_width,output_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=64):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            #TODO: ????
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size = kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = UpSampleConv(input_dim, output_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(output_dim, output_dim, kernel_size = kernel_size)
        elif resample == None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size = kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output


class ReLULayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(ReLULayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.linear(input)
        output = self.relu(output)
        return output

class Discrim(nn.Module):

    def __init__(self, cond_dim=256, mid_ch=64, num_channels=3, negative_slope=0.2, which_conv=nn.Conv2d, bn=nn.BatchNorm2d):
        super().__init__()

        self.dim = 64

        #https://github.com/jalola/improved-wgan-pytorch/blob/master/models/wgan.py
        self.conv1 = MyConvo2d(3, self.dim, 3, he_init = False)
        #self.conv1 = MyConvo2d(1, self.dim, 3, he_init = False)
        self.rb1 = ResidualBlock(self.dim, 2*self.dim, 3, resample = 'down', hw=self.dim)
        self.rb2 = ResidualBlock(2*self.dim, 4*self.dim, 3, resample = 'down', hw=int(self.dim/2))
        self.rb3 = ResidualBlock(4*self.dim, 8*self.dim, 3, resample = 'down', hw=int(self.dim/4))
        self.rb4 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'down', hw=int(self.dim/8))
        self.ln1 = nn.Linear(4*4*8*self.dim, 1)


    def forward(self, x=None, cond=None, xbar=None):
        output = x
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = output.view(-1, 4*4*8*self.dim)
        output = self.ln1(output)
        output = output.view(-1)
        return output

class Gen(nn.Module):
    def __init__(self, cond_dim=0):
        super().__init__()
        self.dim = 64
        self.latent_size = 128

        self.ln1 = nn.Linear(self.latent_size, 4*4*8*self.dim)
        self.rb1 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'up')
        self.rb2 = ResidualBlock(8*self.dim, 4*self.dim, 3, resample = 'up')
        self.rb3 = ResidualBlock(4*self.dim, 2*self.dim, 3, resample = 'up')
        self.rb4 = ResidualBlock(2*self.dim, 1*self.dim, 3, resample = 'up')
        self.bn  = nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1*self.dim, 3, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
  
    def forward(self, x=None,cond=None, xbar=None):
        output = self.ln1(x)
        output = output.view(-1, 8*self.dim, 4, 4)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        #output = output.view(-1, 3, 64, 64)
        return output

#class Discrim(nn.Module):
#
#    def __init__(self, cond_dim=256, mid_ch=64, num_channels=3, negative_slope=0.2, which_conv=nn.Conv2d, bn=nn.BatchNorm2d):
#        super().__init__()
#
#        self.f = nn.LeakyReLU(negative_slope, True)
#
#        self.x_map = nn.Sequential(
#                which_conv(num_channels, mid_ch, 4, 2, 1, bias=False), # 64
#                self.f,
#
#                which_conv(mid_ch, mid_ch*2, 4, 2, 1, bias=False), # 128
#                bn(mid_ch*2),
#                self.f,
#
#                which_conv(mid_ch*2, mid_ch*4, 4, 2, 1, bias=False), 
#                bn(mid_ch*4),
#                self.f,
#
#                which_conv(mid_ch*4, mid_ch*8, 4, 2, 1, bias=False), # 256
#                bn(mid_ch*8),
#                self.f
#                )
#
#        if cond_dim > 0:
#            self.cond_map = nn.Sequential(
#                    nn.Linear(cond_dim, cond_dim),
#                    bn(cond_dim),
#                    self.f,
#                    )
#
#            self.pred = nn.Sequential(
#                    which_conv(mid_ch*8 + cond_dim, 512, 1, 1, 0, bias=False),
#                    bn(512),
#                    self.f,
#                    which_conv(mid_ch*8, 1, (1, 3, 3), 1, 0, bias=False)
#                    )
#        else:
#            self.pred = which_conv(mid_ch*8, 1, 3, 2, 0, bias=False)
#            #self.pred = which_conv(mid_ch*8, 1, (1, 3, 3), 2, 0, bias=False)
#
#    def forward(self, x=None, cond=None, xbar=None):
#        x = self.x_map(x)
#
#        if cond is not None:
#            cond = self.cond_map(cond)
#            cond = cond.view(cond.size(0), -1, 1, 1, 1)
#            cond = cond.expand([-1, -1, x.size(2), x.size(3), x.size(4)])
#            x = torch.cat((x, cond), dim=1)
#
#        out = self.pred(x)
#        out = out.view(out.size(0), -1)
#        return out.mean()
#
#class Gen(nn.Module):
#    def __init__(self, cond_dim=0):
#        super().__init__()
#        nz = 100
#        ngf = 64
#        nc = 3
#        self.main = nn.Sequential(
#            # input is Z, going into a convolution
#            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
#            nn.BatchNorm2d(ngf * 8),
#            nn.ReLU(True),
#            # state size. (ngf*8) x 4 x 4
#            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf * 4),
#            nn.ReLU(True),
#            # state size. (ngf*4) x 8 x 8
#            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf * 2),
#            nn.ReLU(True),
#            # state size. (ngf*2) x 16 x 16
#            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf),
#            nn.ReLU(True),
#            # state size. (ngf) x 32 x 32
#            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
#            nn.Tanh()
#            # state size. (nc) x 64 x 64
#        )to_save_real
#
#    @property
#    def latent_size(self):to_save_real
#        return 100
#
#    def forward(self, x=None,cond=None, xbar=None):
#        x = x.view(x.size(0), -1, 1, 1)
#        out = self.main(x)
#        return out
