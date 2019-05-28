import torch
import torch.nn as nn

from txt2vid.models.layers import ResidualBlock, DownBlock

class Resnet3D(nn.Module):

    def __init__(self, num_channels=3, mid_ch=64, which_conv=nn.Conv3d, which_pool=nn.AvgPool3d, cond_dim=0):
        super().__init__()
        self.activation = nn.ReLU(True)

        inner_res = nn.Sequential(
            which_conv(num_channels, mid_ch, 3, 1, padding=1, bias=False),
            self.activation,
            which_conv(mid_ch, num_channels, 3, 1, padding=1, bias=False),
            which_pool(1, 2)
        )
        identity_res = which_pool(1, 2)
        self.res_block = ResidualBlock(inner_module=inner_res, identity_map=identity_res)

        self.down0 = DownBlock(in_channels=3, out_channels=128)
        self.down1 = DownBlock(in_channels=128, out_channels=256)
        self.down2 = DownBlock(in_channels=256, out_channels=512)
        self.down3 = DownBlock(in_channels=512, out_channels=1024)
        self.fc = nn.Linear(1024, 1)
    
    def forward(self, x=None, cond=None, xbar=None):
        x = self.res_block(x)

        x = self.activation(x)
        x = self.down0(x)
        x = self.activation(x)
        x = self.down1(x)
        x = self.activation(x)
        x = self.down2(x)
        x = self.activation(x)
        x = self.down3(x)

        x = x.view(x.size(0), x.size(1), -1)
        x = x.mean(dim=2)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    from txt2vid.util.misc import count_params

    x = torch.randn(32, 3, 16, 128, 128).cuda()
    net = Resnet3D().cuda()
    print(net)
    print("Num params = %d" % count_params(net))

    x = net(x)
    print(x.size())
