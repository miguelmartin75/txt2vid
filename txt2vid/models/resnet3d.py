import torch
import torch.nn as nn

from txt2vid.models.layers import ResidualBlock, DownBlock

class Resnet3D(nn.Module):

    def __init__(self, num_channels=3, mid_ch=64, which_conv=nn.Conv3d, which_pool=nn.AvgPool3d, cond_dim=0, num_down_blocks=4):
        super().__init__()
        self.activation = nn.ReLU(inplace=False)

        res_path = nn.Sequential(
            which_conv(num_channels, mid_ch, 3, 1, padding=1, bias=False),
            self.activation,
            which_conv(mid_ch, mid_ch, 3, 1, padding=1, bias=False),
            which_pool(1, 2)
        )
        skip_conn = nn.Sequential(which_pool(1, 2), which_conv(num_channels, mid_ch, 1))
        self.res_block = ResidualBlock(inner_module=res_path, identity_map=skip_conn)

        self.down = []
        in_ch = mid_ch
        out_ch = 128
        for i in range(num_down_blocks):
            self.down.append(DownBlock(in_channels=in_ch, out_channels=out_ch,  which_conv=which_conv))
            in_ch = out_ch
            out_ch *= 2

        self.down = nn.ModuleList(self.down)
        self.fc = nn.Linear(in_ch, 1)
    
    def forward(self, x=None, cond=None, xbar=None):
        x = self.res_block(x)

        for down in self.down:
            x = down(x)

        x = torch.sum(x, [2, 3, 4])
        x = self.fc(x)
        return x

if __name__ == '__main__':
    from txt2vid.util.misc import count_params

    x = torch.randn(1, 3, 16, 32, 32).cuda()
    net = Resnet3D().cuda()
    print(net)
    print("Num params = %d" % count_params(net))

    x = net(x)
    print(net)
    print(x.size())

    #from torchsummary import summary
    #summary(net, (3, 16, 128, 128))

    #net2d = Resnet3D(which_conv=nn.Conv2d, which_pool=nn.AvgPool2d).cuda()
    #print("Num params 2d  = %d" % count_params(net2d))
    #summary(net2d, (3, 128, 128))
