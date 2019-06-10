import torch
import torch.nn as nn

from txt2vid.models.layers import ResidualBlock, DownBlock, Attention3d

class Resnet3D(nn.Module):

    def __init__(self, num_channels=1, mid_ch=64, which_conv=nn.Conv3d, which_pool=nn.AvgPool3d, cond_dim=0, num_down_blocks=4, wide=False, with_attn=True):
        super().__init__()
        self.activation = nn.ReLU(inplace=False)

        res_path = nn.Sequential(
            which_conv(num_channels, mid_ch, 3, 1, padding=1),
            self.activation,
            which_conv(mid_ch, mid_ch, 3, 1, padding=1),
            which_pool((1, 2, 2), 2)
        )
        skip_conn = nn.Sequential(which_pool((1, 2, 2), 2), which_conv(num_channels, mid_ch, 1))
        self.res_block = ResidualBlock(inner_module=res_path, identity_map=skip_conn)

        down = []
        in_ch = mid_ch
        out_ch = 128
        for i in range(num_down_blocks):
            down.append(DownBlock(in_channels=in_ch, out_channels=out_ch,  which_conv=which_conv, wide=wide))
            if i == 0 and with_attn:
                down.append(Attention3d(out_ch, which_conv=which_conv))

            in_ch = out_ch
            out_ch *= 2

        self.down = nn.ModuleList(down)
        self.fc_uncond = nn.Linear(in_ch, 1)
        if cond_dim > 0:
            self.fc = nn.Linear(in_ch + cond_dim, 1)

    
    def forward(self, x=None, cond=None, xbar=None, computed_features=None):
        uncond = None
        if computed_features is not None:
            x = computed_features
        else:
            x = self.res_block(x)

            for down in self.down:
                x = down(x)

            x = torch.sum(x, [2, 3, 4])
            computed_features = x
            uncond = self.fc_uncond(x)

        if cond is not None:
            x_plus_cond = torch.cat((x, cond), dim=1)
            cond = self.fc(x_plus_cond)
            return uncond, cond, computed_features

        return uncond, None, computed_features

if __name__ == '__main__':
    from txt2vid.util.misc import count_params

    batch_size = 5
    x = torch.randn(batch_size, 3, 16, 128, 128).cuda()
    net = Resnet3D(num_channels=3).cuda()
    print(net)
    print("Num params = %d" % count_params(net))

    z, _, _ = net(x)

    cond_dim = 100
    net_cond = Resnet3D(num_channels=3, cond_dim=cond_dim).cuda()
    print(net_cond)
    print("Num params = %d" % count_params(net_cond))

    cond = torch.randn(batch_size, cond_dim).cuda()
    z, cond, _ = net_cond(x, cond)
    print("c, cond=")
    print(z.size(), cond.size())

    print(torch.cuda.max_memory_allocated() / 10**9)
    print(torch.cuda.max_memory_cached() / 10**9)
