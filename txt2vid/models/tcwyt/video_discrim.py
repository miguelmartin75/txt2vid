import torch
import torch.nn as nn

class VideoDiscrim(nn.Module):

    def __init__(self, cond_dim=256, mid_ch=64, num_channels=3, negative_slope=0.2, which_conv=nn.Conv3d):
        super().__init__()

        self.f = nn.LeakyReLU(negative_slope, True)

        self.x_map = nn.Sequential(
            which_conv(num_channels, mid_ch, 4, 2, 1, bias=False), # 64
            #nn.BatchNorm3d(mid_ch),
            self.f,

            which_conv(mid_ch, mid_ch*2, 4, 2, 1, bias=False), # 128
            nn.BatchNorm3d(mid_ch*2),
            self.f,

            which_conv(mid_ch*2, mid_ch*4, 4, 2, 1, bias=False), 
            nn.BatchNorm3d(mid_ch*4),
            self.f,

            which_conv(mid_ch*4, mid_ch*8, 4, 2, 1, bias=False), # 256
            nn.BatchNorm3d(mid_ch*8),
            self.f
        )

        if cond_dim > 0:
            self.cond_map = nn.Sequential(
                nn.Linear(cond_dim, cond_dim),
                nn.BatchNorm1d(cond_dim),
                self.f,
            )

            self.pred = nn.Sequential(
                which_conv(mid_ch*8 + cond_dim, 512, 1, 1, 0, bias=False),
                nn.BatchNorm3d(512),
                self.f,
                which_conv(mid_ch*8, 1, (1, 3, 3), 1, 0, bias=False)
            )
        else:
            #self.pred = which_conv(mid_ch*8, 1, 3, 2, 0, bias=False)
            self.pred = which_conv(mid_ch*8, 1, (1, 3, 3), 2, 0, bias=False)

    def forward(self, x=None, cond=None, xbar=None):
        x = self.x_map(x)

        if cond is not None:
            cond = self.cond_map(cond)
            cond = cond.view(cond.size(0), -1, 1, 1, 1)
            cond = cond.expand([-1, -1, x.size(2), x.size(3), x.size(4)])
            x = torch.cat((x, cond), dim=1)

        out = self.pred(x)
        out = out.view(out.size(0), -1)
        return out.mean()

if __name__ == '__main__':
    batch_size = 64
    num_channels = 3
    cond_size = 0
    #frame_size = 48
    frame_size = 64
    num_frames = 16

    vid = torch.randn(batch_size, num_channels, num_frames, frame_size, frame_size)
    if cond_size == 0:
        cond = None
    else:
        cond = torch.randn(batch_size, cond_size)

    discrim = VideoDiscrim(cond_dim=cond_size)
    print(discrim)
    out = discrim(x=vid, cond=cond)

    print("Output video discrim:", out.size())

    from txt2vid.util.misc import count_params
    print("Num params = %d" % count_params(discrim))

