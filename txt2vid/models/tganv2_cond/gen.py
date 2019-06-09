import torch
import torch.nn as nn

from txt2vid.models.layers import UpBlock, RenderBlock, Subsample
from txt2vid.models.conv_lstm import ConvLSTMCell, ConvLSTM

class BaseFrameGen(nn.Module):
    def __init__(self, in_channels=1024, out_channels=128):
        super().__init__()

        self.out_channels = 128
        self.up0 = UpBlock(in_channels=in_channels, out_channels=512)
        self.up1 = UpBlock(in_channels=512, out_channels=256)
        self.up2 = UpBlock(in_channels=256, out_channels=out_channels)

    def forward(self, x, cond=None):
        x = self.up0(x)
        x = self.up1(x)
        x = self.up2(x)
        return x

class MultiScaleGen(nn.Module):

    def __init__(self, latent_size=256, width=64, height=64, num_channels=3, additional_blocks=[64, 32, 32], fm_channels=1024, num_frames=16, cond_dim=256, no_lstm=False):
        super().__init__()

        self.subsample = Subsample()

        self.latent_size = latent_size
        # TODO: configure
        self.fm_channels = fm_channels
        self.fm_width = max(1, (width // 64))
        self.fm_height = max(1, (height // 64))

        #self.latent_plane_ch = self.latent_size
        self.latent_plane_ch = self.fm_channels
        self.fm_size = self.fm_width * self.fm_height * self.latent_plane_ch
        self.fc = nn.Linear(latent_size, self.fm_size)

        self.no_lstm = no_lstm
        if no_lstm:
            from txt2vid.models.tgan.temporal_gen import FrameSeedGenerator
            self.frame_seed_gen = FrameSeedGenerator(self.fm_size, self.fm_size)
        else:
            self.clstm = ConvLSTM(input_channels=self.latent_plane_ch, hidden_channels=[self.fm_channels], kernel_size=3, step=num_frames, effective_step=range(num_frames))

        base = BaseFrameGen()
        render_base = RenderBlock(in_channels=base.out_channels, out_channels=num_channels)

        self.render_blocks = [render_base]
        self.abstract_blocks = [base]

        for i, block in enumerate(additional_blocks):
            prev_block = self.abstract_blocks[i].out_channels

            self.abstract_blocks.append(UpBlock(in_channels=prev_block, out_channels=block))
            self.render_blocks.append(RenderBlock(in_channels=block, out_channels=num_channels))

        self.abstract_blocks = nn.ModuleList(self.abstract_blocks)
        self.render_blocks = nn.ModuleList(self.render_blocks)

    def forward(self, x, cond=None, return_abstract_maps=False, output_blocks=None):
        from torch.nn.parallel import data_parallel

        x = self.fc(x)
        if not self.no_lstm:
            x = x.view(x.size(0), self.latent_plane_ch, self.fm_height, self.fm_width)
            x, _ = self.clstm(x)
            num_frames = len(x)
            x = torch.stack(x)
            x = x.permute(1, 0, 2, 3, 4)
        else:
            x = self.frame_seed_gen(x)
            x = x.view(x.size(0), 16, -1, self.fm_height, self.fm_width)
            num_frames = 16

        def channel_first(a):
            return a.permute(0, 2, 1, 3, 4)

        def time_first(a):
            return a.permute(0, 2, 1, 3, 4)

        def split_frames(a):
            return a.contiguous().view(-1, num_frames, a.size(1), a.size(2), a.size(3))

        def merge_frames(a):
            return a.contiguous().view(-1, a.size(2), a.size(3), a.size(4))

        abstract = []
        rendered = []
        x = merge_frames(x)
        for i in range(len(self.render_blocks)):
            abs_block = self.abstract_blocks[i]
            render_block = self.render_blocks[i]

            if i != 0 and self.training:
                # subsample x
                x = split_frames(x)
                x = channel_first(x)
                x, _  = self.subsample(x)
                x = time_first(x)
                x = merge_frames(x)

                num_frames //= 2

            x = data_parallel(abs_block, x)
            abstract.append(x)

            if i == len(self.render_blocks) - 1 or self.training \
              or (output_blocks is not None and i in output_blocks):
                r = data_parallel(render_block, x)
                r = split_frames(r)
                r = time_first(r)
                rendered.append(r)

        if return_abstract_maps:
            return rendered, abstract
        else:
            return rendered


if __name__ == '__main__':
    batch_size = 2
    latent_size = 256
    device = 'cuda:0'
    cond_dim = 50
    gen = MultiScaleGen(latent_size=latent_size, width=192, height=192, num_channels=3, cond_dim=cond_dim).to(device)

    from txt2vid.util.torch.init import init
    print(gen)
    init(gen, 'xavier')
    cond = torch.randn(batch_size, cond_dim).to(device)
    z = torch.randn(batch_size, latent_size).to(device)
    
    #print("Before render")
    rendered = gen(z)
    #print("After render")

    #for i, a in enumerate(abstract):
    #    print('abstract', i, a.size())
    for i, r in enumerate(rendered):
        print('rendered', i, r.size())

    from txt2vid.util.misc import count_params
    print("Num params = %d" % count_params(gen))

    #from torchsummary import summary
    #summary(gen, (256,))
