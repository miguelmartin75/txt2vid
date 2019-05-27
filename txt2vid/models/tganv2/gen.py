import torch
import torch.nn as nn

from txt2vid.models.layers import UpBlock, RenderBlock, Subsample
from txt2vid.models.conv_lstm import ConvLSTMCell, ConvLSTM

class BaseFrameGen(nn.Module):
    def __init__(self, in_channels=1024, out_size=128):
        super().__init__()

        self.out_size = 128
        self.up0 = UpBlock(in_channels=in_channels, out_channels=512)
        self.up1 = UpBlock(in_channels=512, out_channels=256)
        self.up2 = UpBlock(in_channels=256, out_channels=out_size)

    def forward(self, x):
        x = self.up0(x)
        x = self.up1(x)
        x = self.up2(x)
        return x

class MultiScaleGen(nn.Module):

    def __init__(self, latent_size=128, width=64, height=64, num_channels=3, additional_blocks=[64, 32, 32], fm_channels=1024, num_frames=16):
        super().__init__()

        self.subsample = Subsample()

        self.latent_size = latent_size
        # TODO: configure
        self.fm_channels = fm_channels
        self.fm_width = (width // 64) 
        self.fm_height = (height // 64) 
        self.fm_size = self.fm_width * self.fm_height * latent_size
        self.fc = nn.Linear(latent_size, self.fm_size)

        self.clstm = ConvLSTM(input_channels=self.latent_size, hidden_channels=[self.fm_channels], kernel_size=3, step=num_frames, effective_step=range(num_frames))
        self.base = BaseFrameGen()
        self.render_base = RenderBlock(in_channels=self.base.out_size, out_channels=num_channels)

        self.render_blocks = []
        self.abstract_blocks = []

        for i, block in enumerate(additional_blocks):
            prev_block = self.base.out_size
            if i - 1 >= 0:
                prev_block = additional_blocks[i - 1]

            self.abstract_blocks.append(UpBlock(in_channels=prev_block, out_channels=block))
            self.render_blocks.append(RenderBlock(in_channels=block, out_channels=num_channels))

        self.abstract_blocks = nn.ModuleList(self.abstract_blocks)
        self.render_blocks = nn.ModuleList(self.render_blocks)

    def forward(self, x, only_render_last=False, subsample_bts=None):
        x = self.fc(x)
        x = x.view(x.size(0), self.latent_size, self.fm_height, self.fm_width)
        
        x, _ = self.clstm(x)
        num_frames = len(x)
        x = torch.stack(x)
        x = x.permute(1, 0, 2, 3, 4)

        def split_frames(a):
            return a.contiguous().view(-1, num_frames, a.size(1), a.size(2), a.size(3))

        def channel_first(a):
            return a.permute(0, 2, 1, 3, 4)

        def time_first(a):
            return a.permute(0, 2, 1, 3, 4)

        def merge_frames(a):
            return a.contiguous().view(-1, a.size(2), a.size(3), a.size(4))

        x = merge_frames(x)
        x = self.base(x)

        abstract = [x]
        rendered = []
        if not only_render_last:
            r = split_frames(self.render_base(x))
            r = channel_first(r)
            rendered.append(r)

        for i in range(len(self.render_blocks)):
            x = channel_first(split_frames(x))

            bt = None
            if subsample_bts is not None:
                bt = subsample_bts[i]

            x, _ = self.subsample(x, bt)
            x = time_first(x)
            x = merge_frames(x)
            x = self.abstract_blocks[i](x)
            abstract.append(x)
            if not only_render_last or i == len(self.render_blocks) - 1:
                r = self.render_blocks[i](x)
                r = split_frames(r)
                r = channel_first(r)
                rendered.append(r)

        return abstract, rendered


if __name__ == '__main__':
    batch_size = 64
    latent_size=256
    device = 'cuda:0'
    gen = MultiScaleGen(latent_size=latent_size, width=128, height=128).to(device)
    from txt2vid.util.torch.init import init
    print(gen)
    init(gen, 'xavier')
    z = torch.randn(batch_size, latent_size).to(device)
    
    abstract, rendered = gen(z)

    for i, a in enumerate(abstract):
        print('abstract', i, a.size())
    for i, r in enumerate(rendered):
        print('rendered', i, r.size())

    from txt2vid.util.misc import count_params
    print("Num params = %d" % count_params(gen))
