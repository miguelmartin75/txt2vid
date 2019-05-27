import torch
import torch.nn as nn
import torch.nn.functional as F

from txt2vid.models.resnet3d import Resnet3D

class MultiScaleDiscrim(nn.Module):

    def __init__(self, num_discrims=4, num_channels=3):
        super().__init__()

        self.discrims = []
        for i in range(num_discrims):
            self.discrims.append(Resnet3D())
        self.discrims = nn.ModuleList(self.discrims)
        self.combine = torch.sum
        self.classify = F.sigmoid

    def forward(self, x):
        abstract = x[0]
        rendered = x[1]

        out = 0
        for i, r in enumerate(reversed(rendered)):
            discrim = self.discrims[len(self.discrims) - 1 - i]
            pred = discrim(r)
            out += self.combine(pred)
        out = self.classify(out)
        return out

if __name__ == '__main__':
    batch_size = 64
    latent_size=256
    device = 'cuda:0'
    from txt2vid.models.tganv2.gen import MultiScaleGen
    gen = MultiScaleGen(latent_size=latent_size, width=128, height=128).to(device)
    z = torch.randn(batch_size, latent_size).to(device)
    z = gen(z)

    discrim = MultiScaleDiscrim().to(device)
    print(discrim)
    out = discrim(z)

    print(out)
    print(out.size())

    from txt2vid.util.misc import count_params
    print("Num params = %d" % count_params(discrim))

    print("Gen + Discrim = %d" % (count_params(discrim) + count_params(gen)))
