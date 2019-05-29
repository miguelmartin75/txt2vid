import torch
import torch.nn as nn
import torch.nn.functional as F

from txt2vid.models.resnet3d import Resnet3D

class MultiScaleDiscrim(nn.Module):

    def __init__(self, discrim_down_blocks=[4, 4, 4, 4], num_channels=3, cond_dim=0, underlying_discrim=Resnet3D):
        super().__init__()

        self.sub_discrims = []
        for db in discrim_down_blocks:
            self.sub_discrims.append(underlying_discrim(cond_dim=cond_dim, num_down_blocks=db))
        self.sub_discrims = nn.ModuleList(self.sub_discrims)

    def forward(self, x=None, cond=None, xbar=None):
        # TODO: cond
        from torch.nn.parallel import data_parallel
        out = []
        for i, r in enumerate(x):
            discrim = self.sub_discrims[i]
            pred = data_parallel(discrim, r)
            out.append(pred)
        return out

if __name__ == '__main__':
    batch_size = 1
    latent_size = 256
    device = 'cuda:0'
    from txt2vid.models.tganv2.gen import MultiScaleGen
    gen = MultiScaleGen(latent_size=latent_size, width=256, height=256).to(device)
    z = torch.randn(batch_size, latent_size).to(device)
    z = gen(z)

    discrim = MultiScaleDiscrim().to(device)
    #print(z)
    #print("z=", z[0].size())
    #print(discrim)
    out = discrim(z)

    #print(a[::2].size())
    #for x in out:
    #    a[::2] += x
    #    a = a[::2]
        #print(x.size())
    #print(out)
    #print(out.size())

    from txt2vid.util.misc import count_params
    print("Num params = %d" % count_params(discrim))

    print("Gen + Discrim = %d" % (count_params(discrim) + count_params(gen)))
