import torch
import torch.nn as nn
import torch.nn.functional as F

from txt2vid.models.resnet3d import Resnet3D

class MultiScaleDiscrim(nn.Module):

    def __init__(self, discrim_down_blocks=[4, 4, 4, 4], num_channels=3, cond_dim=0, underlying_discrim=Resnet3D, single_discrim=True):
        super().__init__()

        self.sub_discrims = []
        if single_discrim:
            self.single_discrim = underlying_discrim(cond_dim=cond_dim, num_down_blocks=discrim_down_blocks[-1], num_channels=num_channels)
            self.single_discrim = nn.DataParallel(self.single_discrim)
            self.sub_discrims = [ self.single_discrim for i in range(len(discrim_down_blocks)) ]

        else:
            self.sub_discrims = []
            self.single_discrim = None
            for db in discrim_down_blocks:
                d = underlying_discrim(cond_dim=cond_dim, num_down_blocks=db, num_channels=num_channels)
                d = nn.DataParallel(d)
                self.sub_discrims.append(d)

            self.sub_discrims = nn.ModuleList(self.sub_discrims)

    def forward(self, x=None, cond=None, xbar=None, computed_features=None):
        # TODO: cond
        #from torch.nn.parallel import data_parallel
        out = []
        for i, r in enumerate(x):
            cond_i = None
            xbar_i = None
            cf_i = None
            if cond is not None:
                cond_i = cond[i]
            if xbar is not None:
                xbar_i = xbar[i]
            if cf_i is not None:
                cf_i = computed_features[i]

            discrim = self.sub_discrims[i]

            pred = discrim(r, cond=cond_i, xbar=xbar_i, computed_features=cf_i)

            out.append(pred)
        return out

if __name__ == '__main__':
    batch_size = 1
    latent_size = 256
    device = 'cuda:0'
    cond_dim = 100
    from txt2vid.models.tganv2.gen import MultiScaleGen
    gen = MultiScaleGen(latent_size=latent_size, width=256, height=256).to(device)
    z = torch.randn(batch_size, latent_size).to(device)
    z = gen(z)

    discrim = MultiScaleDiscrim().to(device)
    out = discrim(z)

    cond_dim = 100
    cond = [ torch.randn(batch_size, cond_dim).cuda() for _ in range(len(z)) ]
    discrim_cond = MultiScaleDiscrim(cond_dim=cond_dim).to(device)
    out = discrim_cond(z, cond, computed_features=[ o[-1] for o in out ])
    for x in out:
        print(x)
    #print(uncond, cond)


    from txt2vid.util.misc import count_params
    print("Num params = %d" % count_params(discrim))

    print("Gen + Discrim = %d" % (count_params(discrim) + count_params(gen)))
