import torch
import numpy as np

from txt2vid.util.misc import gen_perm
from txt2vid.gan.losses import gradient_penalty

class CondGan(object):
    def __init__(self, gen=None, discrims=None, cond_encoder=None, discrim_names=None, sample_mapping=None, discrim_lambdas=None):
        assert(gen is not None)
        assert(discrims is not None)
        assert(len(discrims) >= 1)

        if discrim_names is None:
            discrim_names = [ 'discrim-%d' % i for i in range(len(discrims)) ]

        self.gen = gen
        self.discrims = discrims
        self.sample_mapping = sample_mapping
        self.cond_encoder = cond_encoder
        self.discrim_names = discrim_names
        self.discrim_lambdas = discrim_lambdas

    def _map_input(self, x):
        return self.sample_mapping(x) if self.sample_mapping is not None and x is not None else None
    
    def _discrim_weighted_sum(self, losses):
        if self.discrim_lambdas is None:
            return torch.mean(losses)

        lambdas = torch.tensor(self.discrim_lambdas, device=losses.device)
        return torch.sum(lambdas * losses)

    # override me
    def discrim_forward(self, name=None, discrim=None, real=None, real_mapping=None, fake=None, fake_mapping=None, real_cond=None, fake_cond=None, loss=None, gp_lambda=-1):
        fake_pred = None
        real_pred = None
        l = None

        # real, correct captions => should predict "REAL"
        if real_cond is not None and fake_cond is not None:
            # overlap in these real_cc and real_ic (in terms of computation)
            real_cc = discrim(x=real, cond=real_cond, xbar=real_mapping)
            real_pred = real_cc
            if loss is not None:
                real_ic = discrim(x=real, 
                                  cond=fake_cond, 
                                  xbar=real_mapping, 
                                  computed_features=[ temp[-1] for temp in real_cc ])
                fake_cc = discrim(x=fake, cond=real_cond, xbar=fake_mapping)

                loss_uncond = [ loss(fake=f[0], real=r[0]) for f, r in zip(fake_cc, real_cc) ]
                loss_c1 = [ loss(fake=f[1], real=r[1]) for f, r in zip(fake_cc, real_cc) ]
                loss_c2 = [ loss(fake=f[1], real=r[1]) for f, r in zip(real_ic, real_cc) ]

                loss_uncond = torch.stack(loss_uncond).mean()
                loss_c1 = torch.stack(loss_c1)
                loss_c2 = torch.stack(loss_c2)

                loss_cond = (loss_c1.mean() + loss_c2.mean()) / 2

                l = (loss_uncond + loss_cond) / 2.0
        else:
            if real is not None:
                real_pred = discrim(x=real, cond=None, xbar=real_mapping)
                real_pred = [ r[0] for r in real_pred ]

            if fake is not None:
                fake_pred = discrim(x=fake, cond=None, xbar=fake_mapping)
                fake_pred = [ f[0] for f in fake_pred ]

            if loss is not None and fake_pred is not None and real_pred is not None:
                losses = [ loss(fake=f, real=r) for f, r in zip(fake_pred, real_pred) ]
                losses = torch.stack(losses)
                l = losses.mean()

        # TODO potentially move this elsewhere
        if l is not None and gp_lambda > 0:
            gp = gradient_penalty(discrim, 
                                  real_x=real, 
                                  real_xbar=real_mapping,
                                  fake_x=fake,
                                  fake_xbar=fake_mapping,
                                  real_cond=real_cond,
                                  fake_cond=fake_cond)
            l += gp_lambda * gp
                
        return l, fake_pred, real_pred


    def gen_step(self, fake=None, real_pred=None, cond=None, loss=None):
        self.gen.zero_grad()

        if self.cond_encoder is not None:
            self.cond_encoder.zero_grad()

        # TODO: detach?
        fake_mapping = self._map_input(fake)

        losses = []
        for r, name, discrim in zip(real_pred, self.discrim_names, self.discrims):
            fake_cc = discrim(x=fake, cond=cond, xbar=fake_mapping)
            if cond is None:
                temp = []
                for ff, rr in zip(fake_cc, r):
                    temp.append(loss(fake=ff, real=rr))
                losses.append(torch.stack(temp).mean())
            else:
                loss_uncond = [ loss(fake=ff[0], real=rr[0]) for ff, rr in zip(fake_cc, r) ]
                loss_cond = [ loss(fake=ff[1], real=rr[1]) for ff, rr in zip(fake_cc, r) ]

                loss_uncond = torch.stack(loss_uncond).mean()
                loss_cond = torch.stack(loss_cond).mean()

                temp = (loss_cond + loss_uncond) / 2.0
                losses.append(temp)


        return self._discrim_weighted_sum(torch.stack(losses))
    

    def all_discrim_forward(self, fake=None, real=None, cond=None, loss=None, gp_lambda=-1):
        losses = []
        real_pred = []
        fake_pred = []

        real_mapping = self._map_input(real)
        fake_mapping = self._map_input(fake)

        for name, discrim in zip(self.discrim_names, self.discrims):
            real_cond = cond
            fake_cond = None
            if cond is not None:
                fake_cond_0 = real_cond[0][gen_perm(real_cond[0].size(0))]
                fake_cond = [ fake_cond_0[0:r.size(0)] for r in real_cond ]

            l, f, r = self.discrim_forward(name=name, 
                                           discrim=discrim,

                                           real=real, 
                                           real_cond=real_cond, 
                                           real_mapping=real_mapping,

                                           fake=fake, 
                                           fake_cond=fake_cond, 
                                           fake_mapping=fake_mapping, 

                                           loss=loss,
                                           gp_lambda=gp_lambda)
            
            losses.append(l)
            fake_pred.append(f)
            real_pred.append(r)

        return losses, fake_pred, real_pred

    def discrim_step(self, real=None, fake=None, cond=None, loss=None, gp_lambda=-1):
        for discrim in self.discrims:
            discrim.zero_grad()

        if self.cond_encoder is not None:
            self.cond_encoder.zero_grad()

        losses, _, _ = self.all_discrim_forward(real=real, fake=fake, cond=cond, loss=loss, gp_lambda=gp_lambda)
        return self._discrim_weighted_sum(torch.stack(losses))
    
    def count_params(self):
        from txt2vid.util.misc import count_params
        d = np.sum([ count_params(discrim) for discrim in self.discrims ])
        g = count_params(self.gen)
        c = 0
        if self.cond_encoder is not None:
            c = count_params(self.cond_encoder)
        s = 0
        if self.sample_mapping is not None:
            s = count_params(self.sample_mapping)
        return d + g + c + s


    def __call__(self, *args, **kwargs):
        return self.gen(*args, **kwargs)

    @property
    def discrims_params(self):
        return [ d.parameters() for d in self.discrims ]

    def save_dict(self):
        res = { 'gen': self.gen.state_dict()}
        if self.cond_encoder is not None:
            res.update({ 'cond': self.cond_encoder.state_dict() })
        if self.sample_mapping is not None:
            res.update({ 'sample_mapping': self.sample_mapping.state_dict() })

        for name, discrim in zip(self.discrim_names, self.discrims):
            res.update({name: discrim.state_dict()})

        return res

    def load_from_dict(self, to_load):
        self.gen.load_state_dict(to_load['gen'])

        if 'cond' in to_load:
            assert(self.cond_encoder is not None)
            #keys_to_remove = []
            #for key in to_load['cond']:
            #    if 'decoder' in key:
            #        keys_to_remove.append(key)
            #for key in keys_to_remove:
            #    to_load['cond'].pop(key, None)
            self.cond_encoder.load_state_dict(to_load['cond'])

        if 'sample_mapping' in to_load:
            assert(self.sample_mapping is not None)
            self.sample_mapping.load_state_dict(to_load['sample_mapping'])

        for name, discrim in zip(self.discrim_names, self.discrims):
            if name in to_load:
                discrim.load_state_dict(to_load[name])
