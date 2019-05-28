import torch

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

        # real, correct captions => should predict "REAL"
        if real is not None:
            real_pred = discrim(x=real, cond=real_cond, xbar=real_mapping)

        if real_cond is not None and fake_cond is not None:
            # real, incorrect captions => should predict "FAKE"
            real_ic = torch.tensor([], device=real_cond.device)
            if real is not None and fake_cond is not None:
                real_ic = discrim(x=real, cond=fake_cond, xbar=real_mapping)

            # fake, correct captions => should predict "FAKE"
            fake_cc = torch.tensor([], device=real_cond.device)
            if fake is not None and real_cond is not None:
                fake_cc = discrim(x=fake, cond=real_cond, xbar=fake_mapping)

            # TODO
            fake_pred = torch.cat((real_ic, fake_cc), dim=-1)
        else:
            if fake is not None:
                fake_pred = discrim(x=fake, cond=None, xbar=fake_mapping)

        l = None
        if loss is not None and fake_pred is not None and real_pred is not None:
            losses = [ loss(fake=f, real=r) for f, r in zip(fake_pred, real_pred) ]
            losses = torch.stack(losses)
            l = losses.mean()

            # TODO potentially move this elsewhere
            if gp_lambda > 0:
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
            f = discrim(x=fake, cond=cond, xbar=fake_mapping)
            temp = []
            for ff, rr in zip(f, r):
                temp.append(loss(fake=ff, real=rr))
            losses.append(torch.stack(temp).mean())

        return self._discrim_weighted_sum(torch.stack(losses))
    

    def all_discrim_forward(self, fake=None, real=None, cond=None, loss=None, gp_lambda=-1):
        losses = []
        real_pred = []
        fake_pred = []

        real_mapping = self._map_input(real)
        fake_mapping = self._map_input(fake)

        for name, discrim in zip(self.discrim_names, self.discrims):
            real_cond = cond
            fake_cond = cond
            if cond is not None:
                perm = gen_perm(real_cond[0].size(0))
                fake_cond = [ r[perm[0:r.size(0)]] for r in real_cond ]

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
            self.cond_encoder.load_state_dict(to_load['cond'])

        if 'sample_mapping' in to_load:
            assert(self.sample_mapping is not None)
            self.sample_mapping.load_state_dict(to_load['sample_mapping'])

        for name, discrim in zip(self.discrim_names, self.discrims):
            if name in to_load:
                discrim.load_state_dict(to_load[name])
