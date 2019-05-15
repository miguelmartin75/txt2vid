class CondGan(object):
    def __init__(self, gen=None, discrims=None, cond_encoder=None, discrim_names=None):
        assert(gen is not None)
        assert(discrims is not None)
        assert(len(discrims) >= 1)

        if discrim_names is None:
            discrim_names = [ 'discrim-%d' % i for i in range(len(discrims)) ]

        self.gen = gen
        self.discrims = discrims
        self.cond_encoder = cond_encoder
        self.discrim_names = discrim_names

    # override me
    def discrim_forward(self, name=None, discrim=None, real=None, fake=None, real_cond=None, fake_cond=None, loss=None):
        if real_cond is not None and fake_cond is not None:
            # real, correct captions => should predict "REAL"
            real_cc = discrim(real, real_cond)
            # real, incorrect captions => should predict "FAKE"
            real_ic = discrim(real, fake_cond)
            # fake, correct captions => should predict "FAKE"
            fake_cc = discrim(fake, real_cond)

            real_pred = real_cc
            fake_pred = torch.cat((real_ic, fake_cc), dim=0)
        else:
            real_pred = discrim(real, None)
            fake_pred = discrim(fake, None)

        return loss(fake_pred, real_pred)

    def gen_step(self, fake=None, cond=None, retain_graph=False, loss=None):
        self.gen.zero_grad()
        if self.cond_encoder is not None:
            self.cond_encoder.zero_grad()

        pred = []
        for name, discrim in zip(self.discrim_names, self.discrims):
            temp = discrim(fake, cond).mean()
            pred.append(temp)

        pred = torch.stack(pred)
        return loss(pred)

    def discrim_step(self, real=None, fake=None, cond=None, retain_graph=False, loss=None):
        for discrim in self.discrims:
            discrim.zero_grad()

        if self.cond_encoder is not None:
            self.cond_encoder.zero_grad()

        losses = []
        for name, discrim in zip(self.discrim_names, self.discrims):
            real_cond = cond
            fake_cond = cond
            if cond is not None:
                fake_cond = real_cond[gen_perm(real_cond.size(0))]

            l = self.discrim_forward(name=name, discrim=discrim, real_cond=real_cond, fake_cond=fake_cond, real=real, fake=fake, loss=loss)

            losses.append(l)

        return torch.mean(torch.stack(losses))

    def __call__(self, *args, **kwargs):
        return self.gen(*args, **kwargs)

    @property
    def discrims_params(self):
        return [ d.parameters() for d in self.discrims ]

    def save_dict(self):
        res = { 'gen': self.gen.state_dict()}
        if self.cond_encoder is not None:
            res.update({ 'cond': self.cond_encoder.state_dict() })
        for name, discrim in zip(self.discrim_names, self.discrims):
            res.update({name: discrim.state_dict()})

        return res

    def load_from_dict(self, to_load):
        self.gen.load_state_dict(to_load['gen'])
        if 'cond' in to_load:
            assert(self.cond is not None)
            self.cond.load_state_dict(to_load['cond'])
        for name, discrim in zip(self.discrim_names, self.discrims):
            if name in to_load:
                discrim.load_state_dict(to_load[name])

