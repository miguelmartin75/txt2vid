import torch
import torch.nn as nn

class LabelledGanLoss(object):

    def __init__(self, real_label=None, fake_label=None, underlying_loss=None):
        assert(real_label is not None)
        assert(fake_label is not None)
        assert(underlying_loss is not None)
        self.loss = underlying_loss
        self.fake_label = real_label
        self.real_label = fake_label

    def _compute_loss(self, x, label):
        labels = torch.full(x.size(), label, device=x.device)
        return self.loss(x, labels)

    def discrim_loss(self, fake=None, real=None):
        fake = self._compute_loss(fake, self.fake_label)
        real = self._compute_loss(real, self.real_label)
        return fake + real

    def gen_loss(self, fake=None, real=None):
        return self._compute_loss(fake, self.real_label)

class VanillaGanLoss(LabelledGanLoss):

    def __init__(self, bce_loss=True):
        loss = nn.BCEWithLogitsLoss() if bce_loss else nn.CrossEntropyLoss()
        super().__init__(underlying_loss=loss, real_label=1, fake_label=0)

class HingeGanLoss(LabelledGanLoss):

    def __init__(self, margin=2.0):
        self.loss = nn.HingeEmbeddingLoss(margin=margin)
        super().__init__(underlying_loss=self.loss, real_label=1, fake_label=-1)

class WassersteinGanLoss(object):

    def __init__(self):
        pass

    def discrim_loss(self, fake=None, real=None):
        return -(real.mean() - fake.mean())

    def gen_loss(self, fake=None, real=None):
        return -fake.mean()

# for seperate G and D loss (not sure why you'd do this but eh)
class MixedGanLoss(object):
    def __init__(self, g_loss=None, d_loss=None):
        self.g_loss = g_loss
        self.d_loss = d_loss

    def discrim_loss(self, fake=None, real=None):
        return self.d_loss.discrim_loss(fake=fake, real=real)

    def gen_loss(self, fake=None, real=None):
        return self.g_loss.gen_loss(fake=fake, real=real)
