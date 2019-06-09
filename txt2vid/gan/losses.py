import torch
import torch.nn as nn

def get_labels_for(x, label):
    return torch.full(x.size(), label, device=x.device)

# for seperate G and D loss (not sure why you'd do this but eh)
class MixedGanLoss(object):
    def __init__(self, g_loss=None, d_loss=None):
        self.g_loss = g_loss
        self.d_loss = d_loss

    def discrim_loss(self, fake=None, real=None):
        return self.d_loss.discrim_loss(fake=fake, real=real)

    def gen_loss(self, fake=None, real=None):
        return self.g_loss.gen_loss(fake=fake, real=real)

class LabelledGanLoss(object):

    def __init__(self, real_label=None, fake_label=None, underlying_loss=None):
        assert(real_label is not None)
        assert(fake_label is not None)
        assert(underlying_loss is not None)
        self.loss = underlying_loss
        self.fake_label = real_label
        self.real_label = fake_label

    def _compute_loss(self, x, label):
        labels = get_labels_for(x, label)
        return self.loss(x, labels)

    def discrim_loss(self, fake=None, real=None):
        fake = self._compute_loss(fake, self.fake_label)
        real = self._compute_loss(real, self.real_label)
        return fake + real

    def gen_loss(self, fake=None, real=None):
        return self._compute_loss(fake, self.real_label)

class VanillaGanLoss(LabelledGanLoss):

    # TODO: change default back to mean?
    def __init__(self, bce_loss=True, reduction='mean'):
        loss = nn.BCEWithLogitsLoss(reduction=reduction) if bce_loss else nn.CrossEntropyLoss(reduction=reduction)
        super().__init__(underlying_loss=loss, real_label=1, fake_label=0)

class HingeGanLoss(LabelledGanLoss):

    def __init__(self, margin=2.0):
        self.loss = nn.HingeEmbeddingLoss(margin=margin)
        super().__init__(underlying_loss=self.loss, real_label=1, fake_label=-1)


class WassersteinGanLoss(object):

    def __init__(self):
        pass

    def discrim_loss(self, fake=None, real=None):
        #print("real=", real.size())
        #print("fake=", fake.size())
        return -(real.mean() - fake.mean())

    def gen_loss(self, fake=None, real=None):
        #print("real=", real.size())
        #print("fake=", fake.size())
        return -fake.mean()

# Relativistic Standard GAN (RSGAN)
# https://github.com/AlexiaJM/RelativisticGAN
# https://arxiv.org/pdf/1807.00734.pdf
# performs best for 1 discrim/1 gen step (speed pls) => with gradient penalty (GP)
class RSGANLoss(object):
     
    def __init__(self, bce_loss=True):
        self.loss = nn.BCEWithLogitsLoss() if bce_loss else nn.CrossEntropyLoss()

    def discrim_loss(self, fake=None, real=None):
        y = get_labels_for(fake, 1)
        return self.loss(real - fake, y)

    def gen_loss(self, fake=None, real=None):
        y = get_labels_for(fake, 1)
        return self.loss(fake - real, y)

class RaSGANLoss(object):
     
    def __init__(self, bce_loss=True):
        self.loss = nn.BCEWithLogitsLoss() if bce_loss else nn.CrossEntropyLoss()
        self.fake_label = 0
        self.real_label = 1

    def discrim_loss(self, fake=None, real=None):
        fake_labels = get_labels_for(real, self.fake_labels)
        real_labels = get_labels_for(fake, self.real_labels)
        
        loss = 0.0
        loss += self.loss(real - fake.mean(), real_labels) 
        loss += self.loss(fake - real.mean(), fake_labels)
        return loss / 2

    def gen_loss(self, fake=None, real=None):
        fake_labels = get_labels_for(fake, self.fake_labels)
        real_labels = get_labels_for(real, self.real_labels)
        
        loss = 0.0
        loss += self.loss(real - fake.mean(), fake_labels) 
        loss += self.loss(fake - real.mean(), real_labels)
        return loss / 2

# Relativistic average LSGAN (RaLSGAN)
class RaLSGANLoss(object):
    def __init__(self):
        pass

    def discrim_loss(self, fake=None, real=None):
        y_fake = get_labels_for(fake, 1)
        y_real = get_labels_for(real, 1)

        loss = 0.0
        loss += torch.mean((real - torch.mean(fake) - y_real) ** 2) 
        loss += torch.mean((fake - torch.mean(real) + y_fake) ** 2)
        return loss / 2

    def gen_loss(self, fake=None, real=None):
        y_fake = get_labels_for(fake, 1)
        y_real = get_labels_for(real, 1)

        loss = 0.0
        loss += torch.mean((real - torch.mean(fake) + y_fake) ** 2) 
        loss += torch.mean((fake - torch.mean(real) - y_real) ** 2)
        return loss / 2

def _gradient_penalty(discrim, real_x=None, real_xbar=None, fake_x=None, fake_xbar=None, real_cond=None, fake_cond=None, zero_center=False, combine=torch.mean):
    batch_size = real_x.size(0)

    # video
    if real_x.dim() == 4:
        alpha = torch.rand(batch_size, 1, 1, 1, requires_grad=True)
    else:
        assert(real_x.dim() == 5)
        alpha = torch.rand(batch_size, 1, 1, 1, 1, requires_grad=True)

    alpha_x = alpha.expand_as(real_x).to(real_x.device)
    interpolate_x = alpha_x * real_x + ((1 - alpha_x) * fake_x)

    interpolate_xbar = None
    if real_xbar is not None and fake_xbar is not None:
        alpha_xbar = alpha.expand_as(real_xbar).to(real_xbar.device)
        interpolate_xbar = alpha_xbar * real_xbar + ((1 - alpha_xbar) * fake_xbar)

    interpolate_cond = None
    if real_cond is not None and fake_cond is not None:
        alpha = alpha.squeeze()
        if alpha.dim() == 0:
            alpha = alpha.unsqueeze(0)
        alpha = alpha.unsqueeze(1)

        alpha_cond = alpha.expand_as(real_cond).to(real_cond.device)
        interpolate_cond = alpha_cond * real_cond + ((1 - alpha_cond) * fake_cond)

    inputs = [ interpolate_x ]
    if interpolate_cond is not None:
        inputs.append(interpolate_cond)
    if interpolate_xbar is not None:
        inputs.append(interpolate_xbar)

    uncond_output, cond_output, _ = discrim(x=interpolate_x, cond=interpolate_cond, xbar=interpolate_xbar)

    outputs = [ uncond_output ]
    if cond_output is not None:
        outputs.append(cond_output)

    grad_outputs = [ torch.ones(g.size(), device=g.device) for g in outputs ]

    from torch.autograd import grad
    gradients = grad(outputs=outputs, inputs=inputs, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)

    if zero_center:
        gp = combine(gradients.norm(2, dim=1) ** 2)
    else:
        gp = combine((gradients.norm(2, dim=1) - 1) ** 2)
    return gp

def gradient_penalty(discrim, real_x=None, real_xbar=None, fake_x=None, fake_xbar=None, real_cond=None, fake_cond=None):
    if hasattr(discrim, 'sub_discrims'):
        gradients = []
        for i in range(len(real_x)):
            if real_cond is None:
                rx, rxbar, rcond = real_x[i], None, None
                fx, fxbar, fcond = fake_x[i], None, None
            else:
                if real_xbar is None:
                    real_xbar = [ None ] * len(real_x)
                    fake_xbar = [ None ] * len(real_x)

                rx, rxbar, rcond = real_x[i], real_xbar[i], real_cond[i]
                fx, fxbar, fcond = fake_x[i], fake_xbar[i], fake_cond[i]

            gp = _gradient_penalty(discrim.sub_discrims[i], real_x=rx, real_xbar=rxbar, real_cond=rcond, fake_x=fx, fake_xbar=fxbar, fake_cond=fcond, zero_center=True, combine=torch.sum)
            #gp = _gradient_penalty(discrim.sub_discrims[i], real_x=rx, real_xbar=rxbar, real_cond=rcond, fake_x=fx, fake_xbar=fxbar, fake_cond=fcond, zero_center=False, combine=torch.sum)
            gradients.append(gp)

        return torch.stack(gradients).sum()
    else:
        return _gradient_penalty(discrim, real_x=real_x, real_xbar=real_xbar, fake_x=fake_x, fake_xbar=fake_xbar, real_cond=real_cond, fake_cond=fake_cond)


    #assert(fake_x.size(0) == real_x.size(0))

