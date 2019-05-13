import random
import argparse
import numpy as np

import sys
import gc

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torchvision import transforms

import txt2vid.model as model
from txt2vid.data import Vocab
from txt2vid.model import SentenceEncoder

from util.log import status, warn, error
from util.pickle import load
from util.misc import gen_perm

# TODO: load from args
FRAME_SIZE=64

def load_data(video_dir=None, vocab=None, anno=None, batch_size=64, val=False, num_workers=4, num_channels=3, random_frames=0):
    # TODO
    from data import get_loader

    if num_channels == 3:
        transform = transforms.Compose([transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5],[0.5])])

    return get_loader(video_dir, anno, vocab, transform, batch_size, shuffle=not val, num_workers=num_workers, random_frames=random_frames)

# parameters for the training algorithm
class TrainParams(object):
    # algorithm constants
    discrim_steps = 1
    gen_steps = 1

    # if true, will divide each step by discrim/gen_steps respectively
    mean_discrim_loss = True
    mean_gen_loss = True

    # ---------
    # --- misc
    # ---------

    # whether or not to use SummaryWriter
    use_writer = False
    # period of iterations to log, <= 0 to not log
    log_period = 100
    # number of iterations to average loss over
    loss_window_size = 50
    # period to save generated examples, <= 0 to not
    save_example_period = 100
    # period to save trained models, <= 0 to not
    save_model_period = 100

    out_samples_dir = None

    def __init__(self):
        pass

    def read_from(self, args):
        self.__dict__.update(args.__dict__.copy())
        return self


class LabelledGanLoss(object):

    def __init__(self, real_label=None, fake_label=None, underlying_loss=None):
        assert(real_label is not None)
        assert(fake_label is not None)
        assert(underlying_loss is not None)
        self.loss = underlying_loss
        self.fake_label = real_label
        self.real_label = fake_label

    def _compute_loss(self, x, label):
        labels = torch.full(x.size(0), label, device=x.device)
        return self.loss(x, labels)

    def discrim_loss(self, fake=None, real=None):
        fake = self._compute_loss(fake, self.fake_label)
        real = self._compute_loss(real, self.real_label)
        return fake + real

    def gen_loss(self, fake=None):
        return self._compute_loss(fake, self.REAL_EXAMPLE)

class VanillaGanLoss(LabelledGanLoss):

    def __init__(self, bce_loss=True):
        loss = nn.BCEWithLogitsLoss() if bce_loss else nn.CrossEntropyLoss()
        super().__init__(underlying_loss=loss, real_label=1, fake_label=1)

class HingeGanLoss(LabelledGanLoss):

    def __init__(self, margin=2.0):
        self.loss = nn.HingeEmbeddingLoss(margin=margin)
        super().__init__(underlying_loss=loss, real_label=1, fake_label=-1)

class WassersteinGanLoss(object):

    def __init__(self):
        pass

    def discrim_loss(self, fake=None, real=None):
        return -(real.mean() - fake.mean())

    def gen_loss(self, fake=None):
        return -fake.mean()

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

def gan_train(gan=None, num_epoch=None, dataset=None, device=None, optD=None, optG=None, params=None, losses=None):
    from txt2vid.metrics import RollingAvgLoss

    gen_loss = RollingAvgLoss(window_size=params.loss_window_size)
    discrim_loss = RollingAvgLoss(window_size=params.loss_window_size)

    writer = None
    if params.use_writer:
        writer = SummaryWriter()

    for epoch in range(num_epoch):
        if params.log_period > 0:
            print('epoch=', epoch + 1)
            sys.stdout.flush()

        for i, (videos, captions, lengths) in enumerate(dataset):
            iteration = epoch*len(dataset) + i + 1

            videos = videos.to(device)
            captions = captions.to(device)

            batch_size = videos.size(0)
            num_frames = videos.size(2)

            cond = None
            if gan.cond_encoder is not None:
                _, _, cond = gan.cond_encoder(captions, lengths)
                # TODO: fine-tune?
                cond = cond.detach()

            # TODO: configure prior sample space
            latent = torch.randn(batch_size, gan.gen.latent_size, device=device)
            fake = gan(latent, cond=cond)

            # discrim step
            total_discrim_loss = 0
            for j in range(params.discrim_steps):
                loss = gan.discrim_step(real=videos,
                                        fake=fake.detach(),
                                        cond=cond,
                                        loss=losses.discrim_loss,
                                        retain_graph=(j != params.discrim_steps - 1))


                if params.mean_gen_loss:
                    loss /= params.discrim_steps

                optD.step()
                total_discrim_loss += float(loss)

                # TODO: normalisation step for discrim

            discrim_loss.update(float(total_discrim_loss))

            # generator
            total_g_loss = 0
            total_g_loss_recon = 0
            for j in range(params.gen_steps):
                if j != 0:
                    fake = gan(fake_inp)

                loss = gan.gen_step(fake=fake, cond=cond, loss=losses.gen_loss)
                if params.mean_gen_loss:
                    loss /= params.gen_steps

                loss.backward(retain_graph=j != params.gen_steps - 1)
                optG.step()

                total_g_loss += float(loss)
                # TODO: normalisation step for gen
                
            gen_loss.update(float(total_g_loss))

            if iteration != 1 and iteration % params.save_example_period == 0:
                to_save = {
                    'optG': optG.state_dict(),
                    'optD': optD.state_dict()
                }
                to_save.update(gan.save_dict())

                torch.save(to_save, '%s/iter_%d_lossG_%.4f_lossD_%.4f' % (params.out, iteration, gen_loss.get(), discrim_loss.get()))

                del to_save
                to_save = None
            
            if params.log_period > 0 and iteration % params.log_period == 0:
                gc.collect()
                sys.stdout.flush()
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % 
                        (epoch, num_epoch, i, len(dataset), discrim_loss.get(), gen_loss.get()))

            if params.save_example_period > 0:
                if iteration == 1 or iteration % params.save_example_period == 0:
                    to_save_real = videos
                    to_save_fake = fake

                    print(to_save_real.size())

                    num_frames = to_save_real.size(1)

                    # TODO: this is different
                    # depending on the generator
                    # so the generator should probs format or save examples
                    # for now this is fine

                    #to_save_real = to_save_real.permute(0, 2, 1, 3, 4)
                    #to_save_fake = to_save_fake.permute(0, 2, 1, 3, 4).contiguous()
                    to_save_real = to_save_real.view(-1, to_save_real.size(2), to_save_real.size(3), to_save_real.size(4))
                    to_save_fake = to_save_fake.view(-1, to_save_fake.size(2), to_save_fake.size(3), to_save_fake.size(4))

                    print('saving to %s' % params.out_samples)
                    #print(to_save_real.size())
                    vutils.save_image(to_save_real, '%s/real_samples.png' % params.out_samples, normalize=True, nrow=num_frames) #to_save_real.size(0))
                    vutils.save_image(to_save_fake, '%s/fake_samples_epoch_%03d_iter_%06d.png' % (params.out_samples, epoch, iteration), normalize=True, nrow=num_frames)#to_save_fake.size(0))
                    # TODO: check
                    with open('%s/fake_sentences_epoch%03d_iter_%06d.txt' % (params.out_samples, epoch, iteration), 'w') as out_f:
                        for cap in captions:
                            out_f.write('%s\n' % cap)

                    del to_save_fake

def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 100000)

    random.seed(seed)
    torch.manual_seed(seed)
    return seed

def setup_torch(use_cuda):
    cudnn.benchmark = True
    if torch.cuda.is_available() and not use_cuda:
        warn('cuda is available')

    return torch.device("cuda:0" if args.cuda else "cpu")

def main(args):
    seed = set_seed(args.seed)
    device = setup_torch(use_cuda=args.cuda)

    status('Seed: %d' % seed)
    status('Device set to: %s' % device)

    # TODO: changeme
    from txt2vid.models.discrim import Discrim
    from txt2vid.models.gen import Gen
    gen = Gen(z_slow_dim=256, z_fast_dim=256, out_channels=3, bottom_width=4, conv_ch=512, cond_dim=0).to(device)

    discrim = Discrim().to(device)

    discrims = [ discrim ]

    if args.sent_encode_path:
        status("Loading pre-trained sentence model from %s" % args.sent_encode_path)
        txt_encoder = torch.load(args.sent_encode_path)
        if 'txt' in txt_encoder:
            txt_encoder = txt_encoder['txt'].to(device)
        status("Sentence encode size = %d" % txt_encoder.encoding_size)
    else:
        status("Using random init sentence encoder")
        txt_encoder = SentenceEncoder(embed_size=args.word_embed,
                                      hidden_size=args.hidden_state, 
                                      encoding_size=args.sent_encode, 
                                      num_layers=args.txt_layers, 
                                      vocab_size=len(vocab)).to(device)

    gan = CondGan(gen=gen, discrims=discrims, cond_encoder=txt_encoder)
    
    from util.dir import ensure_exists
    ensure_exists(args.out)
    ensure_exists(args.out_samples)

    import txt2vid.model
    txt2vid.model.USE_NORMAL_INIT=args.use_normal_init

    status('Loading vocab from %s' % args.vocab)
    vocab = load(args.vocab)

    status('Loading data from %s' % args.data)
    dataset = load_data(video_dir=args.data, anno=args.anno, vocab=vocab, batch_size=args.batch_size, val=False, num_workers=args.workers, num_channels=args.num_channels, random_frames=args.random_frames)

    optD = optim.Adam([ { "params": p } for p in gan.discrims_params ], lr=args.lr, betas=(args.beta1, args.beta2))

    optG = optim.Adam([ 
        { "params": gen.parameters() }, 
    ], lr=args.lr, betas=(args.beta1, args.beta2))

    if args.model is not None:
        to_load = torch.load(args.model)

        gan.load_from_dict(to_load)
        if 'optD' in to_load:
            optD.load_state_dict(to_load['optD'])
        if 'optG' in to_load:
            optG.load_state_dict(to_load['optG'])

    print(discrim)
    print(gen)
    print(txt_encoder)

    SAMPLE_LATENT_SIZE = args.latent_size

    print("Vocab Size %d" % len(vocab))
    print("Dataset len= %d (%d batches)" % (len(dataset)*args.batch_size, len(dataset)))

    params = TrainParams().read_from(args)
    print(device)
    # TODO: load loss dynamically
    losses = WassersteinGanLoss()
    gan_train(gan=gan, num_epoch=args.epoch, dataset=dataset, device=device, optD=optD, optG=optG, params=params, losses=losses)


if __name__ == '__main__':
    # TODO: args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Seed to use')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    #parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--workers', type=int, default=2, help='number of workers to help with loading/pre-processing data')

    parser.add_argument('--epoch', type=int, default=5, help='number of epochs to perform')

    parser.add_argument('--model', type=str, help='pretrained model', default=None)

    parser.add_argument('--data', type=str, help='video directory', required=True)
    parser.add_argument('--anno', type=str, help='annotation location', required=True)
    parser.add_argument('--vocab', type=str, help='vocab location', required=True)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta1 for adam')
    
    parser.add_argument('--gen_steps', type=int, default=1, help='Number of generator steps to use per iteration')
    parser.add_argument('--discrim_steps', type=int, default=1, help='Number of discriminator steps to use per iteration')

    parser.add_argument('--sent_encode_path', type=str, default=None, help='Initial model for the sentence encoder')

    parser.add_argument('--word_embed', type=int, default=128, help='Dimensionality of each word (sentence model)')
    parser.add_argument('--hidden_state', type=int, default=256, help='Dimensionality of hidden state (sentence model)')
    parser.add_argument('--txt_layers', type=int, default=3, help='Number of layers in the sentence model')
    parser.add_argument('--sent_encode', type=int, default=256, help='Encoding for the sentence')

    parser.add_argument('--latent_size', type=int, default=256, help='Additional number of dimensions for random variable')
    parser.add_argument('--frame_latent_size', type=int, default=256, help='Latent size for each frame')

    parser.add_argument('--recon_lambda', type=float, default=0.1, help='Multiplier for reconstruction loss')
    parser.add_argument('--recon_l2', action='store_true', help='Use L2 loss for recon')

    parser.add_argument('--use_normal_init', type=int, default=0, help='Use normal init')

    parser.add_argument('--out', type=str, default='out', help='dir output path')
    parser.add_argument('--out_samples', type=str, default='out_samples', help='dir output path')

    parser.add_argument('--num_channels', type=int, default=1, help='number of channels in input')
    parser.add_argument('--random_frames', type=int, default=0, help='use random frames')

    parser.add_argument('--weight_clip', type=float, default=0.01, help='weight clip value')

    args = parser.parse_args()
    main(args)
