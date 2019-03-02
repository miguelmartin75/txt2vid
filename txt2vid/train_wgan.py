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
#import txt2vid.videogan as model
from txt2vid.model import SentenceEncoder

from util.log import status, warn, error
from util.pickle import load

#FRAME_SIZE=64
FRAME_SIZE=48

def gen_perm(n):
    old_perm = np.array(range(n))
    new_perm = np.random.permutation(old_perm)
    while (new_perm == old_perm).all():
        new_perm = np.random.permutation(old_perm)
    return new_perm


def load_data(video_dir=None, vocab=None, anno=None, batch_size=64, val=False, num_workers=4, num_channels=3, random_frames=0):
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

def main(args):
    if args.seed is None:
        args.seed = random.randint(1, 100000)
    status('Seed: %d' % args.seed)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        warn('cuda is available, you probably want to use --cuda')

    from util.dir import ensure_exists
    ensure_exists(args.out)
    ensure_exists(args.out_samples)

    import txt2vid.model
    txt2vid.model.USE_NORMAL_INIT=args.use_normal_init

    device = torch.device("cuda:0" if args.cuda else "cpu")
    ngpu = int(args.ngpu)

    vocab = load(args.vocab)
    dataset = load_data(video_dir=args.data, anno=args.anno, vocab=vocab, batch_size=args.batch_size, val=False, num_workers=args.workers, num_channels=args.num_channels, random_frames=args.random_frames)

    # TODO: params
    if args.sent_encode_path:
        txt_encoder = torch.load(args.sent_encode_path)
        if 'txt' in txt_encoder:
            txt_encoder = txt_encoder['txt'].to(device)
    else:
        txt_encoder = SentenceEncoder(embed_size=args.word_embed,
                                      hidden_size=args.hidden_state, 
                                      encoding_size=args.sent_encode, 
                                      num_layers=args.txt_layers, 
                                      vocab_size=len(vocab)).to(device)

    discrim = model.Discrim(txt_encode_size=txt_encoder.encoding_size,
                            num_channels=args.num_channels).to(device)
    gen = model.Generator(latent_size=(txt_encoder.encoding_size + args.latent_size), 
                          num_channels=args.num_channels).to(device)
    frame_map = model.FrameMap(num_channels=args.num_channels).to(device)
    motion_discrim = model.MotionDiscrim(txt_encode_size=txt_encoder.encoding_size).to(device)
    frame_discrim = model.FrameDiscrim(txt_encode_size=txt_encoder.encoding_size).to(device)


    optimizerD = optim.Adam([ 
        { "params": discrim.parameters() }, 
        #{ "params": txt_encoder.parameters() },
        { "params": frame_map.parameters() },
        { "params": frame_discrim.parameters() },
        { "params": motion_discrim.parameters() },
    ], lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.0005)
    optimizerG = optim.Adam([ 
        { "params": gen.parameters() }, 
        #{ "params": txt_encoder.parameters() } 
    ], lr=args.lr, betas=(args.beta1, args.beta2))

    if args.model is not None:
        to_load = torch.load(args.model)

        gen.load_state_dict(to_load['gen'])

        discrim.load_state_dict(to_load['discrim'])
        frame_map.load_state_dict(to_load['frame_map'])
        frame_discrim.load_state_dict(to_load['frame_map'])
        motion_discrim.load_state_dict(to_load['motion_discrim'])

        txt_encoder.load_state_dict(to_load['txt'])
        optimizerD.load_state_dict(to_load['optD'])
        optimizerG.load_state_dict(to_load['optG'])

    print(discrim)
    print(gen)
    print(txt_encoder)
    print(frame_map)
    print(motion_discrim)
    print(frame_discrim)

    REAL_LABEL = 1
    FAKE_LABEL = 0
    SAMPLE_LATENT_SIZE = args.latent_size

    REAL_LABELS = torch.full((args.batch_size,), REAL_LABEL, device=device, dtype=torch.float, requires_grad=False)
    FAKE_LABELS = torch.full((args.batch_size,), FAKE_LABEL, device=device, dtype=torch.float, requires_grad=False)

    criteria = nn.BCELoss()
    recon = nn.L1Loss()
    if args.recon_l2:
        recon = nn.MSELoss()

    print("Vocab Size %d" % len(vocab))
    print("Dataset len= %d (%d batches)" % (len(dataset)*args.batch_size, len(dataset)))

    def gen_step(fake=None, fake_frames=None, cap_fv=None, real_labels=None, real_labels_frames=None, last=True, real_videos=None):
        gen.zero_grad()
        txt_encoder.zero_grad()

        loss_d0 = 0.0
        real_pred = discrim(vids=fake, sent=cap_fv, device=device)
        loss_d0 = criteria(real_pred, real_labels)

        loss_d1 = 0.0
        discrim_frames = frame_discrim(fake_frames, sent=cap_fv, device=device)
        loss_d1 = criteria(discrim_frames, real_labels_frames)

        loss_d2 = 0.0
        real_labels_motion = real_labels_frames[0:-1, :] # (time, batch)
        motion_frames = motion_discrim(fake_frames, sent=cap_fv, device=device)
        loss_d2 = criteria(motion_frames, real_labels_motion)

        loss = loss_d0 + loss_d1 + loss_d2
        loss /= 3.0

        # don't think this is necessary
        recon_loss = 0.0
        if args.recon_lambda > 0:
            recon_loss = recon(fake, real_videos) 
            loss += args.recon_lambda * recon_loss

        loss.backward(retain_graph=not last)

        optimizerG.step()

        return loss, recon_loss

    def discrim_step(videos=None, frames=None, cap_fv=None, real_labels=None, fake_labels=None, real_labels_frames=None, fake_labels_frames=None, last=True, fake=None, fake_frames=None, device=None):

        txt_encoder.zero_grad()
        discrim.zero_grad()
        frame_map.zero_grad()
        motion_discrim.zero_grad()
        frame_discrim.zero_grad()

        incorrect_caps = cap_fv[gen_perm(cap_fv.size(0))]

        fake_frames = frame_map(fake.detach())
        frames = frame_map(videos.detach())

        # TODO: check
        ## D_0 - video sentence pairs
        # real example, correct caption - predict real
        loss_d0 = 0
        real_pred = discrim(vids=videos.detach(), sent=cap_fv, device=device)
        loss_discrim_real_cc = criteria(real_pred, real_labels)
        loss_d0 += loss_discrim_real_cc

        # real example, correct caption
        # predict fake
        fake_pred = discrim(vids=videos.detach(), sent=incorrect_caps, device=device)
        loss_discrim_real_ic = criteria(fake_pred, fake_labels)
        loss_d0 += loss_discrim_real_ic

        # fake example, correct caption
        fake_pred = discrim(vids=fake.detach(), sent=cap_fv, device=device)
        loss_discrim_fake = criteria(fake_pred, fake_labels)
        loss_d0 += loss_discrim_fake
        
        ## D_1 - frame sentence pairs

        loss_d1 = 0
        # real, correct sentence
        real_pred = frame_discrim(frames, sent=cap_fv, device=device)
        loss_frames_real_cc = criteria(real_pred, real_labels_frames)
        loss_d1 += loss_frames_real_cc

        # real, wrong sentence
        fake_pred = frame_discrim(frames, sent=incorrect_caps, device=device)
        loss_frames_real_ic = criteria(fake_pred, fake_labels_frames)
        loss_d1 += loss_frames_real_ic

        # fake, correct sentence
        fake_pred = frame_discrim(fake_frames, sent=cap_fv, device=device)
        loss_frames_fake = criteria(fake_pred, fake_labels_frames)
        loss_d1 += loss_frames_fake

        ## D_2 - motion sentence pairs
        loss_d2 = 0
        real_labels_motion = real_labels_frames[0:-1, :] # (time, batch)
        fake_labels_motion = fake_labels_frames[0:-1, :]

        # real, correct sentence
        real_pred = motion_discrim(frames, sent=cap_fv, device=device)
        loss_motion_real_cc = criteria(real_pred, real_labels_motion)
        loss_d2 += loss_motion_real_cc

        # real, incorrect sentence
        real_pred = motion_discrim(frames, sent=incorrect_caps, device=device)
        loss_motion_real_ic = criteria(real_pred, fake_labels_motion)
        loss_d2 += loss_motion_real_ic

        # fake, correct sentence
        fake_pred = motion_discrim(fake_frames, sent=cap_fv, device=device)
        loss_motion_fake = criteria(fake_pred, fake_labels_motion)
        loss_d2 += loss_motion_fake

        # compute loss
        loss_d0 /= 3.0
        loss_d1 /= 3.0 # BCELoss already takes mean
        loss_d2 /= 3.0 # BCELoss already takes mean

        loss = loss_d0 + loss_d1 + loss_d2
        loss /= 3.0

        loss.backward(retain_graph=not last)
        optimizerD.step()

        return loss

    
    DISCRIM_STEPS = args.discrim_steps
    GEN_STEPS = args.gen_steps

    from txt2vid.metrics import RollingAvgLoss

    LOSS_WINDOW_SIZE=50
    gen_loss = RollingAvgLoss(window_size=LOSS_WINDOW_SIZE)
    gen_recon_loss = RollingAvgLoss(window_size=LOSS_WINDOW_SIZE)
    discrim_loss = RollingAvgLoss(window_size=LOSS_WINDOW_SIZE)

    REAL_LABELS_FRAMES = torch.full((16, args.batch_size), REAL_LABEL, device=device)
    FAKE_LABELS_FRAMES = torch.full((16, args.batch_size), FAKE_LABEL, device=device)

    # TODO: when to backprop for txt_encoder
    for epoch in range(args.epoch):
        print('epoch=', epoch + 1)
        sys.stdout.flush()

        for i, (videos, captions, lengths) in enumerate(dataset):
            # TODO: hyper-params for GAN training
            videos = videos.to(device).permute(0, 2, 1, 3, 4)
            captions = captions.to(device)

            batch_size = videos.size(0)
            real_labels = REAL_LABELS[0:batch_size]
            fake_labels = FAKE_LABELS[0:batch_size]

            num_frames = videos.size(2)
            real_labels_frames = REAL_LABELS_FRAMES[:, 0:batch_size]
            fake_labels_frames = FAKE_LABELS_FRAMES[:, 0:batch_size]

            _, _, cap_fv = txt_encoder(captions, lengths)
            cap_fv = cap_fv.detach()

            latent = torch.randn(batch_size, SAMPLE_LATENT_SIZE, device=device)
            fake_inp = torch.cat((cap_fv, latent), dim=1)
            fake_inp = fake_inp.view(fake_inp.size(0), fake_inp.size(1), 1, 1, 1)

            fake = gen(fake_inp)

            # discrim step
            for j in range(DISCRIM_STEPS):
                ld = discrim_step(videos=videos,
                                  #frames=real_frames,
                                  cap_fv=cap_fv,
                                  real_labels=real_labels, 
                                  fake_labels=fake_labels,
                                  real_labels_frames=real_labels_frames,
                                  fake_labels_frames=fake_labels_frames,
                                  fake=fake.detach(),
                                  #fake_frames=fake_frames,
                                  last=(j == DISCRIM_STEPS - 1),
                                  device=device)
                discrim_loss.update(float(ld))

            #_, _, cap_fv = txt_encoder(captions, lengths)
            #fake_inp = torch.cat((cap_fv, latent), dim=1)
            #fake = gen(fake_inp)
            fake_frames = frame_map(fake)

            # generator
            for j in range(GEN_STEPS):
                if j != 0:
                    fake = gen(fake_inp)

                lg, lgr = gen_step(fake=fake, 
                                   fake_frames=fake_frames,
                                   cap_fv=cap_fv,
                                   real_labels=real_labels,
                                   real_labels_frames=real_labels_frames,
                                   real_videos=videos,
                                   last=(j == GEN_STEPS - 1))
                
                gen_loss.update(float(lg))
                gen_recon_loss.update(float(lgr))

            iteration = epoch*len(dataset) + i

            if iteration != 0 and iteration % 100 == 0:
                
                to_save = {
                    'gen': gen.state_dict(),
                    'discrim': discrim.state_dict(),
                    'frame_map': frame_map.state_dict(),
                    'frame_discrim': frame_discrim.state_dict(),
                    'motion_discrim': motion_discrim.state_dict(),
                    'txt': txt_encoder.state_dict(),
                    'optG': optimizerG.state_dict(),
                    'optD': optimizerD.state_dict()
                }

                torch.save(to_save, '%s/iter_%d_lossG_%.4f_lossD_%.4f' % (args.out, iteration, gen_loss.get(), discrim_loss.get()))

                del to_save
                to_save = None

            if iteration % 10 == 0:
                gc.collect()
                sys.stdout.flush()
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f (recon = %.4f)' % 
                        (epoch, args.epoch, i, len(dataset), discrim_loss.get(), gen_loss.get(), gen_recon_loss.get()))

            if iteration % 50 == 0:
                # TODO: output sentences
                to_save_real = videos
                to_save_fake = fake

                num_frames = to_save_real.size(2)

                #print(captions[0])
                to_save_real = to_save_real.permute(0, 2, 1, 3, 4)
                to_save_real = to_save_real.view(-1, to_save_real.size(2), to_save_real.size(3), to_save_real.size(4))
                to_save_fake = to_save_fake.permute(0, 2, 1, 3, 4).contiguous()
                to_save_fake = to_save_fake.view(-1, to_save_fake.size(2), to_save_fake.size(3), to_save_fake.size(4))

                print('saving to %s' % args.out_samples)
                #print(to_save_real.size())
                vutils.save_image(to_save_real, '%s/real_samples.png' % args.out_samples, normalize=True, nrow=num_frames) #to_save_real.size(0))
                vutils.save_image(to_save_fake, '%s/fake_samples_epoch_%03d_iter_%06d.png' % (args.out_samples, epoch, iteration), normalize=True, nrow=num_frames)#to_save_fake.size(0))

                del to_save_fake




if __name__ == '__main__':
    # TODO: args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Seed to use')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--workers', type=int, default=2, help='number of workers to help with loading/pre-processing data')

    parser.add_argument('--epoch', type=int, default=5, help='number of epochs to perform')

    parser.add_argument('--model', type=str, help='pretrained model', default=None)

    parser.add_argument('--data', type=str, help='video directory', required=True)
    parser.add_argument('--anno', type=str, help='annotation location', required=True)
    parser.add_argument('--vocab', type=str, help='vocab location', required=True)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta1 for adam. default=0.5')
    
    parser.add_argument('--gen_steps', type=int, default=1, help='Number of generator steps to use per iteration')
    parser.add_argument('--discrim_steps', type=int, default=5, help='Number of discriminator steps to use per iteration')

    parser.add_argument('--sent_encode_path', type=str, default=None, help='Initial model for the sentence encoder')

    parser.add_argument('--word_embed', type=int, default=128, help='Dimensionality of each word (sentence model)')
    parser.add_argument('--hidden_state', type=int, default=256, help='Dimensionality of hidden state (sentence model)')
    parser.add_argument('--txt_layers', type=int, default=3, help='Number of layers in the sentence model')
    parser.add_argument('--sent_encode', type=int, default=256, help='Encoding for the sentence')

    parser.add_argument('--latent_size', type=int, default=100, help='Additional number of dimensions for random variable')

    parser.add_argument('--recon_lambda', type=float, default=0.1, help='Multiplier for reconstruction loss')
    parser.add_argument('--recon_l2', action='store_true', help='Use L2 loss for recon')

    parser.add_argument('--use_normal_init', type=int, default=0, help='Use normal init')

    parser.add_argument('--out', type=str, default='out', help='dir output path')
    parser.add_argument('--out_samples', type=str, default='out_samples', help='dir output path')

    parser.add_argument('--num_channels', type=int, default=1, help='number of channels in input')
    parser.add_argument('--random_frames', type=int, default=0, help='use random frames')

    parser.add_argument('--lambda', type=float, default=10, help='gradient penalty')

    args = parser.parse_args()
    main(args)
