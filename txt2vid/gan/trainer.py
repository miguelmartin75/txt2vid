import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from txt2vid.util.dir import ensure_exists
from txt2vid.util.log import status
from txt2vid.util.metrics import RollingAvg
from txt2vid.util.stopwatch import Stopwatch

def add_params_to_parser(parser):
    parser.add_argument('--data_is_imgs', action='store_true', default=False, help='is the data images? If not, assume it is videos')
    parser.add_argument('--img_model', action='store_true', default=False, help='does the GAN only do images?')

    parser.add_argument('--log_period', type=int, default=20, help='period to log')
    parser.add_argument('--loss_window_size', type=int, default=20, help='window size for logging')
    parser.add_argument('--no_mean_discrim_loss', action='store_false', default=True, help='divides each discrim step loss by discrim_steps')
    parser.add_argument('--no_mean_gen_loss', action='store_false', default=True, help='divides each gen step loss by gen_steps')

    parser.add_argument('--sample_batch_size', type=int, default=None, help='batch size to gen samples')
    parser.add_argument('--discrim_steps', type=int, default=1, help='Number of discriminator steps to use per iteration')
    parser.add_argument('--gen_steps', type=int, default=1, help='Number of generator steps to use per iteration')

    parser.add_argument('--gp_lambda', type=float, default=-1, help='GP lambda hyper-param (negative to disable GP)')

    # saving
    parser.add_argument('--save_initial', action='store_true', default=False, help='save initial model')
    parser.add_argument('--save_initial_examples', action='store_true', default=False, help='save initial sample')
    parser.add_argument('--save_model_period', type=int, default=100, help='number of iters until model is saved')
    parser.add_argument('--save_example_period', type=int, default=100, help='number of iters until model is saved')
    parser.add_argument('--use_writer', action='store_true', default=False, help='write losses to SummaryWriter (tensorboardX)')
    parser.add_argument('--out', type=str, default='out', help='dir output path')
    parser.add_argument('--out_samples', type=str, default='out_samples', help='dir output path')

    parser.add_argument('--subsample_input', action='store_true', default=False, help='should subsampling be applied to the input?')

    # TODO: just allow custom datasets in main
    return parser

def test(gan=None, num_samples=1, dataset=None, device=None, params=None, channel_first=True):
    ensure_exists(params.out_samples)

    gan.gen.eval()
    for i in range(num_samples):
        for j, data in enumerate(dataset):
            batch_size = data[0].size(0)

            y = [ a.to(device) if isinstance(a, torch.Tensor) else a for a in data[1:] ]

            cond = None
            if gan.cond_encoder is not None and len(y) >= 2:
                _, _, cond = gan.cond_encoder.encode(y[0], y[1])
                if not end2end:
                    cond = cond.detach()

            z = torch.randn(batch_size, gan.gen.latent_size, device=device)
            if True:
                fake = gan(z, cond=cond, output_blocks=None)
            else:
                fake = gan(z, cond=cond, output_blocks=range(4))

            if cond is not None:
                path = '%s/sentences_%d_%d.txt' % (params.out_samples, i, j)
                save_sentences(y[0], path=path)

            for f in fake:
                if params.img_model:
                    h, w = f.size(2), f.size(3)
                else:
                    h, w = f.size(3), f.size(4)
                path = '%s/%d_%d_%dx%d.jpg' % (params.out_samples, i, j, h, w)
                status("saving to %s" % path)
                save_frames(f, path=path, channel_first=channel_first, is_images=params.img_model)

            break

def save_frames(frames, path=None, channel_first=True, is_images=False):
    if channel_first:
        if is_images:
            frames = frames.unsqueeze(2)
        frames = frames.permute(0, 2, 1, 3, 4).contiguous()

    num_frames = frames.size(1)
    output = frames.view(-1, frames.size(2), frames.size(3), frames.size(4))

    vutils.save_image(output, path, normalize=True, nrow=num_frames)

def save_sentences(captions, path=None):
    for cap in y[0]:
        words = None
        try:
            words = vocab.to_words(cap)
        except:
            words = cap

        out_f.write(words)
        out_f.write('\n')

# TODO: generalise
def train(gan=None, num_epoch=None, dataset=None, device=None, optD=None, optG=None, params=None, vocab=None, losses=None, channel_first=True, end2end=True):
    if params.debug:
        status("Parameter settings:")
        print(locals())
        print()

    if params.sample_batch_size is None:
        params.sample_batch_size = params.batch_size

    ensure_exists(params.out)
    ensure_exists(params.out_samples)

    writer = None
    if params.use_writer:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    def detach_all(x):
        return [ y.detach() for y in x ]

    def multiscale_data(x, cond, device=None):
        assert(channel_first)

        num_data_points = len(params.frame_sizes)
        if num_data_points == 1:
            if cond is None:
                return [x], None
            else:
                return [x], [cond]

        from txt2vid.models.layers import Subsample
        subsampler = Subsample()

        xs, conds = [], []
        for i in range(num_data_points):
            num_frames = x.size(2)
            if i != num_data_points - 1:
                fs = params.frame_sizes[i]
                resized = F.interpolate(x, size=(num_frames, fs, fs))
            else:
                resized = x

            #print(i, resized.size())
            xs.append(resized)#.to(device))

            if params.subsample_input:
                x, _ = subsampler(x)
                if cond is not None:
                    cond = cond[::2]

            if cond is not None:
                conds.append(cond)

        if len(conds) == 0:
            return xs, None

        return xs, conds

    gen_loss = RollingAvg(window_size=params.loss_window_size)
    discrim_loss = RollingAvg(window_size=params.loss_window_size)

    import tqdm

    for epoch in range(num_epoch):
        if params.log_period > 0:
            status('Epoch %d started' % (epoch + 1))

        pbar = tqdm.tqdm(dataset)

        for i, data in enumerate(pbar):
            iteration = epoch*len(dataset) + i + 1

            x = data[0].to(device)
            y = [ a.to(device) if isinstance(a, torch.Tensor) else a for a in data[1:] ]

            batch_size = x.size(0)
            num_frames = 1
            if not params.data_is_imgs:
                num_frames = x.size(1)
                if channel_first:
                    x = x.permute(0, 2, 1, 3, 4)

            if params.img_model and not params.data_is_imgs:
                # make it an image
                # note: assumes x.size(2) == num_frames == 1
                x = x.squeeze(2)

            #print("Multi-scale =")
            #for i, d in enumerate(x):
            #    print(d.size())
            #    num_frames = d.size(2)

            #    to_save_real = d
            #    if channel_first:
            #        to_save_real = to_save_real.permute(0, 2, 1, 3, 4).contiguous()
            #    to_save_real = to_save_real.view(-1, to_save_real.size(2), to_save_real.size(3), to_save_real.size(4))
            #    print("saving to", params.out_samples)
            #    print("d.size=", d.size())
            #    print("save.size=", to_save_real.size())
            #    vutils.save_image(to_save_real, '%s/real_samples_%d.png' % (params.out_samples, i), normalize=True, nrow=num_frames) 

            cond = None
            if gan.cond_encoder is not None and len(y) >= 2:
                _, _, cond = gan.cond_encoder.encode(y[0], y[1])
                if not end2end:
                    cond = cond.detach()

            # get the different scales for the data
            x, cond = multiscale_data(x, cond, device=device)

            # TODO: configure prior sample space
            z = torch.randn(batch_size, gan.gen.latent_size, device=device)
            fake = gan(z, cond=cond)

            # discrim step
            total_discrim_loss = 0
            for j in range(params.discrim_steps):
                loss = gan.discrim_step(real=x,
                                        fake=detach_all(fake),
                                        cond=cond,
                                        loss=losses.discrim_loss,
                                        gp_lambda=params.gp_lambda)

                if not params.no_mean_discrim_loss:
                    loss /= params.discrim_steps
                
                loss.backward(retain_graph=j != params.discrim_steps - 1 or end2end)

                optD.step()
                total_discrim_loss += float(loss)

                # TODO: normalisation step for discrim

            discrim_loss.update(float(total_discrim_loss))

            _, _, real_pred = gan.all_discrim_forward(real=x, cond=cond, fake=None, loss=None)

            # generator
            total_g_loss = 0
            for j in range(params.gen_steps):
                if j != 0:
                    fake = gan(z, cond=cond)

                loss = gan.gen_step(fake=fake, real_pred=real_pred, cond=cond, loss=losses.gen_loss)
                if not params.no_mean_gen_loss:
                    loss /= params.gen_steps

                loss.backward(retain_graph=j != params.gen_steps - 1)
                optG.step()

                total_g_loss += float(loss)
                # TODO: normalisation step for gen

            gen_loss.update(float(total_g_loss))

            if (iteration == 1 and params.save_initial) or iteration % params.save_example_period == 0:
                to_save = {
                    'optG': optG.state_dict(),
                    'optD': optD.state_dict()
                }
                to_save.update(gan.save_dict())

                torch.save(to_save, '%s/iter_%d_lossG_%.4f_lossD_%.4f' % (params.out, iteration, gen_loss.get(), discrim_loss.get()))

                del to_save
                to_save = None
            
            if params.log_period > 0 and iteration % params.log_period == 0:
                sys.stdout.flush()
                desc = 'Iter %d, Loss_D: %.4f Loss_G: %.4f (%.2fGB used, %.2fGB cached)' % (iteration, discrim_loss.get(), gen_loss.get(), torch.cuda.max_memory_allocated() / (10**9), torch.cuda.max_memory_cached() / (10**9))
                pbar.set_description(desc)

                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_max_memory_cached()

            if params.save_example_period > 0:
                if (iteration == 1 and params.save_initial_examples) or iteration % params.save_example_period == 0:
                    to_save_real = x[0]

                    #gan.gen.eval()
                    #with torch.no_grad():
                    #    z = torch.randn(params.sample_batch_size, gan.gen.latent_size, device=device)
                    #    cond_to_use = None
                    #    if cond is not None:
                    #        cond_to_use = cond[0][0:params.sample_batch_size]

                    #    fake = gan(z, cond=cond_to_use)
                    #gan.gen.train()

                    # TODO: this is different
                    # depending on the generator
                    # so the generator should probs format or save examples
                    # for now this is fine
                    status('saving to %s (iteration %d)' % (params.out_samples, iteration))

                    save_frames(to_save_real, '%s/real_samples.png' % params.out_samples, is_images=params.img_model)

                    for to_save_fake in fake:
                        h, w = to_save_fake.size(2), to_save_fake.size(3)
                        path = '%s/fake_samples_epoch_%03d_iter_%06d_%dx%d.png' % (params.out_samples, epoch, iteration, h, w)
                        save_frames(to_save_fake, path=path, channel_first=channel_first, is_images=params.img_model)

                    if cond is not None:
                        path = '%s/sentences_epoch%03d_iter_%06d.txt' % (params.out_samples, epoch, iteration)
                        save_sentences(y[0], path=path)

                    del to_save_fake
