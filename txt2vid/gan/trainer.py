import sys

from txt2vid.util.dir import ensure_exists
from txt2vid.util.log import status
from txt2vid.util.metrics import RollingAvgLoss

import torch
import torchvision.utils as vutils

def add_params_to_parser(parser):
    parser.add_argument('--log_period', type=int, default=20, help='period to log')
    parser.add_argument('--loss_window_size', type=int, default=20, help='window size for logging')
    parser.add_argument('--no_mean_discrim_loss', action='store_false', default=True, help='divides each discrim step loss by discrim_steps')
    parser.add_argument('--no_mean_gen_loss', action='store_false', default=True, help='divides each gen step loss by gen_steps')

    parser.add_argument('--discrim_steps', type=int, default=1, help='Number of discriminator steps to use per iteration')
    parser.add_argument('--gen_steps', type=int, default=1, help='Number of generator steps to use per iteration')

    # saving
    parser.add_argument('--save_initial', action='store_true', default=False, help='save initial model')
    parser.add_argument('--save_initial_examples', action='store_true', default=False, help='save initial sample')
    parser.add_argument('--save_model_period', type=int, default=100, help='number of iters until model is saved')
    parser.add_argument('--save_example_period', type=int, default=100, help='number of iters until model is saved')
    parser.add_argument('--use_writer', action='store_true', default=False, help='write losses to SummaryWriter (tensorboardX)')
    parser.add_argument('--out', type=str, default='out', help='dir output path')
    parser.add_argument('--out_samples', type=str, default='out_samples', help='dir output path')

    return parser

# TODO: generalise
def train(gan=None, num_epoch=None, dataset=None, device=None, optD=None, optG=None, params=None, vocab=None, losses=None, channel_first=True, end2end=True):
    status("Parameter settings:")
    print(locals())
    print()

    ensure_exists(params.out)
    ensure_exists(params.out_samples)

    gen_loss = RollingAvgLoss(window_size=params.loss_window_size)
    discrim_loss = RollingAvgLoss(window_size=params.loss_window_size)

    writer = None
    if params.use_writer:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    for epoch in range(num_epoch):
        if params.log_period > 0:
            status('Epoch %d started' % (epoch + 1))

        for i, (videos, captions, lengths) in enumerate(dataset):
            iteration = epoch*len(dataset) + i + 1

            videos = videos.to(device)
            captions = captions.to(device)

            if channel_first:
                videos = videos.permute(0, 2, 1, 3, 4)

            batch_size = videos.size(0)
            num_frames = videos.size(2)

            cond = None
            if gan.cond_encoder is not None:
                _, _, cond = gan.cond_encoder(captions, lengths)
                if not end2end:
                    cond = cond.detach()

            # TODO: configure prior sample space
            z = torch.randn(batch_size, gan.gen.latent_size, device=device)
            fake = gan(z, cond=cond)

            # discrim step
            total_discrim_loss = 0
            for j in range(params.discrim_steps):
                loss = gan.discrim_step(real=videos,
                                        fake=fake.detach(),
                                        cond=cond,
                                        loss=losses.discrim_loss)

                if not params.no_mean_gen_loss:
                    loss /= params.discrim_steps
                
                loss.backward(retain_graph=j != params.discrim_steps - 1 or end2end)

                optD.step()
                total_discrim_loss += float(loss)

                # TODO: normalisation step for discrim

            discrim_loss.update(float(total_discrim_loss))

            _, _, real_pred = gan.all_discrim_forward(real=videos, cond=cond, fake=None, loss=None)

            # generator
            total_g_loss = 0
            total_g_loss_recon = 0
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
                #gc.collect()
                sys.stdout.flush()
                status('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % 
                        (epoch, num_epoch, i, len(dataset), discrim_loss.get(), gen_loss.get()))

            if params.save_example_period > 0:
                if (iteration == 1 and params.save_initial_examples) or iteration % params.save_example_period == 0:
                    to_save_real = videos
                    to_save_fake = fake

                    # TODO: this is different
                    # depending on the generator
                    # so the generator should probs format or save examples
                    # for now this is fine

                    if channel_first:
                        to_save_real = to_save_real.permute(0, 2, 1, 3, 4)
                        to_save_fake = to_save_fake.permute(0, 2, 1, 3, 4).contiguous()

                    num_frames = to_save_real.size(1)

                    to_save_real = to_save_real.view(-1, to_save_real.size(2), to_save_real.size(3), to_save_real.size(4))
                    to_save_fake = to_save_fake.view(-1, to_save_fake.size(2), to_save_fake.size(3), to_save_fake.size(4))

                    status('saving to %s' % params.out_samples)
                    #print(to_save_real.size())
                    vutils.save_image(to_save_real, '%s/real_samples.png' % params.out_samples, normalize=True, nrow=num_frames) #to_save_real.size(0))
                    vutils.save_image(to_save_fake, '%s/fake_samples_epoch_%03d_iter_%06d.png' % (params.out_samples, epoch, iteration), normalize=True, nrow=num_frames)#to_save_fake.size(0))
                    # TODO: check
                    with open('%s/fake_sentences_epoch%03d_iter_%06d.txt' % (params.out_samples, epoch, iteration), 'w') as out_f:
                        for cap in captions:
                            for tok in cap:
                                out_f.write('%s ' % vocab.get_word(int(tok)))
                            out_f.write('\n')


                    del to_save_fake
