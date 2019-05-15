# constants for the training algorithm
class TrainParams(object):
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

    def __init__(self):
        pass

    def read_from(self, args):
        self.__dict__.update(args.__dict__.copy())
        return self

# TODO: generalise
def train(gan=None, num_epoch=None, dataset=None, device=None, optD=None, optG=None, params=None, losses=None):
    from txt2vid.util.metrics import RollingAvgLoss

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
