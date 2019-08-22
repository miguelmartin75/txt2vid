#!/bin/bash

OUT_NAME=${OUT_NAME:-samples_cooking_128x128_rsgangp_3}
USE_NORMAL=${USE_NORMAL:-0}
RECON_LOSS=${RECON_LOSS:-0.0}

ROOT_DIR=${ROOT_DIR:-./txt2vid_2/}
BASE_DIR=${BASE_DIR:-./cooking_videos/train}
VIDEO_DIR=${VIDEO_DIR:-$BASE_DIR}
ANNO=${ANNO:-./msr/msr.pickle}
#ANNO=${ANNO:-/run/media/doubleu/Linux/synthetic/train/sent.pickle}
#ANNO=${ANNO:-./reddit_videos/captions.pickle}
VOCAB=${VOCAB:-./msr/msr.vocab}
#VOCAB=${VOCAB:-./reddit_videos/vocab.pickle}
OUT_DIR=${OUT_DIR:-/run/media/doubleu/Linux/honours/temp/${OUT_NAME}}
EXAMPLE_DIR=${EXAMPLE_DIR:-/run/media/doubleu/Linux/honours/temp/${OUT_NAME}_samples}

#python3 txt2vid/train/gan.py --data ./config/mrvdc.json --workers 3 --batch_size 1 --epochs 161 --out $OUT_DIR --out_samples $EXAMPLE_DIR --num_channels 3 --vocab $VOCAB --cuda --G txt2vid.models.tganv2.gen.MultiScaleGen --D txt2vid.models.tganv2.discrim.MultiScaleDiscrim --sent txt2vid.models.txt.basic.Seq2Seq --sent_weights msr/sent.pth --frame_sizes 8 16 32 64 --D_names video --G_lr 0.0002 --D_lr 0.0002 --D_beta1 0.5 --D_beta2 .999 --G_beta1 0.5 --G_beta2 .999 --D_loss txt2vid.gan.losses.HingeGanLoss --init_method xavier --discrim_steps 1 --dont_use_sent --seed 100 --gp_lambda .5 --no_mean_discrim_loss --log_period 10 --save_example_period 500 --save_model_period 500 --loss_window_size 50 --use_writer --anno $ANNO --subsample_input --sample_batch_size 20 --test --weights ./cooking_model --num_samples 100 #--weights /run/media/doubleu/Linux/honours/temp/output_tganv2_uncond_synth_rsgangp_128x128/iter_23500_lossG_0.9940_lossD_0.6160 --num_samples 100

python3 txt2vid/train/gan.py --data ./config/mrvdc.json --workers 3 --batch_size 128 --epochs 161 --out $OUT_DIR --out_samples $EXAMPLE_DIR --num_channels 3 --vocab $VOCAB --cuda --G txt2vid.models.tganv2.gen.MultiScaleGen --D txt2vid.models.tganv2.discrim.MultiScaleDiscrim --sent txt2vid.models.txt.basic.Seq2Seq --sent_weights msr/sent.pth --frame_sizes 16 32 64 128 --D_names video --G_lr 0.0002 --D_lr 0.0002 --D_beta1 0.5 --D_beta2 .999 --G_beta1 0.5 --G_beta2 .999 --D_loss txt2vid.gan.losses.RSGANLoss --init_method xavier --discrim_steps 1 --dont_use_sent --seed 100 --gp_lambda .5 --no_mean_discrim_loss --log_period 10 --save_example_period 200 --save_model_period 400 --loss_window_size 50 --use_writer --anno $ANNO --subsample_input --sample_batch_size 20 --weights /run/media/doubleu/Linux/cooking_v2 --num_samples 100 --end2end #--test --end2end #--weights /run/media/doubleu/Linux/honours/temp/output_tganv2_uncond_synth_rsgangp_128x128/iter_23500_lossG_0.9940_lossD_0.6160 --num_samples 100

#python3 txt2vid/train/gan.py --data config/mrvdc.json --workers 4 --batch_size 48 --epochs 100 --out $OUT_DIR --out_samples $EXAMPLE_DIR --num_channels 3 --vocab $VOCAB --cuda --G txt2vid.models.tganv2.gen.MultiScaleGen --D txt2vid.models.tganv2.discrim.MultiScaleDiscrim --sent txt2vid.models.txt.basic.Seq2Seq --sent_weights msr/sent.pth --frame_sizes 8 16 32 64 --D_names video --G_lr 0.0001 --D_lr 0.0001 --D_beta1 0.0 --D_beta2 .9 --G_beta1 0.0 --G_beta2 .9 --D_loss txt2vid.gan.losses.WassersteinGanLoss --init_method xavier --discrim_steps 1 --dont_use_sent --seed 100 --gp_lambda .5 --no_mean_discrim_loss --log_period 10 --save_example_period 50 --loss_window_size 20 --use_writer --anno $ANNO --subsample_input --sample_batch_size 20 #--weights /run/media/doubleu/Linux/honours/temp/output_tganv2_uncond_vanilla_2_synth/iter_2950_lossG_-0.2155_lossD_-0.1343
