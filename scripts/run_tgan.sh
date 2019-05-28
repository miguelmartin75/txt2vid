#!/bin/bash

OUT_NAME=${OUT_NAME:-output_tgan_uncond_gp}
USE_NORMAL=${USE_NORMAL:-0}
RECON_LOSS=${RECON_LOSS:-0.0}

ROOT_DIR=${ROOT_DIR:-./txt2vid_2/}
BASE_DIR=${BASE_DIR:-./cooking_videos/train}
VIDEO_DIR=${VIDEO_DIR:-$BASE_DIR}
ANNO=${ANNO:-./msr/msr.pickle}
VOCAB=${VOCAB:-./msr/msr.vocab}
OUT_DIR=${OUT_DIR:-/run/media/doubleu/Linux/honours/temp/${OUT_NAME}}
EXAMPLE_DIR=${EXAMPLE_DIR:-/run/media/doubleu/Linux/honours/temp/${OUT_NAME}_samples}

# TODO: change G and D to be json files
#python3 txt2vid/train/gan.py --data $VIDEO_DIR --anno $ANNO --workers 4 --batch_size 48 --epochs 100 --out $OUT_DIR --out_samples $EXAMPLE_DIR --num_channels 3 --vocab $VOCAB --cuda --G txt2vid.models.tgan.gen.Gen --D txt2vid.models.tgan.discrim.Discrim --sent txt2vid.models.txt.basic.Seq2Seq --sent_weights msr/sent.pth --frame_size 64 --D_names video frame motion --G_lr 0.0001 --D_lr 0.0001 --D_beta1 0.5 --D_beta2 .9 --G_beta1 0.5 --G_beta2 .9 --D_loss txt2vid.gan.losses.WassersteinGanLoss --init_method xavier --discrim_steps 1 --dont_use_sent
#python3 txt2vid/train/gan.py --data $VIDEO_DIR --anno $ANNO --workers 8 --batch_size 32 --epochs 100 --out $OUT_DIR --out_samples $EXAMPLE_DIR --num_channels 3 --vocab $VOCAB --cuda --G txt2vid.models.tgan.gen.Gen --D txt2vid.models.tgan.discrim.Discrim --sent txt2vid.models.txt.basic.Seq2Seq --sent_weights msr/sent.pth --frame_size 64 --D_names video --G_lr 0.0001 --D_lr 0.0001 --D_beta1 0. --D_beta2 .9 --G_beta1 0. --G_beta2 .9 --D_loss txt2vid.gan.losses.WassersteinGanLoss --init_method xavier --discrim_steps 5 --dont_use_sent --seed 1 --gp_lambda 10 #--no_mean_discrim_loss 

#python3 txt2vid/train/gan.py --data ./config/cifar10.json --workers 8 --batch_size 20 --epochs 100 --out $OUT_DIR --out_samples $EXAMPLE_DIR --num_channels 3 --vocab $VOCAB --cuda --G txt2vid.models.img.models.Gen --D txt2vid.models.img.models.Discrim --sent txt2vid.models.txt.basic.Seq2Seq --sent_weights msr/sent.pth --frame_size 64 --D_names video --G_lr 0.0001 --D_lr 0.0001 --D_beta1 0.5 --D_beta2 .9 --G_beta1 0.5 --G_beta2 .9 --D_loss txt2vid.gan.losses.WassersteinGanLoss --init_method xavier --discrim_steps 5 --dont_use_sent --seed 1 --gp_lambda 10 --img_model --data_is_imgs --no_mean_discrim_loss --log_period 10 --loss_window_size 10 --use_writer #--anno $ANNO 

python3 txt2vid/train/gan.py --data ./config/mrvdc.json --workers 8 --batch_size 32 --epochs 100 --out $OUT_DIR --out_samples $EXAMPLE_DIR --num_channels 3 --vocab $VOCAB --cuda --G txt2vid.models.img.models.Gen --D txt2vid.models.img.models.Discrim --sent txt2vid.models.txt.basic.Seq2Seq --sent_weights msr/sent.pth --frame_size 64 --D_names video --G_lr 0.0001 --D_lr 0.0001 --D_beta1 0.5 --D_beta2 .9 --G_beta1 0.5 --G_beta2 .9 --D_loss txt2vid.gan.losses.WassersteinGanLoss --init_method xavier --discrim_steps 5 --dont_use_sent --seed 1 --gp_lambda 10 --img_model --no_mean_discrim_loss --log_period 10 --loss_window_size 10 --use_writer --anno $ANNO 
