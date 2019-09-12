#!/bin/bash

OUT_NAME=${OUT_NAME:-mrvdc_24.5k_with_real}
USE_NORMAL=${USE_NORMAL:-0}
RECON_LOSS=${RECON_LOSS:-0.0}

ROOT_DIR=${ROOT_DIR:-./txt2vid_2/}
DATA_JSON=${DATA_JSON:-./config/mrvdc.json}
#DATA_JSON=${DATA_JSON:-./config/mrvdc.json}
#ANNO=${ANNO:-/run/media/doubleu/Linux/synthetic/train/sent.pickle}
#VOCAB=${VOCAB:-/run/media/doubleu/Linux/synthetic/train/vocab.pickle}
ANNO=${ANNO:-./msr/msr.pickle}
VOCAB=${VOCAB:-./msr/msr.vocab}
SENT_MODEL=${SENT_MODEL:-./msr/sent.pth}
#ANNO=${ANNO:-./reddit_videos/captions.pickle}
#VOCAB=${VOCAB:-./reddit_videos/vocab.pickle}
OUT_DIR=${OUT_DIR:-/run/media/doubleu/Linux/honours/thesis_results/${OUT_NAME}}
EXAMPLE_DIR=${EXAMPLE_DIR:-/run/media/doubleu/Linux/honours/thesis_results/${OUT_NAME}_samples}

python3 txt2vid/train/gan.py --data $DATA_JSON --workers 3 --batch_size 1 --epochs 200 --out $OUT_DIR --out_samples $EXAMPLE_DIR --num_channels 3 --vocab $VOCAB --cuda --G txt2vid.models.tganv2_cond.gen.MultiScaleGen --D txt2vid.models.tganv2_cond.discrim.MultiScaleDiscrim --sent txt2vid.models.txt.basic.Seq2Seq --frame_sizes 8 16 32 64 --D_names video --G_lr 0.0002 --D_lr 0.0002 --D_beta1 0.5 --D_beta2 .999 --G_beta1 0.5 --G_beta2 .999 --D_loss txt2vid.gan.losses.RSGANLoss --init_method xavier --discrim_steps 1 --seed 100 --gp_lambda .5 --no_mean_discrim_loss --log_period 10 --save_example_period 500 --save_model_period 500 --loss_window_size 50 --use_writer --anno $ANNO --subsample_input --sample_batch_size 20 --test --weights /run/media/doubleu/Linux/honours/temp/msr_64x64_rsgangp_cond_50bs_cont2/iter_25000_lossG_1.0364_lossD_0.5982 --num_samples 200 --sent_weights $SENT_MODEL
