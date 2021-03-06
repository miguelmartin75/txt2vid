#!/bin/bash

OUT_NAME=${OUT_NAME:-output}
USE_NORMAL=${USE_NORMAL:-0}
RECON_LOSS=${RECON_LOSS:-0.0}

ROOT_DIR=${ROOT_DIR:-./txt2vid_2/}
BASE_DIR=${BASE_DIR:-./cooking_videos/train}
VIDEO_DIR=${VIDEO_DIR:-$BASE_DIR}
ANNO=${ANNO:-./msr/msr.pickle}
VOCAB=${VOCAB:-./msr/msr.vocab}
OUT_DIR=${OUT_DIR:-/run/media/doubleu/Linux/honours/temp/${OUT_NAME}}
EXAMPLE_DIR=${EXAMPLE_DIR:-/run/media/doubleu/Linux/honours/temp/${OUT_NAME}_samples_old}

# TODO: change G and D to be json files
#python3 txt2vid/train/gan.py --data $VIDEO_DIR --anno $ANNO --workers 8 --batch_size 32 --epochs 100 --out $OUT_DIR --out_samples $EXAMPLE_DIR --num_channels 3 --vocab $VOCAB --cuda --G txt2vid.models.tgan.gen.Gen --D txt2vid.models.tgan.discrim.Discrim --sent txt2vid.models.txt.basic.SentenceEncoder --end2end
python3 txt2vid/train/gan.py --data $VIDEO_DIR --anno $ANNO --workers 4 --batch_size 48 --epochs 100 --out $OUT_DIR --out_samples $EXAMPLE_DIR --num_channels 3 --vocab $VOCAB --cuda --G txt2vid.models.tcwyt.gen.Gen --D txt2vid.models.tcwyt.video_discrim.VideoDiscrim txt2vid.models.tcwyt.frame_discrim.FrameDiscrim txt2vid.models.tcwyt.motion_discrim.MotionDiscrim --sent txt2vid.models.txt.basic.Seq2Seq --sent_weights msr/sent.pth --frame_size 48 --M txt2vid.models.tcwyt.frame_discrim.FrameMap --D_names video frame motion --G_lr 0.0001 --D_lr 0.0001 --D_beta1 0.9 --D_beta2 .999 --G_beta1 0.9 --G_beta2 .999 --D_loss txt2vid.gan.losses.RaLSGANLoss --init_method xavier --discrim_steps 1
#python3 txt2vid/train/gan.py --data $VIDEO_DIR --anno $ANNO --workers 4 --batch_size 50 --epochs 100 --out $OUT_DIR --out_samples $EXAMPLE_DIR --num_channels 3 --vocab $VOCAB --cuda --G txt2vid.models.tcwyt.gen.Gen --D txt2vid.models.tcwyt.video_discrim.VideoDiscrim --sent txt2vid.models.txt.basic.Seq2Seq --sent_weights msr/sent.pth --frame_size 48 --D_names video --G_lr 0.00005 --D_lr 0.00001 --D_loss txt2vid.gan.losses.RaLSGANLoss
