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
EXAMPLE_DIR=${EXAMPLE_DIR:-/run/media/doubleu/Linux/honours/temp/${OUT_NAME}_samples_2}

#echo python3 txt2vid/train.py --data $VIDEO_DIR --anno $ANNO --workers 8 --batch_size 64 --epoch 100 --use_normal_init ${USE_NORMAL} --recon_lambda $RECON_LOSS --out $OUT_DIR --out_samples $EXAMPLE_DIR --beta1 0.5 --lr 0.0002 --sent_encode_path $FASTDIR/sent.pth --num_channels 3 --vocab $VOCAB --cuda
python3 txt2vid/train.py --data $VIDEO_DIR --anno $ANNO --workers 8 --batch_size 64 --epoch 100 --use_normal_init ${USE_NORMAL} --recon_lambda $RECON_LOSS --out $OUT_DIR --out_samples $EXAMPLE_DIR --beta1 0.1 --beta2 0.9 --lr 0.0001 --sent_encode_path msr/sent.pth --num_channels 3 --vocab $VOCAB --cuda
