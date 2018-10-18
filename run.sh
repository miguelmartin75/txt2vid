#!/bin/bash

OUT_NAME=${OUT_NAME:-output}
USE_NORMAL=${USE_NORMAL:-1}
RECON_LOSS=${RECON_LOSS:-0.1}

ROOT_DIR=${ROOT_DIR:-$FASTDIR/txt2vid/}
BASE_DIR=${BASE_DIR:-$FASTDIR/txt2vid/data/synthetic}
VIDEO_DIR=${VIDEO_DIR:-$BASE_DIR/train/videos}
ANNO=${ANNO:-$BASE_DIR/train/sent.pickle}
VOCAB=${VOCAB:-$BASE_DIR/vocab.pickle}
OUT_DIR=${OUT_DIR:-${FASTDIR}/${OUT_NAME}}
EXAMPLE_DIR=${EXAMPLE_DIR:-${ROOT_DIR}/${OUT_NAME}_samples}

if [[ $USE_NORMAL -eq 1 ]]; then
	echo "normal init"
	python3 txt2vid/train.py --data $VIDEO_DIR --anno $ANNO --vocab $VOCAB --workers 8 --batch_size 64 --epoch 20 --use_normal_init --recon_lambda $RECON_LOSS --out $OUT_DIR --out_samples $EXAMPLE_DIR --beta1 0.5 --lr 0.0002 --word_embed 20 --hidden_state 50 --txt_layers 2 --sent_encode 50 --latent_size 50 #--cuda
else
	echo "xavier init"
	python3 txt2vid/train.py --data $VIDEO_DIR --anno $ANNO --vocab $VOCAB --workers 8 --batch_size 64 --epoch 20 --recon_lambda $RECON_LOSS --out $OUT_DIR --out_samples $EXAMPLE_DIR --word_embed 20 --hidden_state 50 --txt_layers 2 --sent_encode 50 --latent_size 50 --beta1 0.5 --lr 0.0002 #--cuda
fi
