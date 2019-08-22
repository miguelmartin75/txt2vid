#!/bin/bash

#TXT_DATA=./msr/msr.pickle
#VOCAB=./msr/msr.vocab
#TXT_DATA=${TXT_DATA:-./reddit_videos/captions.pickle}
#VOCAB=${VOCAB:-./reddit_videos/vocab.pickle}

TXT_DATA=${TXT_DATA:-/run/media/doubleu/Linux/synthetic/train/sent.pickle}
VOCAB=${VOCAB:-/run/media/doubleu/Linux/synthetic/train/vocab.pickle}

python txt2vid/data/__init__.py --sents $TXT_DATA --out $VOCAB

python3 txt2vid/train/txt.py --data $TXT_DATA --vocab $VOCAB --cuda --out /run/media/doubleu/Linux/synthetic_txt/ --workers 3 --batch_size 128 --seed 1337 --teacher_force 0.5 --epoch 50 #--separate_decoder
