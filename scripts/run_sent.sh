#!/bin/bash

TXT_DATA=./msr/msr.pickle
VOCAB=./msr/msr.vocab

python3 txt2vid/train/txt.py --data $TXT_DATA --vocab $VOCAB --cuda --out /run/media/doubleu/Linux/honours/temp/sent_model_bi_sa --workers 3 --batch_size 64 --seed 1337 --teacher_force 0.5 --epoch 20 #--separate_decoder
