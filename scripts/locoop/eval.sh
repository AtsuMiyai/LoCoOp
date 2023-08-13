#!/bin/bash

# custom config
TRAINER=LoCoOp

CSC=False
CTP=end

DATA=$1
DATASET=$2
CFG=$3

NCTX=16

T=$4
MODEL_dir=$5
Output_dir=$5

python eval_ood_detection.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/LoCoOp/${CFG}.yaml \
--output-dir ${Output_dir} \
--model-dir ${MODEL_dir} \
--load-epoch 50 \
--T ${T} \
TRAINER.LOCOOP.N_CTX ${NCTX} \
TRAINER.LOCOOP.CSC ${CSC} \