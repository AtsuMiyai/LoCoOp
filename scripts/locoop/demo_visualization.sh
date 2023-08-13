#!/bin/bash

# custom config
TRAINER=LoCoOp

CSC=False
CTP=end

DATA=$1
DATASET=$2
CFG=$3

NCTX=16

MODEL_dir=$4
image_path=$5
label=$6

python demo_visualization.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/LoCoOp/${CFG}.yaml \
--model-dir ${MODEL_dir} \
--load-epoch 50 \
--image_path ${image_path} \
--label ${label} \
TRAINER.LOCOOP.N_CTX ${NCTX} \
TRAINER.LOCOOP.CSC ${CSC}