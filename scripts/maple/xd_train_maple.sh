#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/dataset/folder"
TRAINER=MaPLe

DATASET=$1
SEED=$2
MAX_EPOCH=$3

CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16

# output directory
DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}_epoch${MAX_EPOCH}_depth10

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    OPTIM.MAX_EPOCH ${MAX_EPOCH}

fi
