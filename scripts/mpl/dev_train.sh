#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/dataset/folder"
# MaPLe, ZSClip
TRAINER=MPL

DATASET=$1
SEED=$2
MAX_EPOCH=$3

CFG=vit_b4_c1_add2_rm1
SHOTS=16

# output directory
DIR=output/${DATASET}/${TRAINER}/eval/${CFG}_${SHOTS}shots/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}_dev.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    OPTIM.MAX_EPOCH ${MAX_EPOCH}

fi
