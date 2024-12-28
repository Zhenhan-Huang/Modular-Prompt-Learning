#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
DATA="/data/zhenhan/datasets/pl_data"
TRAINER=MPL

DATASET=$1
SEED=$2
MAX_EPOCH=$3

CFG=vit_b4_c1_add2_rm1
SHOTS=16

DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job."
else
    echo "Base to new train, executed from ${PWD}"
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    OPTIM.MAX_EPOCH ${MAX_EPOCH} \
    DATASET.SUBSAMPLE_CLASSES base

fi
