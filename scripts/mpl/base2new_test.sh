#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=MPL

DATASET=$1
SEED=$2
MAX_EPOCH=$3

CFG=vit_b4_c1_add2_rm1
SHOTS=16
LOADEP=$MAX_EPOCH
SUB=new


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}, skip this job."

else
    echo "Base to novel test, executed from ${PWD}"

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}

fi