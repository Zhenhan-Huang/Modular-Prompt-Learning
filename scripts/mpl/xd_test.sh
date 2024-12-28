#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=MPL

DATASET=$1
SEED=$2
MAX_EPOCH=2

CFG=vit_b4_c1_add2_rm1
SHOTS=16


DIR=output/crossdataset/evaluation/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
CKPT=output/crossdataset/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job."
else
    echo "Run this job and save the output to ${DIR}"

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${CKPT} \
    --load-epoch ${MAX_EPOCH} \
    --eval-only \
    OPTIM.MAX_EPOCH ${MAX_EPOCH}

fi