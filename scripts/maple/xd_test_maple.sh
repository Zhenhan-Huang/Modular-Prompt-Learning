#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/dataset/folder"
TRAINER=MaPLe

DATASET=$1
SEED=$2

CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16


DIR=ouput/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
CKPT=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir $CKPT \
    --load-epoch 2 \
    --eval-only
fi