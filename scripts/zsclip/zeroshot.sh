#!/bin/bash

#cd ../..

# custom config
DATA=/data/zhenhan/datasets/pl_data
TRAINER=ZeroshotCLIP
# caltech101, dtd, eurosat, fgvc_aircraft, food101, imagenet, oxford_flowers, oxford_pets, sun397, ucf101
DATASET=$1
# rn50, rn101, vit_b32 or vit_b16
CFG=$2  

#DIR=/data/zhenhan/ann_performance/prompt/${DATASET}/${TRAINER}/${CFG}_examine
DIR=output/prompt/${DATASET}/${TRAINER}/${CFG}_examine

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=3 python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir $DIR \
--eval-only