#!/bin/bash

WDIR=/data/zhenhan/ann_performance/prompt/nassearch/subnet/shots_16/TrainerSubnet/fewshot/dev

for DATASET in Caltech101 DescribableTextures EuroSAT FGVCAircraft Food101 OxfordFlowers OxfordPets StanfordCars UCF101 SUN397; do
    python3 parse_single.py \
        ${WDIR} \
        --nseeds 3 \
        --dataset ${DATASET} \
        --ci95
done

