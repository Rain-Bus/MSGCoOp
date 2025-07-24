#!/bin/bash

# custom config
DATA=~/Datasets/CoOp/
TRAINER=$1
N_PROMPTS=3
KG_WEIGHT=$2
MP_WEIGHT=$3
CFG=vit_b16_ep100_ctxv1 
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots 
CSC=False  # class-specific context (False or True)
SRC_DATASETS=imagenet
# LOADEP=5
LOADEP=100

for DATASET in dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf101 caltech101 sun397
do
    for SEED in 1 2 3
    do
        MODEL_DIR=output_xd/base2new/train_base/${SRC_DATASETS}/shots_${SHOTS}_${KG_WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
        DIR=output_xda/base2new/train_base/${DATASET}/shots_${SHOTS}_${KG_WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Skip this job"
        else
            echo "Run this job and save the output to ${DIR}"
            CUDA_VISIBLE_DEVICES=0 python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --eval-only \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.W ${KG_WEIGHT} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.COOP.N_PROMPTS ${N_PROMPTS} \
            TRAINER.COOP.DIV_WEIGHT ${MP_WEIGHT}
        fi
    done
done
