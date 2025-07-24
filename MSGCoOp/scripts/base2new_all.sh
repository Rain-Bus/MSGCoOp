#!/bin/bash

TRAINER=$1
KG_WEIGHT=$2
MP_WEIGHT=$3

CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_train.sh ${TRAINER} ucf101 ${KG_WEIGHT} ${MP_WEIGHT}
CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_test.sh ${TRAINER} ucf101 ${KG_WEIGHT} ${MP_WEIGHT}

CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_train.sh ${TRAINER} eurosat ${KG_WEIGHT} ${MP_WEIGHT}
CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_test.sh ${TRAINER} eurosat ${KG_WEIGHT} ${MP_WEIGHT}

CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_train.sh ${TRAINER} oxford_pets ${KG_WEIGHT} ${MP_WEIGHT}
CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_test.sh ${TRAINER} oxford_pets ${KG_WEIGHT} ${MP_WEIGHT}

CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_train.sh ${TRAINER} food101 ${KG_WEIGHT} ${MP_WEIGHT}
CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_test.sh ${TRAINER} food101 ${KG_WEIGHT} ${MP_WEIGHT}

CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_train.sh ${TRAINER} oxford_flowers ${KG_WEIGHT} ${MP_WEIGHT}
CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_test.sh ${TRAINER} oxford_flowers ${KG_WEIGHT} ${MP_WEIGHT}

CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_train.sh ${TRAINER} dtd ${KG_WEIGHT} ${MP_WEIGHT}
CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_test.sh ${TRAINER} dtd ${KG_WEIGHT} ${MP_WEIGHT}

CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_train.sh ${TRAINER} caltech101 ${KG_WEIGHT} ${MP_WEIGHT}
CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_test.sh ${TRAINER} caltech101 ${KG_WEIGHT} ${MP_WEIGHT}

CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_train.sh ${TRAINER} fgvc_aircraft ${KG_WEIGHT} ${MP_WEIGHT}
CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_test.sh ${TRAINER} fgvc_aircraft ${KG_WEIGHT} ${MP_WEIGHT}

CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_train.sh ${TRAINER} stanford_cars ${KG_WEIGHT} ${MP_WEIGHT}
CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_test.sh ${TRAINER} stanford_cars ${KG_WEIGHT} ${MP_WEIGHT}

CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_train.sh ${TRAINER} sun397 ${KG_WEIGHT} ${MP_WEIGHT}
CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_test.sh ${TRAINER} sun397 ${KG_WEIGHT} ${MP_WEIGHT}

CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_train.sh ${TRAINER} imagenet ${KG_WEIGHT} ${MP_WEIGHT}
CUDA_VISIBLE_DEVICES=0 bash scripts/base2new_test.sh ${TRAINER} imagenet ${KG_WEIGHT} ${MP_WEIGHT}
