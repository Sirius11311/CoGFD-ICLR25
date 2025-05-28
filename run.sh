#!/bin/bash


# Define variables
COMBINE_CONCEPT="underage_and_alcohol"
COMBINE_THEME="normal_life"
SUB_CONCEPTS=("underage" "alcohol")
CUDA_DEVICES_1="0"
CUDA_DEVICES_2="0,1"

# image prepare
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES_1} python img_prepare.py \
  --concept_combination ${COMBINE_CONCEPT} 

# model finetune
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES_2} python concept_combination_erasing.py \
  --combine_concept_x ${COMBINE_CONCEPT} \
  --combine_theme_y ${COMBINE_THEME} \
  --p1 -1 \
  --p2 1 \
  --lr 2.5e-5 \
  --max-steps 130 \
  --only_xa \
  --iterate_n 2 \
  --start 990 \
  --end 1000 \
  --train_batch_size 20



# image generation
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES_1} python image_generation.py \
  --concept_combination ${COMBINE_CONCEPT} \
  --prompt_generate \
  --seed 188 \
  --generate_ref \
  --sub_concepts "${SUB_CONCEPTS[@]}" 

# erase evaluation
python erase_eval.py \
  --concept_combination ${COMBINE_CONCEPT}