#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
DIR="../Retrosynthesis"


python -m molbart.predict \
  data_path=${DIR}/data/bioformer_test_tgt_wo_aug.txt \
  vocabulary_path=bart_vocab_downstream.json \
  task=backward_prediction \
  model_path=${DIR}/ckpt/last_USPTO_FULL.ckpt \
  output_sampled_smiles=${DIR}/outputs/predicted_reactants_mode2.csv
  batch_size=64 \
  n_beams=10