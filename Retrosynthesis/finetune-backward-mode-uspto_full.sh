#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
DIR="../Retrosynthesis"


python -m molbart.fine_tune \
  datamodule=[molbart.data.seq2seq_data.Uspto50DataModule] \
  data_path=${DIR}/data/uspto_full.pkl \
  model_path=${DIR}/ckpt/last.ckpt \
  task=backward_prediction \
  n_epochs=10 \
  learning_rate=0.001 \
  schedule=cycle \
  batch_size=64 \
  acc_batches=4 \
  augmentation_strategy=all \
  augmentation_probability=0.5

ckpt_path=tb_logs/backward_prediction/version_0/checkpoints
ckpt_from=${ckpt_path}/last.ckpt
ckpt_to=${DIR}/model/finetune_USPTO_FULL.ckpt
echo cp $ckpt_from to $ckpt_to
cp "${ckpt_from}" "${ckpt_to}"

python -m molbart.fine_tune \
  datamodule=[molbart.data.seq2seq_data.Uspto50DataModule] \
  data_path=${DIR}/data/bioformer_wo.pkl \
  model_path=${DIR}/chemformer_models/finetune/finetune_USPTO_FULL.ckpt \
  task=backward_prediction \
  n_epochs=60 \
  learning_rate=0.001 \
  schedule=cycle \
  batch_size=64 \
  acc_batches=4 \
  augmentation_strategy=all \
  augmentation_probability=0.5

ckpt_path=tb_logs/backward_prediction/version_1/checkpoints
ckpt_from=${ckpt_path}/last.ckpt
ckpt_to=${DIR}/chemformer_models/finetune/last_USPTO_FULL.ckpt
echo cp $ckpt_from to $ckpt_to
cp "${ckpt_from}" "${ckpt_to}"

python -m molbart.predict \
  data_path=${DIR}/data/bioformer_test_tgt_wo_aug.txt \
  vocabulary_path=bart_vocab_downstream.json \
  task=backward_prediction \
  model_path=${DIR}/chemformer_models/finetune/last_USPTO_FULL.ckpt \
  output_sampled_smiles=${DIR}/outputs/predicted_reactants_mode2.csv
  batch_size=64 \
  n_beams=10