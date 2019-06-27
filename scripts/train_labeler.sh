#!/bin/zsh
CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='/home/yasu/cudacache' python3 -u main.py labeler -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type labeler -batch_size 50 -remove_el -remove_open -mode train_labeler -eval_period 100 -num_epoch 1000 -eval_batch_size 50  | tee log/labeler.log
