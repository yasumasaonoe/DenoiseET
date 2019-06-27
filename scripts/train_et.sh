#!/bin/zsh
CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='~/cudacache' python3 -u main.py et_model  -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type et_model -remove_el -remove_open -batch_size 100 | tee log/train_et.log
