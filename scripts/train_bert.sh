#!/bin/zsh
CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='/home/yasu/cudacache' python3 -u main.py bert_uncased_small -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type bert_uncase_small -remove_el -remove_open -eval_period 1000 -save_period 1000000 -batch_size 32 -eval_batch_size 8 -bert | tee log/bert.log
