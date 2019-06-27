"""
  This code is largely borrowed from Eunsol Choi's implementation.

  GitHub: https://github.com/uwnlp/open_type
  Paper : https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf

"""

import sys

import torch
import torch.nn as nn
import numpy as np
from model_utils import sort_batch_by_length, SelfAttentiveSum, SimpleDecoder, MultiSimpleDecoder, CNN, BCEWithLogitsLossCustom, ELMoWeightedSum, LinearProjection, MultiSimpleDecoderBinary, RNNDecoderState
from model_utils import ConditionedAttentiveSum, StructuredPerceptronLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.insert(0, './resources')
import constant

sys.path.insert(0, './bert')
from modeling import *

class ModelBase(nn.Module):
  def __init__(self, args, answer_num):
    super(ModelBase, self).__init__()

    self.multitask = args.multitask
    self.loss_func = nn.BCEWithLogitsLoss()

  def sorted_rnn(self, sequences, sequence_lengths, rnn):
    sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(sequences, sequence_lengths)
    packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                 sorted_sequence_lengths.data.long().tolist(),
                                                 batch_first=True)
    packed_sequence_output, _ = rnn(packed_sequence_input, None)
    unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)
    return unpacked_sequence_tensor.index_select(0, restoration_indices)

  def rnn(self, sequences, lstm):
    outputs, _ = lstm(sequences)
    return outputs.contiguous()

  def define_loss(self, logits, targets, data_type):
    if not self.multitask or data_type == 'onto':
      loss = self.loss_func(logits, targets)
      return loss
    if data_type == 'wiki':
      gen_cutoff, fine_cutoff, final_cutoff = constant.ANSWER_NUM_DICT['gen'], constant.ANSWER_NUM_DICT['kb'], \
                                              constant.ANSWER_NUM_DICT[data_type]
    else:
      gen_cutoff, fine_cutoff, final_cutoff = constant.ANSWER_NUM_DICT['gen'], constant.ANSWER_NUM_DICT['kb'], None
    loss = 0.0
    comparison_tensor = torch.Tensor([1.0]).cuda()
    gen_targets = targets[:, :gen_cutoff]
    fine_targets = targets[:, gen_cutoff:fine_cutoff]
    gen_target_sum = torch.sum(gen_targets, 1)
    fine_target_sum = torch.sum(fine_targets, 1)

    if torch.sum(gen_target_sum.data) > 0:
      gen_mask = torch.squeeze(torch.nonzero(torch.min(gen_target_sum.data, comparison_tensor)), dim=1)
      gen_logit_masked = logits[:, :gen_cutoff][gen_mask, :]
      gen_mask = torch.autograd.Variable(gen_mask).cuda()
      gen_target_masked = gen_targets.index_select(0, gen_mask)
      gen_loss = self.loss_func(gen_logit_masked, gen_target_masked)
      loss += gen_loss
    if torch.sum(fine_target_sum.data) > 0:
      fine_mask = torch.squeeze(torch.nonzero(torch.min(fine_target_sum.data, comparison_tensor)), dim=1)
      fine_logit_masked = logits[:,gen_cutoff:fine_cutoff][fine_mask, :]
      fine_mask = torch.autograd.Variable(fine_mask).cuda()
      fine_target_masked = fine_targets.index_select(0, fine_mask)
      fine_loss = self.loss_func(fine_logit_masked, fine_target_masked)
      loss += fine_loss

    if not data_type == 'kb':
      if final_cutoff:
        finer_targets = targets[:, fine_cutoff:final_cutoff]
        logit_masked = logits[:, fine_cutoff:final_cutoff]
      else:
        logit_masked = logits[:, fine_cutoff:]
        finer_targets = targets[:, fine_cutoff:]
      if torch.sum(torch.sum(finer_targets, 1).data) >0:
        finer_mask = torch.squeeze(torch.nonzero(torch.min(torch.sum(finer_targets, 1).data, comparison_tensor)), dim=1)
        finer_mask =  torch.autograd.Variable(finer_mask).cuda()
        finer_target_masked = finer_targets.index_select(0, finer_mask)
        logit_masked = logit_masked[finer_mask, :]
        layer_loss = self.loss_func(logit_masked, finer_target_masked)
        loss += layer_loss
    return loss

  def forward(self, feed_dict, data_type):
    pass


class ETModel(ModelBase):
  def __init__(self, args, answer_num):
    super(ETModel, self).__init__(args, answer_num)
    self.output_dim = args.rnn_dim * 2
    self.mention_dropout = nn.Dropout(args.mention_dropout)
    self.input_dropout = nn.Dropout(args.input_dropout)
    self.dim_hidden = args.dim_hidden
    self.embed_dim = 1024
    self.mention_dim = 1024
    self.headword_dim = 1024
    self.enhanced_mention = args.enhanced_mention

    self.add_headword_emb = args.add_headword_emb
    self.mention_lstm = args.mention_lstm

    if args.enhanced_mention:
      self.head_attentive_sum = SelfAttentiveSum(self.mention_dim, 1)
      self.cnn = CNN()
      self.mention_dim += 50
    self.output_dim += self.mention_dim 

    if self.add_headword_emb:
      self.output_dim += self.headword_dim

    # Defining LSTM here.  
    self.attentive_sum = SelfAttentiveSum(args.rnn_dim * 2, 100)
    self.lstm = nn.LSTM(self.embed_dim + 50, args.rnn_dim, bidirectional=True,
                        batch_first=True)
    self.token_mask = nn.Linear(4, 50)

    if self.mention_lstm:
      self.lstm_mention = nn.LSTM(self.embed_dim, self.embed_dim // 2, bidirectional=True,
                                  batch_first=True)
      self.mention_attentive_sum = SelfAttentiveSum(self.embed_dim, 1)

    self.sigmoid_fn = nn.Sigmoid()
    self.goal = args.goal

    if args.data_setup == 'joint' and args.multitask:
      print("Multi-task learning")
      self.decoder = MultiSimpleDecoder(self.output_dim)
    else:
      self.decoder = SimpleDecoder(self.output_dim, answer_num)

    self.weighted_sum = ELMoWeightedSum()

  def forward(self, feed_dict, data_type):
    token_embed = self.weighted_sum(feed_dict['token_embed'])
    token_mask_embed = self.token_mask(feed_dict['token_bio'].view(-1, 4))
    token_mask_embed = token_mask_embed.view(token_embed.size()[0], -1, 50) # location embedding
    token_embed = torch.cat((token_embed, token_mask_embed), 2)
    token_embed = self.input_dropout(token_embed)
    context_rep = self.sorted_rnn(token_embed, feed_dict['token_seq_length'], self.lstm)
    context_rep, _ = self.attentive_sum(context_rep)
    # Mention Representation
    mention_embed = self.weighted_sum(feed_dict['mention_embed'])
    if self.enhanced_mention:
      if self.mention_lstm:
        mention_hid = self.sorted_rnn(mention_embed, feed_dict['mention_span_length'], self.lstm_mention)
        mention_embed, attn_score = self.mention_attentive_sum(mention_hid)
      else:
        mention_embed, attn_score = self.head_attentive_sum(mention_embed)
      span_cnn_embed = self.cnn(feed_dict['span_chars'])
      mention_embed = torch.cat((span_cnn_embed, mention_embed), 1)
    else:
      mention_embed = torch.sum(mention_embed, dim=1)
    mention_embed = self.mention_dropout(mention_embed) 

    if self.add_headword_emb:
      mention_headword_embed = self.weighted_sum(feed_dict['mention_headword_embed'])
      output = torch.cat((context_rep, mention_embed, mention_headword_embed), 1) # + Headword lstm emb 
    else:
      output = torch.cat((context_rep, mention_embed), 1)

    logits = self.decoder(output, data_type)
    loss = self.define_loss(logits, feed_dict['y'], data_type)
    return loss, logits, attn_score


class Bert(ModelBase):
  """ Using the [CLS] vec with a linear decoder """
  def __init__(self, args, answer_num):
    super(Bert, self).__init__(args, answer_num)

    # --- BERT ---
    if args.model_type == 'bert_uncase_small':
      print('==> Loading BERT config from ' + constant.BERT_UNCASED_SMALL_CONFIG)
      self.bert_config = BertConfig.from_json_file(constant.BERT_UNCASED_SMALL_CONFIG)
    else:
      raise NotImplementedError
    self.bert = BertModel(self.bert_config)
    self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)

    if args.data_setup == 'joint' and args.multitask:
      print("Multi-task learning")
      self.decoder = MultiSimpleDecoder(self.bert_config.hidden_size)
    else:
      self.decoder = SimpleDecoder(self.bert_config.hidden_size, answer_num)

  def forward(self, feed_dict, data_type):
    _, pooled_output = self.bert(feed_dict['bert_input_idx'], feed_dict['bert_token_type_idx'], feed_dict['bert_attention_mask'])
    pooled_output = self.dropout(pooled_output)
    logits = self.decoder(pooled_output, data_type)
    loss = self.define_loss(logits, feed_dict['y'], data_type)
    return loss, logits, None
