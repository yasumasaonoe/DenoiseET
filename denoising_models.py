"""
  This code is largely borrowed from Eunsol Choi's implementation.

  GitHub: https://github.com/uwnlp/open_type
  Paper : https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf

"""

import sys

import torch
import torch.nn as nn
import numpy as np
from models import ModelBase
from model_utils import sort_batch_by_length, SelfAttentiveSum, SimpleDecoder, MultiSimpleDecoder, CNN
from model_utils import  BCEWithLogitsLossCustom, ELMoWeightedSum, LinearProjection, MultiSimpleDecoderBinary
from model_utils import  RerankerAttentiveSum, TypeAttentiveSum, RNNDecoderState, RerankerSimpleScore, HighwayNetwork
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.insert(0, './resources')
import constant


class LabelerBase(ModelBase):

  def __init__(self, args, answer_num):
    super(LabelerBase, self).__init__(args, answer_num)
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
    self.hidden_dim = 300
    self.weighted_sum = ELMoWeightedSum()
    # init labeler params
    self.type_hid_dim = 1024
    self.lstm_label = nn.LSTM(self.embed_dim, self.type_hid_dim // 2, bidirectional=True, batch_first=True)
    self.type_attentive_sum = TypeAttentiveSum(self.output_dim, self.type_hid_dim)
    self.cls_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.]).cuda()) ########### scale positive side

  def define_cls_loss(self, logits, targets):
    loss = self.cls_loss_func(logits, targets)
    return loss

  def define_type_loss(self, logits, targets, cls_targets, data_type):
    loss = self.define_loss(logits, targets, data_type)
    return loss

  def sent_encoder(self, feed_dict, data_type):
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
    return output, attn_score

  def type_encoder(self, types_embed, lengths, sent_vec):
    hid = self.sorted_rnn(types_embed, lengths, self.lstm_label)
    types_embed, attn_score = self.type_attentive_sum(hid, sent_vec)
    return types_embed, attn_score

  def forward(self, feed_dict, data_type):
    pass


class Labeler(LabelerBase):

  def __init__(self, args, answer_num):
    super(Labeler, self).__init__(args, answer_num)
    self.type_embed_dim = 1024
    self.def_embed_dim = 1024
    if args.data_setup == 'joint' and args.multitask:
      print("Multi-task learning")
      self.decoder = MultiSimpleDecoder(self.output_dim + self.type_embed_dim + self.def_embed_dim)
    else:
      self.decoder = SimpleDecoder(self.output_dim + self.type_embed_dim + self.def_embed_dim, answer_num)
    self.type_vocab_size = constant.ANSWER_NUM_DICT['open'] + 3
    self.bos_idx = constant.TYPE_BOS_IDX
    self.eos_idx = constant.TYPE_EOS_IDX
    self.pad_idx = constant.TYPE_PAD_IDX
    self.def_vocab_size = constant.DEF_VOCAB_SIZE
    self.def_pad_idx = constant.DEF_PAD_IDX
    self.type_embedding = nn.Embedding(self.type_vocab_size, self.type_embed_dim, padding_idx=self.pad_idx)
    self.def_embedding = nn.Embedding(self.def_vocab_size, self.def_embed_dim, padding_idx=self.def_pad_idx)
    self.lstm_def = nn.LSTM(self.def_embed_dim, self.def_embed_dim // 2, bidirectional=True, batch_first=True)
    self.def_attentive_sum = TypeAttentiveSum(self.output_dim, self.type_hid_dim)

  def forward(self, feed_dict, data_type):
    sent_vec, sent_attn_score = self.sent_encoder(feed_dict, data_type)
    type_idx = feed_dict['y_noisy_idx']
    type_embed = self.type_embedding(type_idx)
    type_vec = torch.sum(type_embed, 1)
    def_idx = feed_dict['type_definition_idx']
    def_embed = self.def_embedding(def_idx)
    batch_size, max_n_types, max_def_len, def_dim = def_embed.size()
    def_vec = torch.zeros(batch_size, max_n_types, def_dim).cuda()
    for i in range(batch_size):
      n_types = feed_dict["y_noisy_lengths"][i]
      hid = self.sorted_rnn(def_embed[i, :n_types], feed_dict['type_definition_length'][i, :n_types], self.lstm_def)
      def_vec[i, :n_types, :] = hid[:, -1, :]
    def_vec = torch.sum(def_vec, 1)
    output = torch.cat((sent_vec, type_vec, def_vec), 1)
    logits = self.decoder(output, data_type)
    loss = self.define_loss(logits, feed_dict['y'], data_type)
    return loss, logits, None


class Filter(LabelerBase):

  def __init__(self, args, answer_num):
    super(Filter, self).__init__(args, answer_num)
    self.answer_num = answer_num
    self.type_embed_dim = 1024
    self.def_embed_dim = 1024
    self.vocab_size = constant.ANSWER_NUM_DICT['open'] + 3
    self.bos_idx = constant.TYPE_BOS_IDX
    self.eos_idx = constant.TYPE_EOS_IDX
    self.pad_idx = constant.TYPE_PAD_IDX
    self.type_embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_idx)
    self.def_vocab_size = constant.DEF_VOCAB_SIZE
    self.def_pad_idx = constant.DEF_PAD_IDX
    self.cls_decoder = nn.Linear(self.output_dim + self.embed_dim + self.def_embed_dim, 1, bias=True)
    self.def_embedding = nn.Embedding(self.def_vocab_size, self.def_embed_dim, padding_idx=self.def_pad_idx)
    self.lstm_def = nn.LSTM(self.def_embed_dim, self.def_embed_dim // 2, bidirectional=True, batch_first=True)
    self.def_attentive_sum = TypeAttentiveSum(self.output_dim, self.type_hid_dim)
    self.alpha = 1.
    self.relu = nn.ReLU()
    self.highway = HighwayNetwork(self.output_dim + self.embed_dim + self.def_embed_dim, 1, self.relu)

  def forward(self, feed_dict, data_type):
    sent_vec, sent_attn_score = self.sent_encoder(feed_dict, data_type)
    type_idx = feed_dict['y_noisy_idx']
    type_embed = self.type_embedding(type_idx)
    type_vec = torch.sum(type_embed, 1)
    def_idx = feed_dict['type_definition_idx']
    def_embed = self.def_embedding(def_idx)
    batch_size, max_n_types, max_def_len, def_dim = def_embed.size()
    def_vec = torch.zeros(batch_size, max_n_types, def_dim).cuda()
    for i in range(batch_size):
      n_types = feed_dict["y_noisy_lengths"][i]
      hid = self.sorted_rnn(def_embed[i, :n_types], feed_dict['type_definition_length'][i, :n_types], self.lstm_def)
      def_vec[i, :n_types, :] = hid[:, -1, :]
    def_vec = torch.sum(def_vec, 1)
    output = torch.cat((sent_vec, type_vec, def_vec), 1)
    cls_input = self.highway(output)
    cls_logits = self.cls_decoder(cls_input)
    cls_loss = self.define_cls_loss(cls_logits.squeeze(1), feed_dict['y_cls'])
    dummy_logits = -1. * torch.ones((cls_logits.size()[0], self.answer_num))
    dummy_logits[:, :3] = 1.
    return cls_loss, dummy_logits, cls_logits
