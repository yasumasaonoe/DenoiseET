"""
  This code is largely borrowed from Eunsol Choi's implementation.

  GitHub: https://github.com/uwnlp/open_type
  Paper : https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf

"""

import copy
import glob
import json
import logging
import sys
import collections
from collections import defaultdict
from random import shuffle, randint, sample, random
from nltk.corpus import wordnet, stopwords

import numpy as np

sys.path.insert(0, './resources/')
import constant

sys.path.insert(0, './bert/')
import tokenization
from bert_utils import convert_sentence_and_mention_to_features, convert_sentence_to_features

import torch

from allennlp.commands.elmo import ElmoEmbedder


def to_torch(feed_dict):
  torch_feed_dict = {}
  annot_ids = None
  if 'annot_id' in feed_dict:
    annot_ids = feed_dict.pop('annot_id')
  for k, v in feed_dict.items():
    if 'embed' in k:
      torch_feed_dict[k] = torch.autograd.Variable(torch.from_numpy(v), requires_grad=False).cuda().float()
    elif 'token_bio' == k:
      torch_feed_dict[k] = torch.autograd.Variable(torch.from_numpy(v), requires_grad=False).cuda().float()
    elif 'y' == k or k == 'mention_start_ind' or k == 'mention_end_ind' or 'length' in k:
      torch_feed_dict[k] = torch.autograd.Variable(torch.from_numpy(v), requires_grad=False).cuda()
    elif k == 'span_chars':
      torch_feed_dict[k] = torch.autograd.Variable(torch.from_numpy(v), requires_grad=False).cuda()
    else:
      torch_feed_dict[k] = torch.from_numpy(v).cuda()
  return torch_feed_dict, annot_ids


def load_embedding_dict(embedding_path, embedding_size):
  print("Loading word embeddings from {}...".format(embedding_path))
  default_embedding = np.zeros(embedding_size)
  embedding_dict = defaultdict(lambda: default_embedding)
  with open(embedding_path) as f:
    for i, line in enumerate(f.readlines()):
      splits = line.split()
      if len(splits) != embedding_size + 1:
        continue
      assert len(splits) == embedding_size + 1
      word = splits[0]
      embedding = np.array([float(s) for s in splits[1:]])
      embedding_dict[word] = embedding
  print("Done loading word embeddings!")
  return embedding_dict


def get_vocab():
  """
  Get vocab file [word -> embedding]
  """
  char_vocab = constant.CHAR_DICT
  glove_word_vocab = load_embedding_dict(constant.GLOVE_VEC, 300)
  return char_vocab, glove_word_vocab


def get_type_elmo_vec():
  return None


def pad_slice(seq, seq_length, cut_left=False, pad_token="<none>"):
  if len(seq) >= seq_length:
    if not cut_left:
      return seq[:seq_length]
    else:
      output_seq = [x for x in seq if x != pad_token]
      if len(output_seq) >= seq_length:
        return output_seq[-seq_length:]
      else:
        return [pad_token] * (seq_length - len(output_seq)) + output_seq
  else:
    return seq + ([pad_token] * (seq_length - len(seq)))


def get_word_vec(word, vec_dict):
  if word in vec_dict:
    return vec_dict[word]
  return vec_dict['unk']


def init_elmo():
  print('Preparing ELMo...')
  print("Loading options from {}...".format(constant.ELMO_OPTIONS_FILE)) 
  print("Loading weith from {}...".format(constant.ELMO_WEIGHT_FILE)) 
  return ElmoEmbedder(constant.ELMO_OPTIONS_FILE, constant.ELMO_WEIGHT_FILE, cuda_device=0)


def get_elmo_vec(sentence, elmo):
  """ sentence must be a list of words """
  emb = elmo.embed_sentence(sentence)
  return emb # (3, len, dim)


def get_elmo_vec_batch(sentences, elmo):
  """ sentence must be a list of words """
  emb = elmo.embed_batch(sentences)
  return emb # (batch, 3, len, dim)


def init_bert(args, answer_num):
  return BertEmbedder(args, answer_num).cuda()


def get_bert_vec_batch(input_dict, bert):
  return bert(input_dict, None) # set data_type arg None, it's not used


def drop_types_randomly(type_idx, type_str, all_types):
  assert len(type_idx) == len(type_str)
  types = list(zip(type_idx, type_str))
  types_ = [tup for tup in types if tup[1]] # not in set(all_types[0:130])]    # gen=0:9, fine=9:130, finer=130:
  if len(types_) == 0 and len(types) > 0:
    return sample(types, 1)
  elif len(types_) == 1:
    return types_
  selected_types = []
  for i, s in types_:
    if random() > 0.7: ####################### th
      selected_types.append((i, s))
  if len(selected_types) > 0: 
    return selected_types
  else:
    return sample(types_, 1)


def get_example(generator, glove_dict, batch_size, answer_num,
                eval_data=False, simple_mention=True,
                elmo=None, bert=None, bert_tokenizer=None, finetune_bert=False,
                data_config=None, is_labeler=False, is_relabeling=False, type_elmo=None, all_types=None,
                use_type_definition=False):

  use_elmo_batch = True if elmo is not None else False ### use elmo batch
  #use_elmo_batch = True if not eval_data else False ### use elmo batch

  embed_dim = 300 if elmo is None else 1024
  cur_stream = [None] * batch_size
  no_more_data = False

  while True:
    bsz = batch_size
    seq_length = 25
    for i in range(batch_size):
      try:
        cur_stream[i] = list(next(generator))
      except StopIteration:
        no_more_data = True
        bsz = i
        break

    max_seq_length = min(50, max([len(elem[1]) + len(elem[2]) + len(elem[3]) for elem in cur_stream if elem]))
    token_embed = np.zeros([bsz, max_seq_length, embed_dim], np.float32)
    token_seq_length = np.zeros([bsz], np.float32)
    token_bio = np.zeros([bsz, max_seq_length, 4], np.float32)
    mention_start_ind = np.zeros([bsz, 1], np.int64)
    mention_end_ind = np.zeros([bsz, 1], np.int64)
    max_mention_length = min(20, max([len(elem[3]) for elem in cur_stream if elem]))
    max_span_chars = min(25, max(max([len(elem[5]) for elem in cur_stream if elem]), 5))
    max_n_target = max([len(elem[4][:]) for elem in cur_stream if elem])
    annot_ids = np.zeros([bsz], np.object)
    span_chars = np.zeros([bsz, max_span_chars], np.int64)
    mention_embed = np.zeros([bsz, max_mention_length, embed_dim], np.float32)
    targets = np.zeros([bsz, answer_num], np.float32)

    if is_labeler:
      if eval_data and is_relabeling:
        y_tups = [list(zip(elem[4], elem[7])) for elem in cur_stream if elem] # noised y
      else:
        y_tups = [list(zip(elem[9], elem[8])) for elem in cur_stream if elem] # original y
      y_noisy = [[t[1] for t in tup] for tup in y_tups]
      y_noisy_idx = [[t[0] for t in tup] for tup in y_tups]
      y_noisy_lengths = [len(yn) for yn in y_noisy]
      max_y_noisy = max(y_noisy_lengths)
      y_noisy_embed = np.zeros([bsz, max_y_noisy, embed_dim], np.float32) # assum  ELMo 
      y_noisy_lengths_np = np.array(y_noisy_lengths, np.int64)
      y_noisy_idx_np = np.ones([bsz, max_y_noisy], np.int64) * constant.TYPE_PAD_IDX
      y_cls = np.zeros([bsz], np.float32)
      if use_type_definition:
        max_def_length = max([len(x) for elem in cur_stream if elem for x in elem[13]])
        type_definition_idx = np.ones([bsz, max_y_noisy, max_def_length], np.int64) * constant.DEF_PAD_IDX
        type_definition_length = np.zeros([bsz, max_y_noisy], np.int64)

    mention_headword_embed = np.zeros([bsz, embed_dim], np.float32)
    mention_span_length = np.zeros([bsz], np.float32)

    if elmo is not None:
      token_embed = np.zeros([bsz, 3, max_seq_length, embed_dim], np.float32)
      mention_embed = np.zeros([bsz, 3, max_mention_length, embed_dim], np.float32)
      mention_headword_embed = np.zeros([bsz, 3, embed_dim], np.float32)
      elmo_mention_first = np.zeros([bsz, 3, embed_dim], np.float32)
      elmo_mention_last = np.zeros([bsz, 3, embed_dim], np.float32)
    if glove_dict is not None and elmo is not None and not is_labeler:
      token_embed = np.zeros([bsz, 4, max_seq_length, embed_dim], np.float32)
      mention_embed = np.zeros([bsz, 4, max_mention_length, embed_dim], np.float32)
      mention_headword_embed = np.zeros([bsz, 4, embed_dim], np.float32)
      elmo_mention_first = np.zeros([bsz, 4, embed_dim], np.float32)
      elmo_mention_last = np.zeros([bsz, 4, embed_dim], np.float32)
    if finetune_bert:
      bert_max_seq_length = 128
      bert_input_idx = np.zeros([bsz, bert_max_seq_length], np.int64)
      bert_token_type_idx = np.zeros([bsz, bert_max_seq_length], np.int64)
      bert_attention_mask = np.zeros([bsz, bert_max_seq_length], np.int64)
      bert_head_wordpiece_idx = np.zeros([bsz], np.int64)

    # Only Train: batch to ELMo embeddings
    # Will get CUDA memory error if batch size is large
    if use_elmo_batch:
      token_seqs = []
      keys = []
      for i in range(bsz):
        left_seq = cur_stream[i][1]
        if len(left_seq) > seq_length:
          left_seq = left_seq[-seq_length:]
        mention_seq = cur_stream[i][3]
        right_seq = cur_stream[i][2]
        token_seqs.append(left_seq + mention_seq + right_seq)
        keys.append(cur_stream[i][0])

      try:
        elmo_emb_batch = get_elmo_vec_batch(token_seqs, elmo) # (batch, 3, len, dim)
      except:
        print('ERROR:', bsz, token_seqs, cur_stream[i])
        raise

      if is_labeler:
        elmo_y_noisy_emb_batch = get_elmo_vec_batch(y_noisy, elmo)
        elmo_y_noisy_emb_batch = [x[0, :, :] for x in elmo_y_noisy_emb_batch]

    for i in range(bsz):
      left_seq = cur_stream[i][1]
      if len(left_seq) > seq_length:
        left_seq = left_seq[-seq_length:]
      mention_seq = cur_stream[i][3]
      annot_ids[i] = cur_stream[i][0]
      right_seq = cur_stream[i][2]
      mention_headword = cur_stream[i][6]

      token_seq = left_seq + mention_seq + right_seq
      mention_start_ind[i] = min(seq_length, len(left_seq))
      mention_end_ind[i] = min(49, len(left_seq) + len(mention_seq) - 1)
      mention_start_actual = len(left_seq)
      mention_end_actual = len(left_seq) + len(mention_seq) - 1
      if elmo is None and bert is None: # GLoVe or BERT
        if not finetune_bert: # GLoVe
          for j, word in enumerate(token_seq):
            if j < max_seq_length:
              token_embed[i, j, :embed_dim] = get_word_vec(word, glove_dict)
        else: # For BERT Ver.2
          input_idx, input_mask, segment_idx, _, _ = convert_sentence_and_mention_to_features(
              ' '.join(token_seq), ' '.join(mention_seq), bert_max_seq_length, bert_tokenizer
          )
          bert_input_idx[i, :] = input_idx
          bert_token_type_idx[i, :] = segment_idx
          bert_attention_mask[i, :] = input_mask
          bert_head_wordpiece_idx[i] = cur_stream[i][11] if is_labeler else cur_stream[i][8]
      elif elmo is not None and bert is None: # ELMo
        # sentence
        if use_elmo_batch: # Train
          elmo_emb = elmo_emb_batch[i] # (3, len, dim)
          if is_labeler:
            y_noisy_embed[i, :y_noisy_lengths[i], :] = elmo_y_noisy_emb_batch[i]
        else: # Eval
          elmo_emb = get_elmo_vec(token_seq, elmo)
          if is_labeler:
            y_noisy_embed[i, :y_noisy_lengths[i], :] = get_elmo_vec(y_noisy[i], elmo)[0, :, :]
        n_layers, seq_len, elmo_dim = elmo_emb.shape
        assert n_layers == 3, n_layers
        assert seq_len == len(token_seq), (seq_len, len(token_seq), token_seq, elmo_emb.shape)
        assert elmo_dim == embed_dim, (elmo_dim, embed_dim)
        if seq_len <= max_seq_length:
          token_embed[i, :n_layers, :seq_len, :] = elmo_emb
        else:
          token_embed[i, :n_layers, :, :] = elmo_emb[:, :max_seq_length, :] 
        # mention span
        start_ind = len(left_seq)
        end_ind = len(left_seq) + len(mention_seq) - 1     
        elmo_mention = elmo_emb[:, start_ind:end_ind+1, :]
        mention_len = end_ind - start_ind + 1
        assert mention_len == elmo_mention.shape[1] == len(mention_seq),(mention_len, elmo_mention.shape[1], len(mention_seq), mention_seq, elmo_mention.shape, token_seq, elmo_emb.shape) # (mention_len, elmo_mention.shape[0], len(mention_seq))
        if mention_len < max_mention_length: 
          mention_embed[i, :n_layers, :mention_len, :] = elmo_mention 
        else:
          mention_embed[i, :n_layers, :mention_len, :] = elmo_mention[:, :max_mention_length, :]
        # mention first & last words
        elmo_mention_first[i, :n_layers, :] = elmo_mention[:, 0, :]
        elmo_mention_last[i, :n_layers, :] = elmo_mention[:, -1, :]
        # headword
        try:
          headword_location = mention_seq.index(mention_headword)
        except:
          #print('WARNING: ' + mention_headword + ' / ' + ' '.join(mention_seq))
          # find the headword
          headword_location = 0
          headword_candidates = [i for i, word in enumerate(mention_seq) if mention_headword in word]
          if headword_candidates:
            headword_location = headword_candidates[0]
        mention_headword_embed[i, :n_layers, :] = elmo_mention[:, headword_location, :]
        # add 300d-GLoVe
        if glove_dict is not None and not is_labeler:
          # sentence
          for j, word in enumerate(token_seq):
            if j < max_seq_length:
              token_embed[i, 3, j, :300] = get_word_vec(word, glove_dict)
          # mention span
          for j, mention_word in enumerate(mention_seq):
            if j < max_mention_length:
              if simple_mention:
                mention_embed[i, 3, j, :300] = [k / len(cur_stream[i][3]) for k in
                                            get_word_vec(mention_word, glove_dict)]
              else:
                mention_embed[i, 3, j, :300] = get_word_vec(mention_word, glove_dict)
          # mention first & last words
          elmo_mention_first[i, 3, :300] = get_word_vec(mention_seq[0], glove_dict)
          elmo_mention_last[i, 3, :300] = get_word_vec(mention_seq[-1], glove_dict)
          # headword
          mention_headword_embed[i, 3, :300] = get_word_vec(mention_headword, glove_dict)
      for j, _ in enumerate(left_seq):
        token_bio[i, min(j, 49), 0] = 1.0  # token bio: 0(left) start(1) inside(2)  3(after)
      for j, _ in enumerate(right_seq):
        token_bio[i, min(j + len(mention_seq) + len(left_seq), 49), 3] = 1.0
      for j, _ in enumerate(mention_seq):
        if j == 0 and len(mention_seq) == 1:
          token_bio[i, min(j + len(left_seq), 49), 1] = 1.0
        else:
          token_bio[i, min(j + len(left_seq), 49), 2] = 1.0
      token_seq_length[i] = min(50, len(token_seq))

      if elmo is None and not finetune_bert:
        for j, mention_word in enumerate(mention_seq):
          if j < max_mention_length:
            if simple_mention:
              mention_embed[i, j, :embed_dim] = [k / len(cur_stream[i][3]) for k in
                                           get_word_vec(mention_word, glove_dict)]
            else:
              mention_embed[i, j, :embed_dim] = get_word_vec(mention_word, glove_dict)
      span_chars[i, :] = pad_slice(cur_stream[i][5], max_span_chars, pad_token=0)
      for answer_ind in cur_stream[i][4]:
        targets[i, answer_ind] = 1.0
      if is_labeler:
        y_noisy_idx_np[i, :len(y_noisy_idx[i])] = y_noisy_idx[i]
        y_cls[i] = float(cur_stream[i][10])
        if use_type_definition:
          for t_idx in range(len(cur_stream[i][13])):
            type_definition_idx[i, t_idx, :len(cur_stream[i][13][t_idx])] = cur_stream[i][13][t_idx]
            type_definition_length[i, t_idx] = len(cur_stream[i][13][t_idx])
      if elmo is None and not finetune_bert:
        mention_headword_embed[i, :embed_dim] = get_word_vec(mention_headword, glove_dict)
      mention_span_length[i] = min(len(mention_seq), 20)

    feed_dict = {"annot_id": annot_ids,
                 "mention_embed": mention_embed,
                 "span_chars": span_chars,
                 "y": targets,
                 "mention_headword_embed": mention_headword_embed,
                 "mention_span_length": mention_span_length}

    feed_dict["token_bio"] = token_bio
    feed_dict["token_embed"] = token_embed
    feed_dict["token_seq_length"] = token_seq_length
    feed_dict["mention_start_ind"] = mention_start_ind
    feed_dict["mention_end_ind"] = mention_end_ind
    if elmo is not None:
      feed_dict["mention_first"] = elmo_mention_first 
      feed_dict["mention_last"] = elmo_mention_last
    if bert is not None or finetune_bert:
      feed_dict["bert_input_idx"] = bert_input_idx
      feed_dict["bert_token_type_idx"] = bert_token_type_idx
      feed_dict["bert_attention_mask"] = bert_attention_mask
      feed_dict["bert_head_wordpiece_idx"] = bert_head_wordpiece_idx
    if is_labeler:
      feed_dict["y_noisy_embed"] = y_noisy_embed
      feed_dict["y_noisy_lengths"] = y_noisy_lengths_np 
      feed_dict["y_noisy_idx"] = y_noisy_idx_np
      feed_dict["y_cls"] = y_cls
      if use_type_definition:
        feed_dict["type_definition_idx"] = type_definition_idx
        feed_dict["type_definition_length"] = type_definition_length
    if no_more_data:
      if eval_data and bsz > 0:
        yield feed_dict
      break
    yield feed_dict


class TypeDataset(object):
  """Utility class type datasets"""

  def __init__(self, filepattern, vocab, goal, elmo=None, bert=None, args=None):
    self._all_shards = glob.glob(filepattern)
    self.goal = goal
    self.answer_num = constant.ANSWER_NUM_DICT[goal]
    shuffle(self._all_shards)
    self.char_vocab, self.glove_dict = vocab
    self.elmo = elmo
    self.bert = bert # BERT model obj
    self.finetune_bert = args.bert # True/False
    if args.model_type == 'bert_uncase_small':
      print('==> Init tokenizer from ' + constant.BERT_UNCASED_SMALL_VOCAB + ', do_lower_case=True')
      self.bert_tokenizer = tokenization.FullTokenizer(vocab_file=constant.BERT_UNCASED_SMALL_VOCAB, do_lower_case=True)
    else:
      self.bert_tokenizer = None
    if args.model_type in ['labeler', 'filter']:
      self.use_type_definition = True
    else:
      self.use_type_definition = False
    self.args = args
    self.is_labeler = True if args.mode in ['train_labeler', 'test_labeler'] else False
    self.is_relabeling = True if args.mode in ['test_labeler'] else False
    self.type_set = self._load_type_set() if self.is_labeler else None
    self.all_types = self._load_all_types() if self.is_labeler else None
    self.type_definition = constant.DEFINITION
    self.type_def_word2id = constant.DEF_VOCAB_S2I
    self.word2id = constant.ANS2ID_DICT[goal]
    print("Answer num %d" % (self.answer_num))
    print('Found %d shards at %s' % (len(self._all_shards), filepattern))
    logging.info('Found %d shards at %s' % (len(self._all_shards), filepattern))
    self.exclude_types = [
      'entity', 'object', 'whole', 'living_thing', 'organism', 'social_group', 'measure',
      'medimum_of_exchange', 'artifact', 'geographical_area'
    ]

  def _load_npz(self, path):
    with open(path, 'rb') as f:
      data = np.load(f)
    return data

  def _load_type_set(self, path='./resources/type_set.csv'):
    with open(path, 'r') as f:
      type_set = [line.strip().split(',') for line in f.readlines()]
    return type_set

  def _load_all_types(self, path='./resources/types.txt'):
    with open(path, 'r') as f:
      types = [line.strip() for line in f.readlines()]
    return types

  def _get_fake_labels(self, labels):
    if random() > 0.3:
      fake_labels = None
      fake_idx = list(range(len(self.type_set)))
      shuffle(fake_idx)
      for i in fake_idx:
        if len(set(labels).intersection(set(self.type_set[i]))) < 1:
          fake_labels = self.type_set[i]
          break
      if fake_labels:
        return fake_labels, 1
      else:
        return sample(self.type_set, 1)[0], 1
    else:
      return copy.deepcopy(labels), 0

  def get_synonyms(self, word):
    synonyms = []
    for syn in wordnet.synsets(word):
      for l in syn.lemmas():
        synonyms.append(l.name())
    return list(set(synonyms))

  def get_hypernyms(self, word):
    all_hyps = []
    if word.pos() == 'n':
      all_hyps.append(word)
    hyps = word.hypernyms()
    if hyps:
      all_hyps += self.flatten([self.get_hypernyms(h) for h in hyps if h.pos() == 'n'])
    return all_hyps

  def flatten(self, lst):
    for elem in lst:
      if isinstance(elem, collections.Iterable) and not isinstance(elem, (str, bytes)):
        yield from self.flatten(elem)
      else:
        yield elem

  def get_text_from_synset(self, syn):
    texts = []
    if syn.pos() == 'n':
      for l in syn.lemmas():
        name = l.name().lower()
        texts.append(name)
    return list(set(texts))

  def get_all_hypernyms(self, types):
    hyp = []
    for t in types:
      for s in wordnet.synsets(t): # choose the first synset 
        if s.pos() == 'n':
          hyp.append(self.get_hypernyms(s))
    return hyp

  def expand_types_wordnet(self, labels):
    hyp = [self.get_text_from_synset(synset) for synsets in self.get_all_hypernyms(labels) for synset in synsets]
    syn = [self.get_synonyms(t) for t in labels]
    hyp = [t for tt in hyp for t in tt]
    syn = [t for tt in syn for t in tt]
    return list(set([t for t in list(set(hyp + syn)) if t in self.all_types and t not in self.exclude_types] + labels))

  def drop_types_randomly(self, type_idx):
    if len(type_idx) == 1:
      return type_idx
    selected_types = []
    for s in type_idx:
      if random() > 0.3:
        selected_types.append(s)
    if len(selected_types) > 0:
      return selected_types
    else:
      return sample(type_idx, 1)

  def drop_coarse_types(self, type_idx, coarse_types):
    if len(type_idx) == 1:
      return type_idx
    selected_types = []
    for s in type_idx:
      if s not in set(coarse_types):
        selected_types.append(s)
    if len(selected_types) > 0:
      return selected_types
    else:
      return sample(type_idx, 1)

  def _load_shard(self, shard_name, eval_data):
    """Read one file and convert to ids.
    Args:
      shard_name: file path.
    Returns:
      list of (id, global_word_id) tuples.
    """
    with open(shard_name) as f:
      line_elems = [json.loads(sent.strip()) for sent in f.readlines()]
      # drop examples with empty mention span 
      filtered = []
      for line_elem in line_elems:
        if line_elem["mention_span"]:
          filtered.append(line_elem)
      # print(shard_name, ', before / after:', len(line_elems), '/', len(filtered))
      line_elems = filtered
      if not eval_data:
        line_elems = [line_elem for line_elem in line_elems if len(line_elem['mention_span'].split()) < 11]
      annot_ids = [line_elem["annot_id"] for line_elem in line_elems]
      mention_span = [[self.char_vocab[x] for x in list(line_elem["mention_span"])] for line_elem in line_elems]
      mention_seq = [line_elem["mention_span"].split() for line_elem in line_elems]
      mention_headword = [[w["text"] for w in line_elem["mention_span_tree"] if w["dep"] == "ROOT"][0] for line_elem in line_elems]
      left_seq = [line_elem['left_context_token'] for line_elem in line_elems]
      right_seq = [line_elem['right_context_token'] for line_elem in line_elems]
      y_str_list = [line_elem['y_str'] for line_elem in line_elems]
      head_wordpiece_idx = [0] * len(left_seq)
      if self.args.model_type == 'bert_uncase_small':
        for i, ls in enumerate(left_seq):
          if ' '.join(ls):
            head_wordpiece_idx[i] += len(self.bert_tokenizer.tokenize(' '.join(ls)))
          for w in line_elems[i]["mention_span_tree"]:
            if w["dep"] == "ROOT":
              break
            head_wordpiece_idx[i] += len(self.bert_tokenizer.tokenize(w["text"]))
    if self.is_labeler:
      y_ids = []
      for iid, y_strs in enumerate(y_str_list):
        y_ids.append([self.word2id[x] for x in y_strs if x in self.word2id])
      y_ids_noisy = []
      y_str_list_noisy_ = []
      y_ids_noisy_wordnet_expanded = []
      y_ids_noisy_def = []
      y_str_noisy_def = []
      noisy = [self._get_fake_labels(y) for y in y_str_list]
      if self.is_relabeling:
        y_str_list_noisy = [y[0] for y in noisy]
      else:
        #y_str_list_noisy = [line_elem['y_keep30_str'] for line_elem in line_elems]
        y_str_list_noisy = [y[0] for y in noisy]
      y_cls = [y[1] for y in noisy]
      for iid_n, y_strs_n in enumerate(y_str_list_noisy):
        # Adding noise during training
        if not self.is_relabeling:
          y_strs_n = self.drop_types_randomly(y_strs_n) # random 
          #y_strs_n = self.drop_coarse_types(y_strs_n, self.all_types[:constant.ANSWER_NUM_DICT['kb']]) # drop general types 
        y_str_list_noisy_.append(y_strs_n)
        y_ids_noisy.append([self.word2id[x] for x in y_strs_n if x in self.word2id])
        y_strs_wn = self.expand_types_wordnet(y_strs_n)
        y_ids_noisy_wordnet_expanded.append([self.word2id[x] for x in y_strs_wn if x in self.word2id])
        y_ids_noisy_def.append([[self.type_def_word2id[w] for w in self.type_definition[x]] for x in y_strs_n if x in self.word2id])
        y_str_noisy_def.append([[w for w in self.type_definition[x]] for x in y_strs_n if x in self.word2id])
      return zip(annot_ids, left_seq, right_seq, mention_seq, y_ids, mention_span, mention_headword, y_str_list, y_str_list_noisy_, y_ids_noisy, y_cls, head_wordpiece_idx, y_ids_noisy_wordnet_expanded, y_ids_noisy_def, y_str_noisy_def)
    else:
      y_ids = []
      for iid, y_strs in enumerate(y_str_list):
        y_ids.append([self.word2id[x] for x in y_strs if x in self.word2id])
      return zip(annot_ids, left_seq, right_seq, mention_seq, y_ids, mention_span, mention_headword, y_str_list, head_wordpiece_idx)

  def _get_sentence(self, epoch, forever, eval_data):
    for i in range(0, epoch if not forever else 100000000000000):
      for shard in self._all_shards:
        ids = self._load_shard(shard, eval_data)
        for current_ids in ids:
          yield current_ids

  def get_batch(self, batch_size=128, epoch=5, forever=False, eval_data=False, simple_mention=True):
    return get_example(self._get_sentence(epoch, forever=forever, eval_data=eval_data),
                       self.glove_dict,
                       batch_size=batch_size,
                       answer_num=self.answer_num,
                       eval_data=eval_data,
                       simple_mention=simple_mention,
                       elmo=self.elmo,
                       bert=self.bert,
                       finetune_bert=self.finetune_bert,
                       bert_tokenizer=self.bert_tokenizer,
                       is_labeler=self.is_labeler,
                       is_relabeling=self.is_relabeling,
                       all_types=self.all_types,
                       use_type_definition=self.use_type_definition
                      )


if __name__ == '__main__':
  # TEST
  sys.path.insert(0, './resources')
  import config_parser, constant

  args = config_parser.parser.parse_args()

  # elmo
  args.elmo = True
  args.model_type = 'labeler'
  vocab = (constant.CHAR_DICT, None) # dummy empty dict
  elmo = init_elmo()
  bert = None

  # bert
  #args.bert = True
  #args.model_type = 'bert_uncase_small'
  #vocab = (constant.CHAR_DICT, None) # dummy empty dict 
  #elmo = None
  #bert = None

  # glove
  #vocab = get_vocab()
  #elmo = None
  #bert = None

  dataset = TypeDataset(constant.FILE_ROOT + 'crowd/train_tree.json',
                        goal=args.goal, vocab=vocab, elmo=elmo, bert=bert,
                        args=args)
  data_gen = dataset.get_batch(128, 1, eval_data=False)
