# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tokenization
import sys

sys.path.insert(0, '../resources/')
import constant


def convert_sentence_and_mention_to_features(sentence, mention, max_seq_length, tokenizer):
   
  sentence = tokenizer.tokenize(sentence)
  mention = tokenizer.tokenize(mention)
  _truncate_seq_pair(sentence, mention, max_seq_length - 3)

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0   0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambigiously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)

  idx_tracker = 0
  sentence_start_idx = 1

  for token in sentence:
    tokens.append(token)
    segment_ids.append(0)
    idx_tracker += 1
  
  sentence_end_idx = idx_tracker  
  tokens.append("[SEP]")
  segment_ids.append(0)
  idx_tracker += 1 
  mention_start_idx = idx_tracker + 1

  for token in mention:
    tokens.append(token)
    segment_ids.append(1)
    idx_tracker += 1

  mention_end_idx = idx_tracker 
  tokens.append("[SEP]")
  segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length, print(input_ids, len(input_ids), max_seq_length)
  assert len(input_mask) == max_seq_length, print(input_mask, len(input_mask), max_seq_length)
  assert len(segment_ids) == max_seq_length, print(segment_ids, len(segment_ids), max_seq_length)

  return input_ids, input_mask, segment_ids, (sentence_start_idx, sentence_end_idx), (mention_start_idx, mention_end_idx) # inclusive


def convert_sentence_to_features(sentence, max_seq_length, tokenizer):
   
  sentence = tokenizer.tokenize(sentence)
  sentence = sentence[0:(max_seq_length - 2)]
  
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)

  idx_tracker = 0
  sentence_start_idx = 1

  for token in sentence:
    tokens.append(token)
    segment_ids.append(0)
    idx_tracker += 1
  
  sentence_end_idx = idx_tracker  
  tokens.append("[SEP]")
  segment_ids.append(0)
  idx_tracker += 1 
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length, print(input_ids, len(input_ids), max_seq_length)
  assert len(input_mask) == max_seq_length, print(input_mask, len(input_mask), max_seq_length)
  assert len(segment_ids) == max_seq_length, print(segment_ids, len(segment_ids), max_seq_length)

  return input_ids, input_mask, segment_ids, (sentence_start_idx, sentence_end_idx), (None, None) # inclusive


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        tokens_a.pop() 
        #if len(tokens_a) > len(tokens_b):
        #    tokens_a.pop()
        #else:
        #    tokens_b.pop()


if __name__ == '__main__':
  
  # TEST
  tokenizer = tokenization.FullTokenizer(vocab_file=constant.BERT_UNCASED_SMALL_VOCAB, do_lower_case=True)
  examples = [('The year also marked a setback for gays seeking marriage equality .', 'The year'),
              ('But the Republicans may ultimately be able to tighten restrictions in some areas .', 'some areas'),
              ('Harmel served in every national government from 1950 to 1973 , and retired from political life in 1977 .', 'every national government')]
  for ex in examples:
    ids, mask, sent, sent_idx, mention_idx = convert_sentence_and_mention_to_features(*ex , 30, tokenizer) 
    print('input:', '[CLS] ' + ex[0] + ' [SEP] ' + ex[1] + ' [SEP]')
    print('input_ids:', ids)
    print('input_mask:', mask)
    print('segment_ids:', sent)
    print('sentence boundary', sent_idx)
    print('mention boundary', mention_idx)
    print()
    print()

  examples = ['The year also marked a setback for gays seeking marriage equality .',
              'But the Republicans may ultimately be able to tighten restrictions in some areas .',
              'Harmel served in every national government from 1950 to 1973 , and retired from political life in 1977 .']
  for ex in examples:
    ids, mask, sent, sent_idx, mention_idx = convert_sentence_to_features(ex , 30, tokenizer) 
    print('input:', '[CLS] ' + ex + ' [SEP]')
    print('input_ids:', ids)
    print('input_mask:', mask)
    print('segment_ids:', sent)
    print('sentence boundary', sent_idx)

