from collections import namedtuple, defaultdict


def load_vocab_dict(vocab_file_name, vocab_max_size=None, start_vocab_count=None):
  with open(vocab_file_name) as f:
    text = [x.strip() for x in f.readlines()]
    if vocab_max_size:
      text = text[:vocab_max_size]
    if start_vocab_count:
      file_content = dict(zip(text, range(0 + start_vocab_count, len(text) + start_vocab_count)))
    else:
      file_content = dict(zip(text, range(0, len(text))))
  return file_content


def load_definition_dict(path):
  with open(path, 'r') as f:
    definition = [[y.strip() for y in x.strip().split('<sep>')] for x in f.readlines()]
    definition = {k:v.strip().split() for k,v in definition}
  return definition


def get_definition_vocab(def_dict):
  counts = {}
  for _, v in def_dict.items():
    for word in v:
      if word not in counts:
        counts[word] = 0
      counts[word] += 1
  vocab = {'<unk>': 0, '<pad>': 1}
  idx = 2
  for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    vocab[k] = idx
    idx += 1
  return vocab


BASE_PATH = '.' # 

FILE_ROOT = BASE_PATH + '/data/'
GLOVE_VEC = BASE_PATH + '/data/pretrained_vector/glove.840B.300d.txt'
ELMO_VEC = BASE_PATH + '/data/pretrained_vector/type_elmo.npz'
EXP_ROOT = BASE_PATH + '/model'

# --- BERT ---
BERT_ROOT = FILE_ROOT + '/bert/'

BERT_UNCASED_SMALL_ROOT = BERT_ROOT + 'uncased_L-12_H-768_A-12/'
BERT_UNCASED_SMALL_CONFIG = BERT_UNCASED_SMALL_ROOT + 'bert_config.json'
BERT_UNCASED_SMALL_MODEL = BERT_UNCASED_SMALL_ROOT + 'pytorch_model.bin'
BERT_UNCASED_SMALL_VOCAB = BERT_UNCASED_SMALL_ROOT + 'vocab.txt'

# --- Definition ---
DEFINITION = load_definition_dict(FILE_ROOT + '/ontology/types_definition.txt') 
DEF_VOCAB_S2I = get_definition_vocab(DEFINITION)
DEF_VOCAB_I2S = {v: k for k, v in DEF_VOCAB_S2I.items()}
DEF_VOCAB_SIZE = len(DEF_VOCAB_S2I)  # 10473
DEF_PAD_IDX = DEF_VOCAB_S2I['<pad>'] # 1

# ------------------

ANSWER_NUM_DICT = {"open": 10331, "onto":89, "wiki": 4600, "kb":130, "gen":9}

KB_VOCAB = load_vocab_dict(FILE_ROOT + "/ontology/types.txt", 130)
WIKI_VOCAB = load_vocab_dict(FILE_ROOT + "/ontology/types.txt", 4600)
ANSWER_VOCAB = load_vocab_dict(FILE_ROOT + "/ontology/types.txt")
ONTO_ANS_VOCAB = load_vocab_dict(FILE_ROOT + '/ontology/onto_ontology.txt')
ANS2ID_DICT = {"open": ANSWER_VOCAB, "wiki": WIKI_VOCAB, "kb": KB_VOCAB, "onto":ONTO_ANS_VOCAB}

ELMO_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

TYPE_BOS_IDX = 10332
TYPE_EOS_IDX = 10331
TYPE_PAD_IDX = 10333

open_id2ans = {v: k for k, v in ANSWER_VOCAB.items()}
wiki_id2ans = {v: k for k, v in WIKI_VOCAB.items()}
kb_id2ans = {v:k for k,v in KB_VOCAB.items()}
g_id2ans = {v: k for k, v in ONTO_ANS_VOCAB.items()}

ID2ANS_DICT = {"open": open_id2ans, "wiki": wiki_id2ans, "kb": kb_id2ans, "onto":g_id2ans}
label_string = namedtuple("label_types", ["head", "wiki", "kb"])
LABEL = label_string("HEAD", "WIKI", "KB")

CHAR_DICT = defaultdict(int)
char_vocab = [u"<unk>"]
with open(FILE_ROOT + "/ontology/char_vocab.english.txt") as f:
  char_vocab.extend(c.strip() for c in f.readlines())
  CHAR_DICT.update({c: i for i, c in enumerate(char_vocab)})
