"""
  This code is largely borrowed from Eunsol Choi's implementation.

  GitHub: https://github.com/uwnlp/open_type
  Paper : https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf

"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model_id", help="Identifier for model")
# Data
parser.add_argument("-train_data", help="Train data", default="ontonotes/augmented_train_tree.json")
parser.add_argument("-dev_data", help="Dev data", default="ontonotes/g_dev_tree.json")
parser.add_argument("-eval_data", help="Test data", default="ontonotes/g_test_tree.json")
parser.add_argument("-num_epoch", help="The number of epoch", default=5000, type=int)
parser.add_argument("-batch_size", help="The batch size", default=100, type=int)
parser.add_argument("-eval_batch_size", help="The batch size", default=100, type=int)
parser.add_argument("-goal", help="Limiting vocab to smaller vocabs (either ontonote or figer)", default="open",
                    choices=["open", "onto", "wiki", 'kb'])
parser.add_argument("-seed", help="Pytorch random Seed", default=1888)
parser.add_argument("-gpu", help="Using gpu or cpu", default=False, action="store_true")

# learning
parser.add_argument("-mode", help="Whether to train or test", default="train", choices=["train", "test", "train_labeler", "test_labeler"])
parser.add_argument("-learning_rate", help="start learning rate", default=0.001, type=float)
parser.add_argument("-mention_dropout", help="drop out rate for mention", default=0.5, type=float)
parser.add_argument("-input_dropout", help="drop out rate for sentence", default=0.2, type=float)
parser.add_argument("-bert_learning_rate", help="BERT: start learning rate", default=2e-5, type=float)
parser.add_argument("-bert_warmup_proportion", help="Proportion of training to perform linear learning rate warmup for.", default=0.1, type=float)

# Data ablation study
parser.add_argument("-add_crowd", help="Add indomain data as train", default=False, action='store_true')
parser.add_argument("-data_setup", help="Whether to use joint data set-up", default="single", choices=["single", "joint"])
parser.add_argument("-only_crowd", help="Only using indomain data as train", default=False, action='store_true')
parser.add_argument("-remove_el", help="Remove supervision from entity linking", default=False, action='store_true')
parser.add_argument("-remove_open", help="Remove supervision from headwords", default=False, action='store_true')
parser.add_argument("-add_expanded_head", help="Add expanded label data as train", default=False, action='store_true')
parser.add_argument("-add_expanded_el", help="Add expanded label data as train", default=False, action='store_true')

# Model
parser.add_argument("-model_type", default="et_model", choices=["et_model", "labeler", "filter", "bert_uncase_small"])
parser.add_argument("-multitask", help="Using a multitask loss term.", default=False, action='store_true')
parser.add_argument("-enhanced_mention", help="Use attention and cnn for mention representation", default=False, action='store_true')
parser.add_argument("-dim_hidden", help="The number of hidden dimension.", default=100, type=int)
parser.add_argument("-rnn_dim", help="The number of RNN dimension.", default=100, type=int)
parser.add_argument("-add_headword_emb", help="Adding headword emb.", default=False, action='store_true')
parser.add_argument("-mention_lstm", help="Using LSTM for mention embedding.", default=False, action='store_true')
#parser.add_argument("-custom_loss", help="Using custom loss.", default=False, action='store_true')
parser.add_argument("-elmo", help="Using ELMo.", default=False, action='store_true')
parser.add_argument("-bert", help="Finetune BERT.", default=False, action='store_true')
parser.add_argument("-threshold", help="threshold", default=0.5, type=float)

# BERT
parser.add_argument("-bert_param_path", help="BERT: pretrained param path", default="")

# Save / log related
parser.add_argument("-save_period", help="How often to save", default=1000000, type=int)
parser.add_argument("-eval_period", help="How often to run dev", default=500, type=int)
parser.add_argument("-log_period", help="How often to save", default=100, type=int)
parser.add_argument("-load", help="Load existing model.", action='store_true')
parser.add_argument("-reload_model_name", help="")
