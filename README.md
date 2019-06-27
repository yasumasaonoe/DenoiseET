# Learning to Denoise Distantly-Labeled Data for Entity Typing
This is a PyTorch implementation of the fine-grained entity typing system presented in the NAACL 2019 paper [Learning to Denoise Distantly-Labeled Data for Entity Typing](https://www.aclweb.org/anthology/N19-1250).

## Dependencies
The code is developed with `python 3.6` and `pytorch 0.4.0`. We use [spaCy](https://spacy.io/) to preprocess data.

## Data
The ultra-fine entity typing dataset is available [here](https://homes.cs.washington.edu/~eunsol/open_entity.html).

#### Preprocessing
Our models require mention headwords. See `./data_tools/add_tree.py` how to add headwords to the original data. `./data/crowd` contains the preprocessed manually-annotated data.

## Training Models
#### Ultra-Fine Entity Typing
Entity Typing Model:
```
python3 main.py et_model -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type et_model -remove_el -remove_open
```
Relabeling Model:
```
python3 main.py filter -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type filter -remove_el -remove_open -mode train_labeler
```
Filtering Model:
```
python3 main.py labeler -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type labeler -remove_el -remove_open -mode train_labeler
```
BERT:
```
python3 main.py bert_uncased_small -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type bert_uncase_small -remove_el -remove_open
```
#### Ontonotes
Coming soon...

See sample scripts in `./scripts`.

## Denoising Data
Coming soon...

## Acknowledgements

Our code is largely borrowed from Eunsol Choi's implementation.

GitHub: https://github.com/uwnlp/open_type
Paper : https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf
