# Learning to Denoise Distantly-Labeled Data for Entity Typing
This is a PyTorch implementation of the fine-grained entity typing system presented in the NAACL 2019 paper [Learning to Denoise Distantly-Labeled Data for Entity Typing](https://www.aclweb.org/anthology/N19-1250).

## Dependencies
The code is developed with `python 3.6` and `pytorch 0.4.0`. We use [spaCy](https://spacy.io/) to preprocess data.

## Data
The ultra-fine entity typing dataset is available [here](https://homes.cs.washington.edu/~eunsol/open_entity.html). Download the `data` folder from [here](https://drive.google.com/file/d/1FN06VY77Llo_mNSuCO-Qpcqk5GtKArds/view?usp=sharing). Modify `./resources/constant.py` accordingly to make shure that all paths are pointing to the right directories. 

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
python3 main.py labeler -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type labeler -remove_el -remove_open -mode train_labeler
```
Filtering Model:
```
python3 main.py filter -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type filter -remove_el -remove_open -mode train_labeler
```
BERT:
```
python3 main.py bert_uncased_small -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type bert_uncase_small -remove_el -remove_open
```
#### Ontonotes
Coming soon...

## Evaluating Models
Once you trained an entity typing model, you can evaluate it on the dev/test set with the command below. `[MODEL NAME]` is the model file (without suffix).

Entity Typing Model:
```
python3 main.py et_model_eval -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type et_model -mode test -reload_model_name [MODEL NAME] -eval_data crowd/dev_tree.json -load
```
#### Ontonotes
Coming soon...


## Denoising Data
Once filter and relabeling models are trained, you can run them on the dataset of your choice. `[MODEL NAME]` is the model file (without suffix). `[DATA FILE NAME]` is the data file that you want to denoise.

Filtering Model:
```
python3 -u main.py filter_eval -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type filter -mode test_labeler -reload_model_name [MODEL NAME] -eval_data [DATA FILE NAME] -load
```
After running this command, `filter_eval.json` will be saved in the current directory. The model predictions are stored with the `pred` key.


Relabeling Model:
```
python3 -u main.py filter_eval -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type filter -mode test_labeler -reload_model_name [MODEL NAME] -eval_data [DATA FILE NAME] -load
```
After running this command, `labeler_eval.json` will be saved in the current directory. The model predictions are stored with the `cls_pred` key (`1` if the example is classified as a bad example, `0` otherwise).

## Questions
Contact us at `yasumasa@cs.utexas.edu` if you have any questions!


## Acknowledgements
Our code is largely borrowed from Eunsol Choi's implementation.

GitHub: https://github.com/uwnlp/open_type
Paper : https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf
