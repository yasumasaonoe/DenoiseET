"""
  This code is largely borrowed from Eunsol Choi's implementation.

  GitHub: https://github.com/uwnlp/open_type
  Paper : https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf

"""

import numpy as np
import json, sys, pickle
from eval_metric import mrr, macro, f1 

def stratify(all_labels, types):
  """
  Divide label into three categories.
  """
  coarse = types[:9]
  fine = types[9:130]
  return ([l for l in all_labels if l in coarse],
          [l for l in all_labels if ((l in fine) and (not l in coarse))],
          [l for l in all_labels if (not l in coarse) and (not l in fine)])

def get_mrr(pred_fname):
  dicts = pickle.load(open(pred_fname, "rb"))
  mrr_value = mrr(dicts['pred_dist'], dicts['gold_id_array'])
  return mrr_value

def compute_prf1(fname):
  with open(fname) as f:
    total = json.load(f)  
  true_and_predictions = []
  for k, v in total.items():
    true_and_predictions.append((v['gold'], v['pred']))
  count, pred_count, avg_pred_count, p, r, f1 = macro(true_and_predictions)
  perf_total = "{0}\t{1:.2f}\tP:{2:.1f}\tR:{3:.1f}\tF1:{4:.1f}".format(count, avg_pred_count, p * 100,
                                                                    r * 100, f1 * 100)
  print(perf_total)

def compute_granul_prf1(fname, type_fname):
  with open(fname) as f:
    total = json.load(f)  
  coarse_true_and_predictions = []
  fine_true_and_predictions = []
  finer_true_and_predictions = []
  with open(type_fname) as f:
    types = [x.strip() for x in f.readlines()]
  for k, v in total.items():
    coarse_gold, fine_gold, finer_gold = stratify(v['gold'], types)
    coarse_pred, fine_pred, finer_pred = stratify(v['pred'], types)
    coarse_true_and_predictions.append((coarse_gold, coarse_pred))
    fine_true_and_predictions.append((fine_gold, fine_pred))
    finer_true_and_predictions.append((finer_gold, finer_pred))

  for true_and_predictions in [coarse_true_and_predictions, fine_true_and_predictions, finer_true_and_predictions]:
    count, pred_count, avg_pred_count, p, r, f1 = macro(true_and_predictions)
    perf = "{0}\t{1:.2f}\tP:{2:.1f}\tR:{3:.1f}\tF1:{4:.1f}".format(count, avg_pred_count, p * 100,
                                                                    r * 100, f1 * 100)
    print(perf)

def load_augmented_input(fname):
  output_dict = {}
  with open(fname) as f:
    for line in f:
      elem = json.loads(line.strip())
      mention_id = elem.pop("annot_id")
      output_dict[mention_id] = elem
  return output_dict

def visualize(gold_pred_fname, original_fname, type_fname):
  with open(gold_pred_fname) as f:
    total = json.load(f) 
  original = load_augmented_input(original_fname)
  with open(type_fname) as f:
    types = [x.strip() for x in f.readlines()]
  for annot_id, v in total.items():
    elem = original[annot_id]
    mention = elem['mention_span']
    left = elem['left_context_token']
    right = elem['right_context_token']
    text_str = ' '.join(left)+" __"+mention+"__ "+' '.join(right)
    gold = v['gold']
    print('  |  '.join([text_str, ', '.join([("__"+v+"__" if v in gold else v )for v in v['pred']]), ','.join(gold)]))

#######
def compute_length_prf1(fname, data_fname):
  with open(fname) as f:
    total = json.load(f)  
  data = original = load_augmented_input(data_fname)
  counts = {}
  for annot_id, v in total.items():
      ex = data[annot_id]
      mention_len = len(ex['mention_span'].strip().split()) 
      if mention_len not in counts:
        counts[mention_len] = []
      counts[mention_len].append((v['gold'], v['pred']))

  for k, v in sorted(counts.items(), key=lambda x: x[0])[:20]:
    count, pred_count, avg_pred_count, p, r, f1 = macro(v)
    perf = "{0}\t{1:.2f}\tP:{2:.1f}\tR:{3:.1f}\tF1:{4:.1f}\tLEN:{5}".format(count, avg_pred_count, p * 100,
                                                                    r * 100, f1 * 100, k)
    print(perf)


def load_json(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  return [json.loads(line.strip()) for line in lines]


def compute_acc_by_type_freq(fname, type_bucket_count_file, types_file):
  with open(fname) as f:
    total = json.load(f)
  with open(type_bucket_count_file, 'rb') as f:
    type_bucket_count = pickle.load(f)
  with open(types_file, 'r') as f:
    types = [t.strip() for t in f.readlines()]
  #print('TOTAL:', sum([len(v['gold']) for k, v in total.items()]))
  type2bucket = {t[0]:k for k, v in type_bucket_count.items() for t in v}
  TP_FP_counts = {'unseen': 0.}
  TP_FN_counts = {'unseen': 0.}
  TP_counts = {'unseen': 0.}               
  for annot_id, v in total.items():
    gold = v['gold']
    pred = v['pred'] 
    for t in set(pred).intersection(set(gold)):
      if t in type2bucket:
        bucket = type2bucket[t]
        if bucket not in TP_counts:
          TP_counts[bucket] = 0.
        TP_counts[bucket] += 1.
      else:
        TP_counts['unseen'] += 1.
    for t in set(pred):
      if t in type2bucket:
        bucket = type2bucket[t]
        if bucket not in TP_FP_counts:
          TP_FP_counts[bucket] = 0.
        TP_FP_counts[bucket] += 1.
      else:
        TP_FP_counts['unseen'] += 1.
    for t in set(gold):
      if t in type2bucket:
        bucket = type2bucket[t]
        if bucket not in TP_FN_counts:
          TP_FN_counts[bucket] = 0.
        TP_FN_counts[bucket] += 1.
      else:
        TP_FN_counts['unseen'] += 1.

  ordered_keys = sorted([k for k,v in TP_counts.items() if k != 'unseen'], key=lambda x: int(x.split('-')[0]), reverse=True) # + ['unseen']
  for k in ordered_keys:
    precision = TP_counts[k] / TP_FP_counts[k]
    recall = TP_counts[k] / TP_FN_counts[k]
    f1_score = f1(precision, recall)
    perf = "{0}\tCORRECT:{1}\tP:{2:.2f}\tR:{3:.2f}\tF1:{4:.2}".format(k, int(TP_counts[k]), precision*100., recall*100., f1_score*100.)
    print(perf)

def compute_prf1_single_type(fname, type_, data=None):
  print('---------- ' + type_ + ' ----------')
  with open(fname) as f:
    total = json.load(f)  
  gold_binary = []
  pred_binary = [] 
  for k, v in total.items():
    if type_ in v['gold']:
      gold_binary.append(1.)
    else:
      gold_binary.append(0.)
    if type_ in v['pred']:
      pred_binary.append(1.)
      print_example(data[k])
    else:
      pred_binary.append(0.)
  count = len(gold_binary)
  TP_FN_counts = sum([1.  for gold in gold_binary if int(gold) == 1])
  TP_FP_counts = sum([1.  for pred in pred_binary if int(pred) == 1])
  TP_counts = sum([1.  for pred, gold in zip(pred_binary, gold_binary) if int(pred) == 1 and int(gold) == 1])
  p = TP_counts / TP_FP_counts if TP_FP_counts > 0 else 0.
  r = TP_counts / TP_FN_counts if TP_FN_counts > 0 else 0.
  f1_ = f1(p, r)
  output_str = "Type: {0}\t#: {1} TP:{2} TP+FP:{3} TP+FN:{4} P:{5:.3f} R:{6:.3f} F1:{7:.3f}".format(type_, count, int(TP_counts), int(TP_FP_counts), int(TP_FN_counts), p, r, f1_)
  accuracy = sum([pred == gold for pred, gold in zip(pred_binary, gold_binary)]) / float(len(gold_binary))
  output_str += '\t Dev accuracy: {0:.1f}%'.format(accuracy * 100)
  print(output_str)

def print_example(ex):
  gold = ex['y_str']
  sent = ' '.join(ex['left_context_token']) + ' [' + ex['mention_span'] + '] ' + ' '.join(ex['right_context_token']) 
  print('ID  : ' + ex['annot_id'])
  print('SENT: ' + sent)
  print('GOLD: ' + ',  '.join(gold))
  print()
#######

if __name__ == '__main__':
  gold_pred_str_fname = sys.argv[1]+'.json'
  type_fname = './resources/types.txt'
  data_fname = '/backup2/yasu/data/ultra_fine_entity_typing/data/crowd/dev_tree_keep30.json'
  type_bucket_count_file_crowd = './resources/type_bucket_count_crowd_only.pkl'
  type_bucket_count_file_full = './resources/type_bucket_count.pkl' #'./resources/type_bucket_count_full_train.pkl'
  types_file = './resources/types.txt'
  dev_data = load_json(data_fname)
  dev_data = {d['annot_id']:d for d in dev_data}

  # compute precision, recall, f1
  compute_prf1(gold_pred_str_fname)
  print()
  print('printing performance for coarse, fine, finer labels in order')
  compute_granul_prf1(gold_pred_str_fname, type_fname)
  print()
  print('printing performance by mention span length')
  compute_length_prf1(gold_pred_str_fname, data_fname)
  print()
  print('printing performance by type frequency - crowd only')
  compute_acc_by_type_freq(gold_pred_str_fname, type_bucket_count_file_crowd, types_file)
  print()
  print('printing performance by type frequency - full data')
  compute_acc_by_type_freq(gold_pred_str_fname, type_bucket_count_file_full, types_file)
  print()

  #compute_prf1_single_type(gold_pred_str_fname, 'person')
  #compute_prf1_single_type(gold_pred_str_fname, 'entity')
  #compute_prf1_single_type(gold_pred_str_fname, 'politician', data=dev_data)
  #compute_prf1_single_type(gold_pred_str_fname, 'water')
  #compute_prf1_single_type(gold_pred_str_fname, 'adult', data=dev_data)
  #compute_prf1_single_type(gold_pred_str_fname, 'accomplishment')
  #compute_prf1_single_type(gold_pred_str_fname, 'time')
  #compute_prf1_single_type(gold_pred_str_fname, 'music')
  #compute_prf1_single_type(gold_pred_str_fname, 'sport')
  #compute_prf1_single_type(gold_pred_str_fname, 'military')
  #compute_prf1_single_type(gold_pred_str_fname, 'day')
  #compute_prf1_single_type(gold_pred_str_fname, 'president')


