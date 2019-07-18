#!/usr/bin/env python3

"""
  This code is largely borrowed from Eunsol Choi's implementation.

  GitHub: https://github.com/uwnlp/open_type
  Paper : https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf

"""

import datetime
import gc
import logging
import pickle
import os
import sys
import time, json
import torch

import data_utils
import eval_metric
import models
import denoising_models
from data_utils import to_torch
from eval_metric import mrr, f1
from model_utils import get_gold_pred_str, get_eval_string, get_output_index
from tensorboardX import SummaryWriter
from torch import optim
from random import shuffle

sys.path.insert(0, './resources')
import config_parser, constant, eval_metric

sys.path.insert(0, './bert')
import tokenization
from optimization import BERTAdam

from collections import defaultdict

class TensorboardWriter:
  """
  Wraps a pair of ``SummaryWriter`` instances but is a no-op if they're ``None``.
  Allows Tensorboard logging without always checking for Nones first.
  """

  def __init__(self, train_log: SummaryWriter = None, validation_log: SummaryWriter = None) -> None:
    self._train_log = train_log
    self._validation_log = validation_log

  def add_train_scalar(self, name: str, value: float, global_step: int) -> None:
    if self._train_log is not None:
      self._train_log.add_scalar(name, value, global_step)

  def add_validation_scalar(self, name: str, value: float, global_step: int) -> None:
    if self._validation_log is not None:
      self._validation_log.add_scalar(name, value, global_step)


def get_data_gen(dataname, mode, args, vocab_set, goal, elmo=None, bert=None):
  dataset = data_utils.TypeDataset(constant.FILE_ROOT + dataname, goal=goal, vocab=vocab_set,
                                   elmo=elmo, bert=bert, args=args)
  if mode == 'train':
    data_gen = dataset.get_batch(args.batch_size, args.num_epoch, forever=False, eval_data=False,
                                 simple_mention=not args.enhanced_mention)
  elif mode == 'dev':
    data_gen = dataset.get_batch(args.eval_batch_size, 1, forever=True, eval_data=True,
                                 simple_mention=not args.enhanced_mention)
  else:
    data_gen = dataset.get_batch(args.eval_batch_size, 1, forever=False, eval_data=True,
                                 simple_mention=not args.enhanced_mention)
  return data_gen


def get_joint_datasets(args):

  if args.elmo:
    vocab = (constant.CHAR_DICT, None) # dummy empty dict
    elmo = data_utils.init_elmo()
    bert = None
  elif args.bert:
    vocab = (constant.CHAR_DICT, None) # dummy empty dict 
    elmo = None
    bert = None
  else: # glove
    vocab = data_utils.get_vocab()
    elmo = None
    bert = None
  train_gen_list = []
  valid_gen_list = []
  if args.mode in ['train', 'train_labeler']:
    if not args.remove_open and not args.only_crowd:
      train_gen_list.append(
        ("open", get_data_gen('train_full/open_train_tree*.json', 'train', args, vocab, "open", elmo=elmo, bert=bert)))
        #("open", get_data_gen('distant_supervision/headword_train_tree.json', 'train', args, vocab, "open", elmo=elmo, bert=bert)))
      valid_gen_list.append(("open", get_data_gen('distant_supervision/headword_dev_tree.json', 'dev', args, vocab, "open", elmo=elmo, bert=bert)))
    if not args.remove_el and not args.only_crowd:
      valid_gen_list.append(
        ("wiki",
         get_data_gen('distant_supervision/el_dev_tree.json', 'dev', args, vocab, "wiki" if args.multitask else "open", elmo=elmo, bert=bert)))
      train_gen_list.append(
        ("wiki",
         #get_data_gen('distant_supervision/el_train_tree.json', 'train', args, vocab, "wiki" if args.multitask else "open", elmo=elmo, bert=bert)))
         get_data_gen('train_full/el_train_full_tree.json', 'train', args, vocab, "wiki" if args.multitask else "open", elmo=elmo, bert=bert)))
    if args.add_crowd or args.only_crowd:
      train_gen_list.append(
        ("open", get_data_gen('crowd/train_m_tree.json', 'train', args, vocab, "open", elmo=elmo, bert=bert)))
    if args.add_expanded_head:
      train_gen_list.append(
        ("open", get_data_gen('train_full/open_train_1m_cls_relabeled.json', 'train', args, vocab, "open", elmo=elmo, bert=bert)))
    if args.add_expanded_el:
      train_gen_list.append(
        ("wiki", get_data_gen('train_full/el_train_1m_cls_relabeled.json', 'train', args, vocab,  "wiki" if args.multitask else "open", elmo=elmo, bert=bert)))
  #crowd_dev_gen = get_data_gen('crowd/dev.json', 'dev', args, vocab, "open")
  crowd_dev_gen = None # get_data_gen('crowd/dev_tree.json', 'dev', args, vocab, "open", elmo=elmo, bert=bert)
  return train_gen_list, valid_gen_list, crowd_dev_gen, elmo, bert, vocab


def get_datasets(data_lists, args):
  data_gen_list = []
  if args.elmo:
    vocab = (constant.CHAR_DICT, None) # dummy empty dict
    elmo = data_utils.init_elmo()
    bert = None
  elif args.bert:
    vocab = (constant.CHAR_DICT, None) # dummy empty dict 
    elmo = None
    bert = None
  else:
    vocab = data_utils.get_vocab()
    elmo = None
    bert = None
  for dataname, mode, goal in data_lists:
    data_gen_list.append(get_data_gen(dataname, mode, args, vocab, goal, elmo=elmo, bert=bert))
  return data_gen_list, elmo


def _train(args):
  if args.data_setup == 'joint':
    train_gen_list, val_gen_list, crowd_dev_gen, elmo, bert, vocab = get_joint_datasets(args)
  else:
    train_fname = args.train_data
    dev_fname = args.dev_data
    print(train_fname, dev_fname)
    data_gens, elmo = get_datasets([(train_fname, 'train', args.goal),
                              (dev_fname, 'dev', args.goal)], args)
    train_gen_list = [(args.goal, data_gens[0])]
    val_gen_list = [(args.goal, data_gens[1])]
  train_log = SummaryWriter(os.path.join(constant.EXP_ROOT, args.model_id, "log", "train"))
  validation_log = SummaryWriter(os.path.join(constant.EXP_ROOT, args.model_id, "log", "validation"))
  tensorboard = TensorboardWriter(train_log, validation_log)

  if args.model_type == 'et_model':
    print('==> Entity Typing Model')
    model = models.ETModel(args, constant.ANSWER_NUM_DICT[args.goal])
  elif args.model_type == 'bert_uncase_small':
    print('==> Bert Uncased Small')
    model = models.Bert(args, constant.ANSWER_NUM_DICT[args.goal])
  else:
    print('Invalid model type: -model_type ' + args.model_type)
    raise NotImplementedError

  model.cuda()
  total_loss = 0
  batch_num = 0
  best_macro_f1 = 0.
  start_time = time.time()
  init_time = time.time()
  if args.bert:
    if args.bert_param_path:
      print('==> Loading BERT from ' + args.bert_param_path)
      model.bert.load_state_dict(torch.load(args.bert_param_path, map_location='cpu'))
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
        ]
    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.bert_learning_rate,
                         warmup=args.bert_warmup_proportion,
                         t_total=-1) # TODO: 
  else:
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
  #optimizer = optim.SGD(model.parameters(), lr=1., momentum=0.)

  if args.load:
    load_model(args.reload_model_name, constant.EXP_ROOT, args.model_id, model, optimizer)

  for idx, m in enumerate(model.modules()):
    logging.info(str(idx) + '->' + str(m))

  while True:
    batch_num += 1  # single batch composed of all train signal passed by.
    for (type_name, data_gen) in train_gen_list: 
      try:
        batch = next(data_gen)
        batch, _ = to_torch(batch)
      except StopIteration:
        logging.info(type_name + " finished at " + str(batch_num))
        print('Done!')
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   '{0:s}/{1:s}.pt'.format(constant.EXP_ROOT, args.model_id))
        return
      optimizer.zero_grad()
      loss, output_logits, _ = model(batch, type_name)
      loss.backward()
      total_loss += loss.item()
      optimizer.step()

      if batch_num % args.log_period == 0 and batch_num > 0:
        gc.collect()
        cur_loss = float(1.0 * loss.clone().item())
        elapsed = time.time() - start_time
        train_loss_str = ('|loss {0:3f} | at {1:d}step | @ {2:.2f} ms/batch'.format(cur_loss, batch_num,
                                                                                    elapsed * 1000 / args.log_period))
        start_time = time.time()
        print(train_loss_str)
        logging.info(train_loss_str)
        tensorboard.add_train_scalar('train_loss_' + type_name, cur_loss, batch_num)

      if batch_num % args.eval_period == 0 and batch_num > 0:
        output_index = get_output_index(output_logits, threshold=args.threshold)
        gold_pred_train = get_gold_pred_str(output_index, batch['y'].data.cpu().clone(), args.goal)
        print(gold_pred_train[:10]) 
        accuracy = sum([set(y) == set(yp) for y, yp in gold_pred_train]) * 1.0 / len(gold_pred_train)

        train_acc_str = '{1:s} Train accuracy: {0:.1f}%'.format(accuracy * 100, type_name)
        print(train_acc_str)
        logging.info(train_acc_str)
        tensorboard.add_train_scalar('train_acc_' + type_name, accuracy, batch_num)
        if args.goal != 'onto':
          for (val_type_name, val_data_gen) in val_gen_list:
            if val_type_name == type_name:
              eval_batch, _ = to_torch(next(val_data_gen))
              evaluate_batch(batch_num, eval_batch, model, tensorboard, val_type_name, args, args.goal)

    if batch_num % args.eval_period == 0 and batch_num > 0 and args.data_setup == 'joint':
      # Evaluate Loss on the Turk Dev dataset.
      print('---- eval at step {0:d} ---'.format(batch_num))
      ###############
      #feed_dict = next(crowd_dev_gen)
      #eval_batch, _ = to_torch(feed_dict)
      #crowd_eval_loss = evaluate_batch(batch_num, eval_batch, model, tensorboard, "open", args.goal, single_type=args.single_type)
      ###############
      crowd_eval_loss, macro_f1 = evaluate_data(batch_num, 'crowd/dev_tree.json', model,
                                                tensorboard, "open", args, elmo, bert)

      if best_macro_f1 < macro_f1:
        best_macro_f1 = macro_f1
        save_fname = '{0:s}/{1:s}_best.pt'.format(constant.EXP_ROOT, args.model_id)
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_fname)
        print(
          'Total {0:.2f} minutes have passed, saving at {1:s} '.format((time.time() - init_time) / 60, save_fname))

    if batch_num % args.eval_period == 0 and batch_num > 0 and args.goal == 'onto':
      # Evaluate Loss on the Turk Dev dataset.
      print('---- OntoNotes: eval at step {0:d} ---'.format(batch_num))
      crowd_eval_loss, macro_f1 = evaluate_data(batch_num, args.dev_data, model, tensorboard,
                                                args.goal, args, elmo, bert)

    if batch_num % args.save_period == 0 and batch_num > 30000:
      save_fname = '{0:s}/{1:s}_{2:d}.pt'.format(constant.EXP_ROOT, args.model_id, batch_num)
      torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_fname)
      print(
        'Total {0:.2f} minutes have passed, saving at {1:s} '.format((time.time() - init_time) / 60, save_fname))
  # Training finished! 
  torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
             '{0:s}/{1:s}.pt'.format(constant.EXP_ROOT, args.model_id))


def _train_labeler(args):
  if args.data_setup == 'joint':
    train_gen_list, val_gen_list, crowd_dev_gen, elmo, bert, vocab = get_joint_datasets(args)
  else:
    train_fname = args.train_data
    dev_fname = args.dev_data
    print(train_fname, dev_fname)
    data_gens, elmo = get_datasets([(train_fname, 'train', args.goal),
                              (dev_fname, 'dev', args.goal)], args)
    train_gen_list = [(args.goal, data_gens[0])]
    val_gen_list = [(args.goal, data_gens[1])]
  train_log = SummaryWriter(os.path.join(constant.EXP_ROOT, args.model_id, "log", "train"))
  validation_log = SummaryWriter(os.path.join(constant.EXP_ROOT, args.model_id, "log", "validation"))
  tensorboard = TensorboardWriter(train_log, validation_log)

  if args.model_type == 'labeler':
    print('==> Labeler')
    model = denoising_models.Labeler(args, constant.ANSWER_NUM_DICT[args.goal])
  elif args.model_type == 'filter':
    print('==> Filter')
    model = denoising_models.Filter(args, constant.ANSWER_NUM_DICT[args.goal])
  else:
    print('Invalid model type: -model_type ' + args.model_type)
    raise NotImplementedError

  model.cuda()
  total_loss = 0
  batch_num = 0
  best_macro_f1 = 0.
  start_time = time.time()
  init_time = time.time()

  if args.bert:
    if args.bert_param_path:
      print('==> Loading BERT from ' + args.bert_param_path)
      model.bert.load_state_dict(torch.load(args.bert_param_path, map_location='cpu'))
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
        ]
    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.bert_learning_rate,
                         warmup=args.bert_warmup_proportion,
                         t_total=-1) # TODO: 
  else:
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
  #optimizer = optim.SGD(model.parameters(), lr=1., momentum=0.)

  if args.load:
    load_model(args.reload_model_name, constant.EXP_ROOT, args.model_id, model, optimizer)

  for idx, m in enumerate(model.modules()):
    logging.info(str(idx) + '->' + str(m))

  while True:
    batch_num += 1  # single batch composed of all train signal passed by.
    for (type_name, data_gen) in train_gen_list:
      try:
        batch = next(data_gen)
        batch, _ = to_torch(batch)
      except StopIteration:
        logging.info(type_name + " finished at " + str(batch_num))
        print('Done!')
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   '{0:s}/{1:s}.pt'.format(constant.EXP_ROOT, args.model_id))
        return
      optimizer.zero_grad()
      loss, output_logits, cls_logits = model(batch, type_name)
      loss.backward()
      total_loss += loss.item()
      optimizer.step()

      if batch_num % args.log_period == 0 and batch_num > 0:
        gc.collect()
        cur_loss = float(1.0 * loss.clone().item())
        elapsed = time.time() - start_time
        train_loss_str = ('|loss {0:3f} | at {1:d}step | @ {2:.2f} ms/batch'.format(cur_loss, batch_num,
                                                                                    elapsed * 1000 / args.log_period))
        start_time = time.time()
        print(train_loss_str)
        logging.info(train_loss_str)
        tensorboard.add_train_scalar('train_loss_' + type_name, cur_loss, batch_num)

      if batch_num % args.eval_period == 0 and batch_num > 0:
        output_index = get_output_index(output_logits, threshold=args.threshold)
        gold_pred_train = get_gold_pred_str(output_index, batch['y'].data.cpu().clone(), args.goal)
        print(gold_pred_train[:10])
        accuracy = sum([set(y) == set(yp) for y, yp in gold_pred_train]) * 1.0 / len(gold_pred_train)

        train_acc_str = '{1:s} Train accuracy: {0:.1f}%'.format(accuracy * 100, type_name)
        if cls_logits is not None:
          cls_accuracy =  sum([(1. if pred > 0. else 0.) == gold for pred, gold in zip(cls_logits, batch['y_cls'].data.cpu().numpy())])  / float(cls_logits.size()[0])
          cls_tp = sum([(1. if pred > 0. else 0.) == 1. and gold == 1. for pred, gold in zip(cls_logits, batch['y_cls'].data.cpu().numpy())])
          cls_precision = cls_tp  / float(sum([1. if pred > 0. else 0. for pred in cls_logits])) 
          cls_recall = cls_tp  / float(sum(batch['y_cls'].data.cpu().numpy()))
          cls_f1 = f1(cls_precision, cls_recall)
          train_cls_acc_str = '{1:s} Train cls accuracy: {0:.2f}%  P: {2:.3f}  R: {3:.3f}  F1: {4:.3f}'.format(cls_accuracy * 100, type_name, cls_precision, cls_recall, cls_f1)
        print(train_acc_str)
        if cls_logits is not None:
          print(train_cls_acc_str)
        logging.info(train_acc_str)
        tensorboard.add_train_scalar('train_acc_' + type_name, accuracy, batch_num)
        if args.goal != 'onto':
          for (val_type_name, val_data_gen) in val_gen_list:
            if val_type_name == type_name:
              eval_batch, _ = to_torch(next(val_data_gen))
              evaluate_batch(batch_num, eval_batch, model, tensorboard, val_type_name, args, args.goal)

    if batch_num % args.eval_period == 0 and batch_num > 0 and args.data_setup == 'joint':
      # Evaluate Loss on the Turk Dev dataset.
      print('---- eval at step {0:d} ---'.format(batch_num))
      crowd_eval_loss, macro_f1 = evaluate_data(batch_num, 'crowd/dev_tree.json', model,
                                                tensorboard, "open", args, elmo, bert, vocab=vocab)

      if best_macro_f1 < macro_f1:
        best_macro_f1 = macro_f1
        save_fname = '{0:s}/{1:s}_best.pt'.format(constant.EXP_ROOT, args.model_id)
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_fname)
        print(
          'Total {0:.2f} minutes have passed, saving at {1:s} '.format((time.time() - init_time) / 60, save_fname))

    if batch_num % args.eval_period == 0 and batch_num > 0 and args.goal == 'onto':
      # Evaluate Loss on the Turk Dev dataset.
      print('---- OntoNotes: eval at step {0:d} ---'.format(batch_num))
      crowd_eval_loss, macro_f1 = evaluate_data(batch_num, args.dev_data, model, tensorboard,
                                                args.goal, args, elmo)

    if batch_num % args.save_period == 0 and batch_num > 0:
      save_fname = '{0:s}/{1:s}_{2:d}.pt'.format(constant.EXP_ROOT, args.model_id, batch_num)
      torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_fname)
      print(
        'Total {0:.2f} minutes have passed, saving at {1:s} '.format((time.time() - init_time) / 60, save_fname))
  # Training finished! 
  torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
             '{0:s}/{1:s}.pt'.format(constant.EXP_ROOT, args.model_id))


def evaluate_batch(batch_num, eval_batch, model, tensorboard, val_type_name, args, goal):
  model.eval()
  loss, output_logits, _ = model(eval_batch, val_type_name)
  output_index = get_output_index(output_logits, threshold=args.threshold)
  gold_pred = get_gold_pred_str(output_index, eval_batch['y'].data.cpu().clone(), goal)
  eval_accu = sum([set(y) == set(yp) for y, yp in gold_pred]) * 1.0 / len(gold_pred)
  eval_str = get_eval_string(gold_pred)
  eval_loss = loss.clone().item()
  eval_loss_str = 'Eval loss: {0:.7f} at step {1:d}'.format(eval_loss, batch_num)
  tensorboard.add_validation_scalar('eval_acc_' + val_type_name, eval_accu, batch_num)
  tensorboard.add_validation_scalar('eval_loss_' + val_type_name, eval_loss, batch_num)
  print(val_type_name + ":" +eval_loss_str)
  print(gold_pred[:3])
  print(val_type_name+":"+ eval_str)
  logging.info(val_type_name + ":" + eval_loss_str)
  logging.info(val_type_name +":" +  eval_str)
  model.train()
  return eval_loss


def evaluate_data(batch_num, dev_fname, model, tensorboard, val_type_name, args, elmo, bert, actual_f1=True, vocab=None):
  model.eval()
  if vocab is None:
    vocab = (constant.CHAR_DICT, None)
  dev_gen = get_data_gen(dev_fname, 'test', args, vocab, args.goal, elmo=elmo, bert=bert)
  gold_pred = []
  binary_out = []
  eval_loss = 0.
  total_ex_count = 0
  if args.mode in ['train_labeler', 'test_labeler']:
    cls_correct = 0.
    cls_total = 0.
    cls_tp = 0.
    cls_t_gold = 0.
    cls_t_pred = 0.
  for n, batch in enumerate(dev_gen): 
    total_ex_count += len(batch['y'])
    eval_batch, annot_ids = to_torch(batch)
    if args.mode in ['train_labeler', 'test_labeler']:
      loss, output_logits, cls_logits = model(eval_batch, val_type_name)
      if cls_logits is not None:
        cls_correct +=  sum([(1. if pred > 0. else 0.) == gold for pred, gold in zip(cls_logits, batch['y_cls'])])
        cls_total += float(cls_logits.size()[0])
        cls_tp += sum([(1. if pred > 0. else 0.) == 1. and gold == 1. for pred, gold in zip(cls_logits, batch['y_cls'])])
        cls_t_gold += float(sum(batch['y_cls']))
        cls_t_pred += float(sum([1. if pred > 0. else 0. for pred in cls_logits]))
    else:
      loss, output_logits, _ = model(eval_batch, val_type_name)
    output_index = get_output_index(output_logits, threshold=args.threshold)
    gold_pred += get_gold_pred_str(output_index, eval_batch['y'].data.cpu().clone(), args.goal)
    eval_loss += loss.clone().item()
  eval_accu = sum([set(y) == set(yp) for y, yp in gold_pred]) * 1.0 / len(gold_pred)
  eval_str = get_eval_string(gold_pred)
  _, _, _, _, _, macro_f1 = eval_metric.macro(gold_pred)
  eval_loss_str = 'Eval loss: {0:.7f} at step {1:d}'.format(eval_loss, batch_num)
  tensorboard.add_validation_scalar('eval_acc_' + val_type_name, eval_accu, batch_num)
  tensorboard.add_validation_scalar('eval_loss_' + val_type_name, eval_loss, batch_num)
  print('EVAL: seen ' + repr(total_ex_count) + ' examples.')
  print(val_type_name + ":" +eval_loss_str)
  print(gold_pred[:3])
  if args.mode in ['train_labeler', 'test_labeler'] and cls_logits is not None:
    cls_accuracy = cls_correct / cls_total * 100.
    cls_precision = cls_tp / cls_t_pred
    cls_recall = cls_tp / cls_t_gold
    cls_f1 = f1(cls_precision, cls_recall)
    cls_str = '  CLS accuracy: {0:.2f}%  P: {1:.3f}  R: {2:.3f}  F1: {3:.3f}'.format(cls_accuracy, cls_precision, cls_recall, cls_f1) 
    print(val_type_name+":"+ eval_str + cls_str)
  else:
    print(val_type_name+":"+ eval_str)
  logging.info(val_type_name + ":" + eval_loss_str)
  logging.info(val_type_name +":" +  eval_str)
  model.train()
  dev_gen = None
  return eval_loss, macro_f1 


def evaluate_data_cv(k_fold_count, batch_num, model, tensorboard, val_type_name, args, elmo, actual_f1=False):
  model.eval()
  data_gen = get_data_gen('crowd/cv_3fold/dev_tree_{0}.json'.format(repr(k_fold_count)), 'test', args, (constant.CHAR_DICT, None), args.goal, elmo=elmo)
  gold_pred = []
  annot_ids = []
  binary_out = []
  eval_loss = 0.
  total_ex_count = 0
  print('==> evaluate_data_cv')
  for n, batch in enumerate(data_gen):
    total_ex_count += len(batch['y'])
    eval_batch, annot_id = to_torch(batch)
    loss, output_logits, _ = model(eval_batch, val_type_name)
    if actual_f1:
      output_index = get_output_index(output_logits, threshold=args.threshold)
    else:
      output_index = get_output_index_rank(output_logits, topk=args.topk)

    y = eval_batch['y'].data.cpu().clone().numpy() 
    gold_pred = get_gold_pred_str(output_index, y, args.goal)
    annot_ids.extend(annot_id) 
    eval_loss += loss.clone().item()
  eval_accu = sum([set(y) == set(yp) for y, yp in gold_pred]) * 1.0 / len(gold_pred)
  eval_str = get_eval_string(gold_pred)
  eval_loss_str = 'Eval loss: {0:.7f} at step {1:d}'.format(eval_loss, batch_num)
  tensorboard.add_validation_scalar('eval_acc_' + val_type_name, eval_accu, batch_num)
  tensorboard.add_validation_scalar('eval_loss_' + val_type_name, eval_loss, batch_num)
  print('EVAL: seen ' + repr(total_ex_count) + ' examples.')
  print(val_type_name + ":" + eval_loss_str)
  #print(gold_pred[:3])
  print(val_type_name+":"+ eval_str)
  logging.info(val_type_name + ":" + eval_loss_str)
  logging.info(val_type_name + ":" + eval_str)
  model.train()
  data_gen = None
  output_dict = {}
  for a_id, (gold, pred) in zip(annot_ids, gold_pred):
    output_dict[a_id] = {"gold": gold, "pred": pred}
  return eval_loss, output_dict


def load_model(reload_model_name, save_dir, model_id, model, optimizer=None):
  if reload_model_name:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, reload_model_name)
  else:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, model_id)
  checkpoint = torch.load(model_file_name)
  model.load_state_dict(checkpoint['state_dict'])
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer'])
  else:
    total_params = 0
    # Log params
    for k in checkpoint['state_dict']:
      elem = checkpoint['state_dict'][k]
      param_s = 1
      for size_dim in elem.size():
        param_s = size_dim * param_s
      print(k, elem.size())
      total_params += param_s
    param_str = ('Number of total parameters..{0:d}'.format(total_params))
    logging.info(param_str)
    print(param_str)
  logging.info("Loading old file from {0:s}".format(model_file_name))
  print('Loading model from ... {0:s}'.format(model_file_name))


def load_model_partially(reload_model_name, save_dir, model_id, model, freeze=False, optimizer=None):
  if reload_model_name:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, reload_model_name)
  else:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, model_id)
  checkpoint = torch.load(model_file_name)
  pretrained_state_dict = checkpoint['state_dict']
  model_state_dict = model.state_dict()
  pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
  model_state_dict.update(pretrained_state_dict)
  model.load_state_dict(model_state_dict)
  if freeze:
    for pname, param in model.named_parameters():
      if pname in pretrained_state_dict:
        param.requires_grad = False
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer'])
  else:
    total_params = 0
    # Log params
    for k in checkpoint['state_dict']:
      elem = checkpoint['state_dict'][k]
      param_s = 1
      for size_dim in elem.size():
        param_s = size_dim * param_s
      print(k, elem.size())
      total_params += param_s
    param_str = ('Number of total parameters..{0:d}'.format(total_params))
    logging.info(param_str)
    print(param_str)
  logging.info("Loading old file from {0:s}".format(model_file_name))
  print('Loading model from ... {0:s}'.format(model_file_name))


def _test(args):
  assert args.load
  test_fname = args.eval_data
  data_gens, _ = get_datasets([(test_fname, 'test', args.goal)], args)
  if args.model_type == 'et_model':
    print('==> Entity Typing Model')
    model = models.ETModel(args, constant.ANSWER_NUM_DICT[args.goal])
  elif args.model_type == 'bert_uncase_small':
    print('==> Bert Uncased Small')
    model = models.Bert(args, constant.ANSWER_NUM_DICT[args.goal])
  else:
    print('Invalid model type: -model_type ' + args.model_type)
    raise NotImplementedError
  model.cuda()
  model.eval()
  load_model(args.reload_model_name, constant.EXP_ROOT, args.model_id, model)

  for name, dataset in [(test_fname, data_gens[0])]:
    print('Processing... ' + name)
    total_gold_pred = []
    total_annot_ids = []
    total_probs = []
    total_ys = []
    batch_attn = []
    for batch_num, batch in enumerate(dataset):
      print(batch_num)
      if not isinstance(batch, dict):
        print('==> batch: ', batch)
      eval_batch, annot_ids = to_torch(batch)
      loss, output_logits, attn_score = model(eval_batch, args.goal)
      #batch_attn.append((batch, attn_score.data))
      output_index = get_output_index(output_logits, threshold=args.threshold)
      #output_prob = model.sigmoid_fn(output_logits).data.cpu().clone().numpy()
      y = eval_batch['y'].data.cpu().clone().numpy()
      gold_pred = get_gold_pred_str(output_index, y, args.goal)
      #total_probs.extend(output_prob)
      #total_ys.extend(y)
      total_gold_pred.extend(gold_pred)
      total_annot_ids.extend(annot_ids)
    #mrr_val = mrr(total_probs, total_ys)
    #print('mrr_value: ', mrr_val)
    #pickle.dump({'gold_id_array': total_ys, 'pred_dist': total_probs},
    #            open('./{0:s}.p'.format(args.reload_model_name), "wb"))
    with open('./{0:s}.json'.format(args.reload_model_name), 'w') as f_out:
      output_dict = {}
      counter = 0
      for a_id, (gold, pred) in zip(total_annot_ids, total_gold_pred):
        #attn = batch_attn[0][1].squeeze(2)[counter]
        #attn = attn.cpu().numpy().tolist()
        #print(attn, int(batch_attn[0][0]['mention_span_length'][counter]), sum(attn))
        #print(mntn_emb[counter])
        #print()
        #print(int(batch_attn[0][0]['mention_span_length'][counter]), batch_attn[0][0]['mention_embed'][counter].shape)
        #attn = attn[:int(batch_attn[0][0]['mention_span_length'][counter])]
        output_dict[a_id] = {"gold": gold, "pred": pred} #, "attn": attn, "mntn_len": int(batch_attn[0][0]['mention_span_length'][counter])}
        counter += 1
      json.dump(output_dict, f_out)
    eval_str = get_eval_string(total_gold_pred)
    print(eval_str)
    logging.info('processing: ' + name)
    logging.info(eval_str)


def _test_labeler(args):
  assert args.load
  test_fname = args.eval_data
  data_gens, _ = get_datasets([(test_fname, 'test', args.goal)], args)
  if args.model_type == 'labeler':
    print('==> Labeler')
    model = denoising_models.Labeler(args, constant.ANSWER_NUM_DICT[args.goal])
  elif args.model_type == 'filter':
    print('==> Filter')
    model = denoising_models.Filter(args, constant.ANSWER_NUM_DICT[args.goal])
  else:
    print('Invalid model type: -model_type ' + args.model_type)
    raise NotImplementedError

  model.cuda()
  model.eval()
  load_model(args.reload_model_name, constant.EXP_ROOT, args.model_id, model)

  for name, dataset in [(test_fname, data_gens[0])]:
    print('Processing... ' + name)
    total_gold_pred_pcls_ycls_ynoise = []
    total_annot_ids = []
    total_probs = []
    total_ys = []
    batch_attn = []
    for batch_num, batch in enumerate(dataset):
      print(batch_num)
      if not isinstance(batch, dict):
        print('==> batch: ', batch)
      eval_batch, annot_ids = to_torch(batch)
      #print('eval_batch')
      #for k, v in eval_batch.items():
      #  print(k, v.size())
      loss, output_logits, cls_logits = model(eval_batch, args.goal)
      #print('loss', loss)
      #print('output_logits', output_logits)
      #print('cls_logits', cls_logits)
      #batch_attn.append((batch, attn_score.data))
      output_index = get_output_index(output_logits, threshold=args.threshold)
      #print('output_index', output_index)
      #output_prob = model.sigmoid_fn(output_logits).data.cpu().clone().numpy()
      y = eval_batch['y'].data.cpu().clone().numpy()
      y_cls = eval_batch['y_cls'].data.cpu().clone().numpy()
      y_noisy_idx = eval_batch['y_noisy_idx'].data.cpu().clone().numpy()
      gold_pred_pcls_ycls_ynoise = get_gold_pred_str(output_index, y, args.goal, cls_logits=cls_logits, y_cls=y_cls, y_noisy_idx=y_noisy_idx)
      #print('gold_pred_pcls_ycls_ynoise', gold_pred_pcls_ycls_ynoise)
      #total_probs.extend(output_prob)
      #total_ys.extend(y)
      total_gold_pred_pcls_ycls_ynoise.extend(gold_pred_pcls_ycls_ynoise)
      total_annot_ids.extend(annot_ids)
    #mrr_val = mrr(total_probs, total_ys)
    #print('mrr_value: ', mrr_val)
    #pickle.dump({'gold_id_array': total_ys, 'pred_dist': total_probs},
    #            open('./{0:s}.p'.format(args.reload_model_name), "wb"))
    pickle.dump((total_annot_ids, total_gold_pred_pcls_ycls_ynoise),
                open('./{0:s}_gold_pred.p'.format(args.reload_model_name), "wb"))
    with open('./{0:s}.json'.format(args.model_id), 'w') as f_out:
      output_dict = {}
      if args.model_type == 'filter':
        for a_id, (gold, pred, cls, ycls, ynoise) in zip(total_annot_ids, total_gold_pred_pcls_ycls_ynoise):
          output_dict[a_id] = {"gold": gold, "pred": pred, "cls_pred": cls, "cls_gold": ycls, "y_noisy": ynoise}
      elif args.model_type == 'labeler':
        for a_id, (gold, pred) in zip(total_annot_ids, total_gold_pred_pcls_ycls_ynoise):
          output_dict[a_id] = {"gold": gold, "pred": pred}
      else:
        print('Invalid model type: -model_type ' + args.model_type)
        raise NotImplementedError
      json.dump(output_dict, f_out)
    eval_str = get_eval_string(list(zip(*list(zip(*gold_pred_pcls_ycls_ynoise))[:2])))
    print(eval_str)
    logging.info('processing: ' + name)
    logging.info(eval_str)


if __name__ == '__main__':
  config = config_parser.parser.parse_args()
  print(config)
  torch.cuda.manual_seed(config.seed)
  logging.basicConfig(
    filename=constant.EXP_ROOT +"/"+ config.model_id + datetime.datetime.now().strftime("_%m-%d_%H") + config.mode + '.txt',
    level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
  logging.info(config)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  if config.model_type in ['et_model', 'labeler', 'filter']:
    config.elmo = True
  elif config.model_type in ['bert_uncase_small']:
    config.bert = True
    config.bert_param_path = constant.BERT_UNCASED_SMALL_MODEL

  if config.mode == 'train':
    print('==> mode: train')
    _train(config)
  elif config.mode == 'test':
    print('==> mode: test')
    _test(config)
  elif config.mode == 'train_labeler':
    print('==> mode: train_labeler')
    _train_labeler(config)   # DEBUG
  elif config.mode == 'test_labeler':
    print('==> mode: test_labeler')
    _test_labeler(config)
  else:
    raise ValueError("invalid value for 'mode': {}".format(config.mode))
