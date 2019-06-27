import argparse
import multiprocessing
import json
import spacy
from spacy import displacy
from multiprocessing import Pool

nlp = spacy.load('en')

parser = argparse.ArgumentParser()
parser.add_argument("-read_from", help="data path", default="")
parser.add_argument("-save_to", help="new data path", default="")
config = parser.parse_args()

def load_json(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [json.loads(line.strip()) for line in lines]

def save_json(path, data):
    with open(path, 'w') as f:
        for d in data:
            json.dump(d, f)
            f.write('\n')

def get_tree(sent):
    doc = nlp(sent)
    tree = []
    for token in doc:
        d = {}
        d['text'] = token.text
        d['lemma'] = token.lemma_ 
        d['pos'] = token.pos_
        d['tag'] = token.tag_ 
        d['dep'] = token.dep_
        d['shape'] = token.shape_ 
        d['is_alpha'] = token.is_alpha
        d['is_stop'] = token.is_stop
        tree.append(d)
    return tree

def add_tree(x):
    #print('processing: ' + x['annot_id'])
    x['mention_span_tree'] = get_tree(x['mention_span'].strip())
    return x
   

if __name__ == '__main__':
    print('==> loading data from: ' + config.read_from)
    data = load_json(config.read_from)
    cores_to_use = 20 #multiprocessing.cpu_count()
    print('==> using ' + str(cores_to_use) + ' cores')
    p = Pool(cores_to_use)
    data_w_tree = p.map(add_tree, data)
    print('==> saving data to: ' + config.save_to)
    save_json(config.save_to, data_w_tree)
