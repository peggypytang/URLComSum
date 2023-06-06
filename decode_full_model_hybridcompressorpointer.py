""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
from os.path import join, exists
from datetime import timedelta
from time import time
import pickle as pkl
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op
from torch.autograd import Variable
from cytoolz import identity, concat, curry
import numpy as np
import torch
from torch.utils.data import DataLoader

from torch import multiprocessing as mp, nn
from toolz.sandbox import unzip
#from utils import make_vocab
#from data.batcher import tokenize
from DocumentHybridExtractiveCompressivePointer import DocumentHybridExtractiveCompressivePointer
from datasets import load_dataset

try:
    DATA_DIR = os.environ['DATA']

except KeyError:
    print('please use environment variable to specify data directories')

def collate_func(inps):
    return [a for a in inps]

def decode(dataset_str, dataset_doc_field, save_path, model_dir, model_name, split, batch_size, cuda, embedding_dim, hidden_dim, tkner):
    start = time()
    summarizer = DocumentHybridExtractiveCompressivePointer(tkner, embedding_dim, hidden_dim)
    print("summarizer", summarizer)
    summarizer.cuda()

    summarizer.load_state_dict(torch.load(join(model_dir,model_name)))


    if dataset_str == "cnn_dailymail":
        dataset = load_dataset(dataset_str, '3.0.0', split='test')
    elif dataset_str == "newsroom":
        dataset = load_dataset(dataset_str, data_dir="/home/ptan6545/newsroom_complete", split='test')
    elif args.dataset_str == "wikihow":
        dataset = load_dataset('wikihow', 'all', data_dir="/home/ptan6545/wikiHow", split='test')
    elif args.dataset_str == "reddit_tifu":
        dataset = load_dataset('reddit_tifu', 'long')['test']
    else:
        dataset = load_dataset(dataset_str, split='test')
    
    n_data = len(dataset)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # prepare save paths and logs
    if not exists(join(save_path, 'outputExt')):
        os.makedirs(join(save_path, 'outputExt'))

    if not exists(join(save_path, 'outputExtPointer')):
        os.makedirs(join(save_path, 'outputExtPointer'))

    if not exists(join(save_path, 'output')):
        os.makedirs(join(save_path, 'output'))

    if not exists(join(save_path, 'outputCompPointer')):
        os.makedirs(join(save_path, 'outputCompPointer'))

    print("loader length", len(dataloader))
    # Decoding
    i = 0

    with torch.no_grad():
        for ib, documents in enumerate(dataloader):

            bodies = documents[dataset_doc_field]

            print("bodies", bodies)
            if bodies[0]:
                ext_sampled_summaries, _, ext_sampled_idx_batch, comp_sampled_summaries, _, comp_sampled_idx_batch = summarizer.forward(bodies, args.max_ext_output_length, args.max_comp_output_length)

                #sampled_summaries, _, sampled_pointers = summarizer.forward(bodies, args.max_ext_output_length, args.max_comp_output_length)

            
                print("decoded", comp_sampled_summaries[0])

                #print("decoded ext_arts_w", ext_arts_w)
                with open(join(save_path, 'outputExt/{}.dec'.format(i)),'w') as f:
                    #f.write(', '.join(str(x) for x in ext_sent))
                    f.write(ext_sampled_summaries[0])

                with open(join(save_path, 'outputExtPointer/{}.dec'.format(i)),'w') as f:
                    f.write(', '.join(str(x) for x in ext_sampled_idx_batch))

                with open(join(save_path, 'output/{}.dec'.format(i)),'w') as f:
                    #f.write(', '.join(str(x) for x in ext_sent))
                    f.write(comp_sampled_summaries[0])

                with open(join(save_path, 'outputCompPointer/{}.dec'.format(i)),'w') as f:
                    f.write(', '.join(str(x) for x in comp_sampled_idx_batch))
            else:
                with open(join(save_path, 'output/{}.dec'.format(i)),'w') as f:
                    #f.write(', '.join(str(x) for x in ext_sent))
                    f.write('')
                

            i += 1
            print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                i, n_data, i/n_data*100,
                timedelta(seconds=int(time()-start))
            ), end='')
    print()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--path', required=True, help='path to store/eval')
    #parser.add_argument('--model_dir', help='root of the full model')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')
    parser.add_argument('--model_dir', required=True, help='path to ext model')
    parser.add_argument('--model_name', required=True, help='ext model')
    parser.add_argument("--dataset_str", type=str, default="cnn_dailymail") ## cnn_dailymail, newsroom, gigaword, xsum
    parser.add_argument("--dataset_doc_field", type=str, default="article") ##cnn_dailymail = article, newsroom=text, gigaword=document, xsum = document

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=1,
                        help='batch size of faster decoding')
    parser.add_argument("--max_ext_output_length", type=int, default=46, help="Maximum output length. Saves time if the sequences are short.")
    parser.add_argument("--max_comp_output_length", type=int, default=46, help="Maximum output length. Saves time if the sequences are short.")
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument("--embedding_dim", type=int, default=300, help="Embedding size")
    parser.add_argument("--hidden_dim", type=int, default=int(300/2), help="Hidden size")
    parser.add_argument("--lstm_layers", type=int, default=3, help="LSTM layers")
    parser.add_argument("--tkner", type=str, default="bert", help="bert or w2v")

    args = parser.parse_args()
    #args.cuda = torch.cuda.is_available() and not args.no_cuda
    print("torch.cuda.is_available()", torch.cuda.is_available())
    args.cuda = True

    data_split = 'test' if args.test else 'val'
    decode(args.dataset_str, args.dataset_doc_field, args.path, args.model_dir, args.model_name,
           data_split, args.batch, 
           args.cuda, args.embedding_dim, args.hidden_dim, args.tkner)
