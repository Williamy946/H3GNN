import argparse
import pickle
import random
import time

import torch

from util import Data, split_validation
from model import *
from datetime import datetime as dt

import os

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='xing', help='dataset name: diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=300, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--max_hist_len', type=int, default=1000, help='max history length')
parser.add_argument('--layer', type=int, default=1, help='the number of layer used')
parser.add_argument('--pair_layer', type=int, default=2, help='the number of pair-wise graph layer used')
parser.add_argument('--ishybrid', action="store_true", help='w/o global graph')
parser.add_argument('--isfixeduser', action="store_true", help='w/o fixed user embedding')
parser.add_argument('--ishyper', action="store_true", help='w/o hypergraph embeddings (dynamic user emb)')
parser.add_argument('--isnorm', action="store_true", help='normalize embeddings after updating')
parser.add_argument('--ishist', action="store_true", help='historical session')
parser.add_argument('--hybrid',type=int, default=0, help='0 for attention, 1 for Cascade, 2 for MLP, 3 for Addition, 4 for Mean')
parser.add_argument('--encoder',type=int, default=0, help='0 for full, 1 for w/o position, 2 for w/o last')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')
parser.add_argument('--device', type=int, default=1, help='gpu device')

opt = parser.parse_args()

opt.isfixeduser = False # True : Without user embedding information, False : With user embedding
opt.ishybrid = True # True : Enable Hybrid Mode (Mix Global and HyperGNN), False : Disable Hybrid Mode
opt.ishyper = True # True : Use item embedding from HyperGNN, False: Use item embedding from Global GNN
opt.isnorm = True # True : Enable Normalize Embedding, False : Disable it

def random_sample(data, ratio):
    sam_size = int(len(data[0])*ratio)
    ind = random.sample(range(len(data[0])), sam_size)
    sampled_data = [[data[i][index] for index in ind] for i in range(3)]
    return sampled_data

def main():

    if opt.dataset == 'lastfm':
        n_node = 50000
        n_user = 1000
        opt.lr = 0.0005
        opt.layer = 1
        opt.embSize = 300
    elif opt.dataset == 'xing':
        n_node = 59122
        n_user = 11479
        opt.lr = 0.001
        opt.embSize = 200
        opt.layer = 5
        opt.pair_layer = 2
    elif opt.dataset == 'reddit':
        n_node = 27453
        n_user = 18271
        opt.layer = 1
        opt.pair_layer = 2
        opt.embSize = 200
    else:
        n_node = 309

    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))

    val_data = random_sample(test_data, 0.1)

    train_data = Data(train_data, opt, shuffle=True, n_node=n_node, n_user=n_user, max_hist_len=opt.max_hist_len)

    test_data = Data(test_data, opt, shuffle=True, n_node=n_node, n_user=n_user, max_hist_len=opt.max_hist_len,
                     train_hist=train_data.train_hist, train_edge=train_data.edge_item)

    val_data = Data(val_data, opt, shuffle=True, n_node=n_node, n_user=n_user, max_hist_len=opt.max_hist_len,
                    train_hist=train_data.train_hist, train_edge=train_data.edge_item)

    print("Data preprocess done!")
    model = trans_to_cuda(H3GNN(opt=opt, adjacency=train_data.adjacency, adjacency_2=train_data.adjacency_2, u_adj=train_data.u_adj,
                                global_A=train_data.global_A,
                                n_node=n_node, n_user=n_user, lr=opt.lr, l2=opt.l2,
                                layers=opt.layer, emb_size=opt.embSize, batch_size=opt.batchSize, dataset=opt.dataset))

    top_K = [5, 10, 20]
    best_results = {}
    best_perform = 0
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]
    print(opt)
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss, best_perform = train_test(model, epoch, train_data, val_data, test_data, best_perform)

        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
        if best_results['metric10'][0] < metrics['hit10']:
            for K in top_K:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))


if __name__ == '__main__':
    main()
