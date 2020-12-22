# -*- coding: utf-8 -*-
"""
Description: SACN from github.com/JD-AI-Research-Silicon-Valley/SACN,
             Use DGL to implement GraphConvolution.
"""

import argparse
import math
import dgl
import numpy as np
import time
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.knowledge_graph import load_data
from torch.nn.init import xavier_normal_
from dgl import function as fn
from torch.utils.data import DataLoader

from evaluation_dgl import ranking_and_hits
from utils import EarlyStopping
from batch_prepare import TrainBatchPrepare, EvalBatchPrepare


# GCN
class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_relations, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.num_relations = num_relations
        self.alpha = torch.nn.Embedding(num_relations + 1, 1, padding_idx=0)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, all_edge_type, input):
        with g.local_scope():
            feats = torch.mm(input, self.weight)
            g.srcdata['ft'] = feats
            # match `A = A + A.transpose(0, 1)`
            train_edge_num = int((all_edge_type.shape[0] - input.shape[0]) / 2)
            transpose_all_edge_type = torch.cat((all_edge_type[train_edge_num:train_edge_num * 2],
                                                 all_edge_type[:train_edge_num], all_edge_type[-input.shape[0]:]))
            # alp = self.alpha(torch.cat([all_edge_type, all_edge_type])).reshape(-1, 1)
            alp = self.alpha(all_edge_type) + self.alpha(transpose_all_edge_type)
            g.edata['a'] = alp

            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))

            output = g.dstdata['ft']

            if self.bias is not None:
                return output + self.bias
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# SACN
class SACN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, args):
        super(SACN, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, args.init_emb_size, padding_idx=0)
        self.gc1 = GraphConvolution(args.init_emb_size, args.gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(args.gc1_emb_size, args.embedding_dim, num_relations)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout_rate)
        self.feature_map_drop = torch.nn.Dropout(args.dropout_rate)
        self.loss = torch.nn.BCELoss()
        # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.conv1 = nn.Conv1d(2, args.channels, args.kernel_size, stride=1,
                               padding=int(math.floor(args.kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(args.channels)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(args.embedding_dim * args.channels, args.embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(args.gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(args.init_emb_size)

        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)

    def forward(self, g, all_edge, e1, rel, entity_id):
        emb_initial = self.emb_e(entity_id)
        x = self.gc1(g, all_edge, emb_initial)

        x = self.bn3(x)
        x = torch.tanh(x)
        x = F.dropout(x, args.dropout_rate, training=self.training)
        x = self.bn4(self.gc2(g, all_edge, x))

        e1_embedded_all = torch.tanh(x)
        e1_embedded_all = F.dropout(e1_embedded_all, args.dropout_rate, training=self.training)
        e1_embedded = e1_embedded_all[e1]
        rel_embedded = self.emb_rel(rel)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)

        x = self.conv1(x)
        x = self.bn1(x)

        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred


def process_triplets(triplets, all_dict, num_rels):
    """
    process triples, store the id of all entities corresponding to (head, rel)
    and (tail, rel_reverse) into dict
    """
    data_dict = {}
    for i in range(triplets.shape[0]):
        e1, rel, e2 = triplets[i]
        rel_reverse = rel + num_rels

        if (e1, rel) not in data_dict:
            data_dict[(e1, rel)] = set()
        if (e2, rel_reverse) not in data_dict:
            data_dict[(e2, rel_reverse)] = set()

        if (e1, rel) not in all_dict:
            all_dict[(e1, rel)] = set()
        if (e2, rel_reverse) not in all_dict:
            all_dict[(e2, rel_reverse)] = set()

        all_dict[(e1, rel)].add(e2)
        all_dict[(e2, rel_reverse)].add(e1)

        data_dict[(e1, rel)].add(e2)
        data_dict[(e2, rel_reverse)].add(e1)

    return data_dict


def preprocess_data(train_data, valid_data, test_data, num_rels):
    all_dict = {}

    train_dict = process_triplets(train_data, all_dict, num_rels)
    valid_dict = process_triplets(valid_data, all_dict, num_rels)
    test_dict = process_triplets(test_data, all_dict, num_rels)

    return train_dict, valid_dict, test_dict, all_dict


def main(args):
    # load graph data
    data = load_data(args.dataset)
    num_nodes = data.num_nodes
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels

    stopper = EarlyStopping(patience=args.patience)

    # check cuda
    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    # create model
    model = SACN(num_entities=num_nodes, num_relations=num_rels * 2 + 1, args=args)

    # build graph
    g = dgl.graph([])
    g.add_nodes(num_nodes)
    src, rel, dst = train_data.transpose()
    # add reverse edges, reverse relation id is between [num_rels, 2*num_rels)
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    # get new train_data with reverse relation
    train_data_new = np.stack((src, rel, dst)).transpose()

    # unique train data by (h,r)
    train_data_new_pandas = pandas.DataFrame(train_data_new)
    train_data_new_pandas = train_data_new_pandas.drop_duplicates([0, 1])
    train_data_unique = np.asarray(train_data_new_pandas)

    g.add_edges(src, dst)
    # add graph self loop
    g.add_edges(g.nodes(), g.nodes())
    # add self loop relation type, self loop relation's id is 2*num_rels.
    rel = np.concatenate((rel, np.ones([num_nodes]) * num_rels * 2))
    print(g)
    entity_id = torch.LongTensor([i for i in range(num_nodes)])

    model = model.to(device)
    g = g.to(device)
    all_rel = torch.LongTensor(rel).to(device)
    entity_id = entity_id.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # process the triples and get all tails corresponding to (h,r)
    # here valid_dict and test_dict are not used.
    train_dict, valid_dict, test_dict, all_dict = preprocess_data(train_data, valid_data, test_data, num_rels)

    train_batch_prepare = TrainBatchPrepare(train_dict, num_nodes)

    # eval needs to use all the data in train_data, valid_data and test_data
    eval_batch_prepare = EvalBatchPrepare(all_dict, num_rels)

    train_dataloader = DataLoader(
        dataset=train_data_unique,
        batch_size=args.batch_size,
        collate_fn=train_batch_prepare.get_batch,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    valid_dataloader = DataLoader(
        dataset=valid_data,
        batch_size=args.batch_size,
        collate_fn=eval_batch_prepare.get_batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        collate_fn=eval_batch_prepare.get_batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)

    # training loop
    print("start training...")
    for epoch in range(args.n_epochs):
        model.train()
        epoch_start_time = time.time()
        for step, batch_tuple in enumerate(train_dataloader):
            e1_batch, rel_batch, labels_one_hot = batch_tuple
            e1_batch = e1_batch.to(device)
            rel_batch = rel_batch.to(device)
            labels_one_hot = labels_one_hot.to(device)
            labels_one_hot = ((1.0 - 0.1) * labels_one_hot) + (1.0 / labels_one_hot.size(1))

            pred = model.forward(g, all_rel, e1_batch, rel_batch, entity_id)
            optimizer.zero_grad()
            loss = model.loss(pred, labels_one_hot)
            loss.backward()
            optimizer.step()

        print("epoch : {}".format(epoch))
        print("epoch time: {:.4f}".format(time.time() - epoch_start_time))
        print("loss: {}".format(loss.data))

        model.eval()
        if epoch % args.eval_every == 0:
            with torch.no_grad():
                val_mrr = ranking_and_hits(g, all_rel, model, valid_dataloader, 'dev_evaluation', entity_id, device)
            if stopper.step(val_mrr, model):
                break

    print("training done")
    model.load_state_dict(torch.load('es_checkpoint.pt'))
    ranking_and_hits(g, all_rel, model, test_dataloader, 'test_evaluation', entity_id, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SACN')
    parser.add_argument("--init_emb_size", type=int, default=100,
                        help="initial embedding size")
    parser.add_argument("--gc1_emb_size", type=int, default=150,
                        help="embedding size after gc1")
    parser.add_argument("--embedding_dim", type=int, default=200,
                        help="embedding dim")
    parser.add_argument("--input_dropout", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--dropout_rate", type=float, default=0.2,
                        help="dropout rate")
    parser.add_argument("--channels", type=int, default=200,
                        help="number of channels")
    parser.add_argument("--kernel_size", type=int, default=5,
                        help="kernel size")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=5000,
                        help="number of training epochs")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("-d", "--dataset", type=str, default="FB15k-237",
                        help="dataset to use")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop patience")

    args = parser.parse_args()
    print(args)
    main(args)
