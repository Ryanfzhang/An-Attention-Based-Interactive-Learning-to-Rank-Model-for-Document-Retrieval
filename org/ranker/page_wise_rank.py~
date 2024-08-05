#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C)2020 All rights reserved.
#
#   Author        : Eskimo
#   Email         : zhangfaninner@163.com
#   File Name     : page_wise_rank.py
#   Last Modified : 2020-02-09 15:21
#   Describe      :
#
# ====================================================

import sys
# import os
import torch
from torch import nn
import os
import numpy as np
import torch.nn.functional as F
from torch import optim
import math

from org.l2r_global import L2R_GLOBAL
device = L2R_GLOBAL.global_device

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_k, d_v, dropout = 0.1):
        super().__init__() #params
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_input = d_input
        self.dropout = dropout

        self.w_q = nn.Linear(d_input, n_head*d_k)
        self.w_k = nn.Linear(d_input, n_head*d_k)
        self.w_v = nn.Linear(d_input, n_head*d_v)
        self.layer_norm = nn.LayerNorm(d_input)
        self.fc = nn.Linear(n_head*d_v, d_input)
        self.Dropout = nn.Dropout(self.dropout)

        self.weight_init()
    def weight_init(self):
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0/ (self.d_input + self.d_k)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0/ (self.d_input + self.d_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0/ (self.d_input + self.d_k)))
        nn.init.xavier_normal_(self.fc.weight)
    
    def forward(self, x):
        q, k, v = x, x, x
        bs, seq_len_q, _ = q.shape
        bs, seq_len_k, _ = k.shape 
        bs, seq_len_v, _ = v.shape 
        
        residual = q
        q = self.w_q(q).view(bs, seq_len_q, self.n_head, self.d_k)
        k = self.w_k(k).view(bs, seq_len_k, self.n_head, self.d_k)
        v = self.w_v(v).view(bs, seq_len_v, self.n_head, self.d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_q, self.d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_k, self.d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_v, self.d_v)
        attention = torch.matmul(q, k.transpose(1,2))#nb, seq, seq 
        attention = attention/np.power(self.d_k,0.5)
        attention = F.softmax(attention, dim=2)
        output = torch.matmul(attention, v)
        
        output = output.view(self.n_head, bs, seq_len_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(bs, seq_len_q, -1)

        output = self.Dropout(self.fc(output))
        output = self.layer_norm(residual+ output)
        return output
 
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class Encoder(nn.Module):   
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.stacked_layer= nn.Sequential()
        for i in range(args.num_layers):
            self.stacked_layer.add_module("MultiHeadAttention_{}".format(i), MultiHeadAttention(args.n_head, args.hidden_size, args.hidden_size, args.hidden_size))
            self.stacked_layer.add_module("PositionwiseFeedForward_{}".format(i), PositionwiseFeedForward(args.hidden_size, args.hidden_size))

    def forward(self, x):
        output = self.stacked_layer(x)
        return output

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args        
        self.input_linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.context_linear = nn.Conv1d(args.hidden_size, args.hidden_size, 1, 1)
        self.V = nn.Parameter(torch.FloatTensor(args.hidden_size), requires_grad=True)
        self._inf = nn.Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)

        self.tanh= nn.Tanh()
        self.softmax = nn.Softmax(dim =-1)
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input, context, mask):

        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1)) #(b, e, s)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)#(b, e, s)

        V= self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)#(b, 1, e)
        att = torch.bmm(V, self.tanh(inp+ctx)).squeeze(1)


        att = torch.where(mask, att, self.inf)
        alpha = self.softmax(att)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha
    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)



class PageWiseRank(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fnn = nn.Sequential(
            nn.Linear(args.feature_size, args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.Sigmoid(),
            )
        self.Encoder = Encoder(args)
        self.w = nn.Parameter(torch.rand(2*args.hidden_size),requires_grad=True)
        self.gru = nn.GRUCell(args.hidden_size, args.hidden_size)

        #parameters
        self.h0 = nn.Parameter(torch.zeros(args.hidden_size), requires_grad= False)
        self.mask = nn.Parameter(torch.ones(1), requires_grad = False)
        self._zero = nn.Parameter(torch.zeros(1), requires_grad = False)
        self._inf = nn.Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self._init()

    def _init(self):
        for name, m in self.named_modules():
            m.apply(self.weight_init)
        torch.nn.init.uniform_(self.w.data)

    def weight_init(self,m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data , 0)
        

    def reward_ndcg(self, label, position):
        return (2.**label.float()-1)/np.log(2.+position)
    def forward(self, batch_ranks, labels):
        #初始化
        bs, seq_len, _ = batch_ranks.size()
        page_size = self.args.page
        batch_doc_reprs = self.fnn(batch_ranks)
        hidden_size = batch_doc_reprs.size(2)
        num_page = seq_len//page_size
        last_page_size = seq_len-num_page*page_size

        inf = self._inf.unsqueeze(0).expand(bs, seq_len)
        zero = self._zero.unsqueeze(0).unsqueeze(0).expand(bs, seq_len, hidden_size)
        mask = self.mask.unsqueeze(0).expand(bs, seq_len).bool()
        runner = torch.arange(seq_len).unsqueeze(0).expand(bs, -1).to(device)
        permuted_doc_label, pros_list, reward_list, indices=[], [], [], []

        h_t = self.h0.unsqueeze(0).expand(bs, -1)
        t=0

        while t<num_page:
            mask_batch_doc_reprs= torch.where(mask.unsqueeze(2).expand(-1,-1, hidden_size), batch_doc_reprs, zero)
            outs= self.Encoder(mask_batch_doc_reprs)
            score = torch.matmul(torch.cat([outs, h_t.unsqueeze(1).expand(-1, seq_len,-1)],dim =2), self.w)
            score = nn.functional.softmax(score, -1)
            score  = torch.where(mask , score, inf)
            _pros, _indices = score.sort(dim = 1, descending=True)
            for i in range(page_size):
                pro, indice = _pros[:,i], _indices[:,i]
                one_hot_pointer = (runner== indice.unsqueeze(1).expand(-1, outs.size(1)))
                mask = mask*(~one_hot_pointer)
                embedding_mask = one_hot_pointer.unsqueeze(2).expand(-1, -1, hidden_size)
                doc = torch.masked_select(outs, embedding_mask).view(bs,-1)
                label = torch.masked_select(labels, one_hot_pointer).view(bs)
                permuted_doc_label.append(label)
                pros_list.append(pro)
                reward_list.append((self.args.discount_pagewise**t)*(self.args.discount**i)* self.reward_ndcg(label, i))
                indices.append(indice)
                h_t = self.gru(doc,h_t)
            t=t+1

        mask_batch_doc_reprs= torch.where(mask.unsqueeze(2).expand(-1,-1, hidden_size), batch_doc_reprs, zero)
        outs= self.Encoder(mask_batch_doc_reprs)
        score = torch.matmul(torch.cat([outs, h_t.unsqueeze(1).expand(-1, seq_len,-1)],dim =2), self.w)
        score = nn.functional.softmax(score, -1)
        score  = torch.where(mask , score, inf)
        _pros, _indices = score.sort(dim = 1,descending=True)
        for i in range(last_page_size):
            pro, indice = _pros[:,i], _indices[:,i]
            one_hot_pointer = (runner== indice.unsqueeze(1).expand(-1, outs.size(1)))
            mask = mask*(~one_hot_pointer)
            embedding_mask = one_hot_pointer.unsqueeze(2).expand(-1, -1, hidden_size)
            doc = torch.masked_select(outs, embedding_mask).view(bs,-1)
            label = torch.masked_select(labels, one_hot_pointer).view(bs)
            permuted_doc_label.append(label)
            pros_list.append(pro)
            reward_list.append(self.args.discount**i* self.reward_ndcg(label, i))
            indices.append(indice)

        assert torch.sum(mask)==0, print(mask)

        probs = torch.stack(pros_list, dim =1)
        self.indices = torch.stack(indices, dim =1)
        rewards = torch.stack(reward_list,dim =1)
        rewards = rewards.float().to(device)

        page_gamma = torch.arange(num_page+1).unsqueeze(0).unsqueeze(2).expand(bs, -1, page_size).contiguous().view(bs, -1)[:,:seq_len].float().to(device)

        rewards = rewards*(self.args.discount ** page_gamma).float().to(device)
        r = torch.cumsum( rewards.flip((1)), dim =1).flip((1))
        R = (r*(1/self.args.discount_pagewise)**page_gamma).float().to(device)
        self.loss = -torch.sum( R*torch.log(probs))

        return 0.9**(torch.sort(self.indices, dim = -1, descending=False)[1].float())

    def get_parameters(self):
        return list(self.parameters())

    '''
    def reward_ndcg(self, label_list):
        return (2**label_list[-1]-1.)/np.log(len(label_list)+1.)
    '''

