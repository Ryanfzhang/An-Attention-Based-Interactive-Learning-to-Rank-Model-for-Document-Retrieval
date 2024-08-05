#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C)2019 All rights reserved.
#
#   Author        : Eskimo
#   Email         : zhangfaninner@163.com
#   File Name     : ListNet.py
#   Last Modified : 2019-09-18 21:01
#   Describe      :
#
# ====================================================

import sys
# import os

import torch
import os
import numpy as np
from torch import nn
from torch import optim
from org.l2r_global import L2R_GLOBAL
device = L2R_GLOBAL.global_device

class ListNet(object):

    def __init__(self, args, id = None, ranking_function = None, opt = "Adam", lr = 1e-3, weight_decay = 1e-3):
        self.ranking_function = rf(args)
        self.opt = opt
        self.lr = lr
        self.weight_decay = weight_decay
        self.init_optimizer()

    def reset_parameters(self):
        pass

    def init_optimizer(self):
        if "Adam" == self.opt:
            self.optimizer = optim.Adam(self.ranking_function.get_parameters(), lr=self.lr, weight_decay=self.weight_decay)  # use regularization
        elif 'RMS' == self.opt:
            self.optimizer = optim.RMSprop(self.ranking_function.get_parameters(), lr=self.lr, weight_decay=self.weight_decay)  # use regularization
        else:
            raise NotImplementedError

    def train(self, batch_ranks, batch_std , train=True, sp = False):
        self.ranking_function.train()
        ranking = self.ranking_function(batch_ranks, batch_std)

        if not sp:
            self.optimizer.zero_grad()
            self.ranking_function.loss.backward()
            self.optimizer.step()
        else:
            self.optimizer.zero_grad()
            self.ranking_function.sp_loss.backward()
            self.optimizer.step()
        return self.ranking_function.loss

    def predict(self, batch_ranks, batch_std, train=False):
        self.ranking_function.eval()
        with torch.no_grad():
            ranking = self.ranking_function(batch_ranks, batch_std)
        return ranking 

    def save_model(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(self.ranking_function.state_dict(), dir+'/'+name+'.torch')

    def load(self, name):
        self.ranking_function= torch.load(name)

class rf(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.w = nn.Sequential(
                nn.Linear(args.feature_size,1),)

    def forward(self, batch_ranks, label):
        score = self.w(batch_ranks).squeeze(-1)
        score = nn.functional.softmax(score, -1)
        label = nn.functional.softmax(label, -1)
        self.loss = -torch.sum(label * torch.log(score))

        return score

    def get_parameters(self):
        return list(self.parameters())
