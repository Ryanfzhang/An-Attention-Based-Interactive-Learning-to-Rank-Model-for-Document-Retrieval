#!/bin/bash
# ====================================================
#   Copyright (C)2019 All rights reserved.
#
#   Author        : Eskimo
#   Email         : zhangfaninner@163.com
#   File Name     : train.sh
#   Last Modified : 2019-09-25 20:43
#   Describe      :
#
# ====================================================

rm -rf /mnt/disk/zf/data/MQ2007/Buffered*

python3 l2r.py -dataset MQ2007_super -dir_data /mnt/disk/zf/data/MQ2007/ -epoch 100 -fold 1 -discount 0.9 -discount_pagewise 0.6 -num_layers 1 -n_head 1 -model "PagewiseRank" -weight_decay 0.0005 -page 5
