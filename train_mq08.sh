#!/bin/bash
# ====================================================
#   Copyright (C)2019 All rights reserved.
#
#   Author        : Eskimo
#   Email         : zhangfaninner@163.com
#   File Name     : train_mq08.sh
#   Last Modified : 2019-11-24 10:20
#   Describe      :
#
# ====================================================
rm -rf /mnt/disk/RL/zf/data/MQ2008/Buffered*
python3 l2r.py -dataset MQ2008_super -dir_data /mnt/disk/RL/zf/data/MQ2008/ -epoch 50 -fold 1 -discount 0.9 -n_head 1 -model "PagewiseRank" -weight_decay 0.0005 -discount_pagewise 0.3 -page 5
