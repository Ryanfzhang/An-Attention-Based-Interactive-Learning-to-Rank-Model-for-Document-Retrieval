#!/bin/bash
# ====================================================
#   Copyright (C)2019 All rights reserved.
#
#   Author        : Eskimo
#   Email         : zhangfaninner@163.com
#   File Name     : train_mq07_2.sh
#   Last Modified : 2019-12-05 20:24
#   Describe      :
#
# ====================================================


rm -rf /home/ryan/data/MQ2007/Buffered*
python l2r.py -dataset MQ2007_super -dir_data /home/ryan/data/MQ2007/ -epoch 100 -fold 1 -discount 0.9 -num_layers 1 -n_head 8 -model "attRNN" -weight_decay 0.0005
python l2r.py -dataset MQ2007_super -dir_data /home/ryan/data/MQ2007/ -epoch 100 -fold 1 -discount 0.9 -num_layers 1 -n_head 4 -model "attRNN" -weight_decay 0.0005
