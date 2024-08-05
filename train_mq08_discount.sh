#!/bin/bash
# ====================================================
#   Copyright (C)2019 All rights reserved.
#
#   Author        : Eskimo
#   Email         : zhangfaninner@163.com
#   File Name     : train_mq08_discount.sh
#   Last Modified : 2019-11-28 21:00
#   Describe      :
#
# ====================================================


rm -rf /home/ryan/data/MQ2008/Buffered*
python l2r.py -dataset MQ2008_super -dir_data /home/ryan/data/MQ2008/ -epoch 50 -fold 1 -discount 0.6 -n_head 1 -model "attRNN" -weight_decay 0.0005
python l2r.py -dataset MQ2008_super -dir_data /home/ryan/data/MQ2008/ -epoch 50 -fold 1 -discount 0.7 -n_head 1 -model "attRNN" -weight_decay 0.0005
python l2r.py -dataset MQ2008_super -dir_data /home/ryan/data/MQ2008/ -epoch 50 -fold 1 -discount 0.8 -n_head 1 -model "attRNN" -weight_decay 0.0005
python l2r.py -dataset MQ2008_super -dir_data /home/ryan/data/MQ2008/ -epoch 50 -fold 1 -discount 1 -n_head 1 -model "attRNN" -weight_decay 0.0005
