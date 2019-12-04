#!/bin/bash
python3 train_ssd.py --dataset_type voc --datasets /home/grobot/mywork/deeplearning/myyolo/yolov3/data --net mb3-ssd-lite --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001 --batch_size 5
