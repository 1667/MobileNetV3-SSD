#!/bin/sh
python run_ssd_example.py \
    mb3-ssd-lite models/mb3-ssd-lite.pth models/voc-model-labels.txt \
    /home/grobot/mywork/firedetection/objectdetection/testfire/1_21.jpg
