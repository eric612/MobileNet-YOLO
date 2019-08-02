#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log
./tools/caffe train --solver models/mobilenetv2_voc/yolo_lite/solver.prototxt --weights models/mobilenetv2_voc/MobileNetV2.caffemodel --gpu=0 2>&1 | tee $LOG

