#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log
./tools/caffe train --solver models/mobilenetv1_voc/mobilenet_yolov3_lite_solver.prototxt --weights models/MobileNet/MobileNet_deploy.caffemodel --gpu=0 2>&1 | tee $LOG

