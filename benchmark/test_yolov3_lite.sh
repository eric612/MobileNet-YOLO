#!/bin/bash
LOG=log/test-`date +%Y-%m-%d-%H-%M-%S`.log
build/tools/caffe time -gpu 0 -model benchmark/mobilenet_yolov3_lite_deploy.prototxt --gpu=0 2>&1 | tee $LOG
#-model examples/merge_bn/trt.prototxt