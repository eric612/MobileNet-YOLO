#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log
./tools/caffe train --solver models/darknet_yolov3/solver.prototxt --weights models/darknet_yolov3/yolov3-spp.caffemodel --gpu=0 2>&1 | tee $LOG
#--weights models/darknet_yolov3/yolov3.caffemodel
#--snapshot models/darknet_yolov3/deploy_iter_1000.solverstate
#--weights models/darknet_yolov3/yolov3-spp.caffemodel