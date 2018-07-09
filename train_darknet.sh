#!/bin/bash
LOG=log/darknet-train-`date +%Y-%m-%d-%H-%M-%S`.log
./tools/caffe train --solver models/darknet/darknet19_solver.prototxt --weights=models/convert/darknet19_conv.caffemodel --gpu=0 2>&1 | tee $LOG
#--weights=examples/conv18.caffemodel
#--snapshot models/yolov2/yolov2_deploy_iter_5000.solverstate