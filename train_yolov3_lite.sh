#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log
./tools/caffe train --solver models/yolov3/mobilenet_yolov3_lite_solver.prototxt --weights models/MobileNet/mobilenet_iter_73000.caffemodel --gpu=0 2>&1 | tee $LOG
#--weights models/MobileNet/mobilenet_iter_73000.caffemodel
#--snapshot models/yolov3/mobilenet_yolo_deploy_iter_1000.solverstate