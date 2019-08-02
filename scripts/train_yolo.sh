#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log
./build/tools/caffe train --solver models/yolov2/mobilenet_yolo_solver.prototxt --weights models/MobileNet/mobilenet_iter_73000.caffemodel --gpu=0 2>&1 | tee $LOG
#--weights models/MobileNet/mobilenet_iter_73000.caffemodel
#--snapshot models/yolov2/mobilenet_yolo_deploy_iter_5000.solverstate