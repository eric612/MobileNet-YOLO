#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log
./tools/caffe train --solver models/tusimple/mobilenet_yolov3_lite_solver.prototxt --weights models/MobileNet/MobileNet_deploy.caffemodel    --gpu=0 2>&1 | tee $LOG
#--weights models/MobileNet/mobilenet_iter_73000.caffemodel
#--snapshot models/tusimple/mobilenet_yolov3_lite_deploy_iter_16000.solverstate
#--weights models/MobileNet/mobilenet_yolov3_lite_deploy_iter_59000.caffemodel
#--weights models/MobileNet/MobileNet_deploy.caffemodel 
