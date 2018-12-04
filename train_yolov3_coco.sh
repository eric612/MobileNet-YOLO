#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log
./tools/caffe train --solver models/yolov3_coco/mobilenet_yolov3_solver.prototxt --snapshot models/yolov3_coco/mobilenet_yolov3_deploy_iter_1000.solverstate --gpu=0 2>&1 | tee $LOG
#--weights models/MobileNet/mobilenet_iter_73000.caffemodel
#--snapshot models/yolov3_coco/mobilenet_yolov3_deploy_iter_78000.solverstate
#--weights models/MobileNet/MobileNet_deploy.caffemodel 