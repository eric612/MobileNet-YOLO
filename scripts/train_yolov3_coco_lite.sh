#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log
./build/tools/caffe train --solver models/yolov3_coco/mobilenet_yolov3_lite_solver.prototxt --weights models/MobileNet/mobilenet_yolov3_deploy_iter_78000.caffemodel --gpu=0 2>&1 | tee $LOG
#pre-trained weights downloads : https://drive.google.com/file/d/1tVdLzBA5T_HjDQkJv2ldr99X-T_s5UMn/view?usp=sharing

#--weights models/MobileNet/mobilenet_iter_73000.caffemodel
#--snapshot models/yolov3_coco/mobilenet_yolov3_lite_deploy_iter_37000.solverstate
#--weights models/MobileNet/MobileNet_deploy.caffemodel 
#--weights models/MobileNet/mobilenet_yolov3_deploy_iter_56000.caffemodel