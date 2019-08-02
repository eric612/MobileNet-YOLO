#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log
./tools/caffe train --solver models/pelee/VOC0712/SSD_304x304/solver.prototxt --weights models/peleenetv2_inet_288_7243.caffemodel --gpu=0 2>&1 | tee $LOG
#--weights models/MobileNet/mobilenet_iter_73000.caffemodel
#--snapshot models/cityscapes/mobilenet_yolov3_deploy_iter_4000.solverstate
#--weights models/MobileNet/MobileNet_deploy.caffemodel 
#--snapshot models/fisheye/mobilenet_yolov3_deploy_iter_2000.solverstate