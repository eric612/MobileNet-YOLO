#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log
./tools/caffe train --solver models/pelee/solver.prototxt --snapshot models/pelee/pelee_SSD_304x304_iter_150000.solverstate --gpu=0 2>&1 | tee $LOG
#--weights models/peleenetv2_inet_288_7243.caffemodel
#--snapshot models/pelee/Fisheye/SSD_304x304/pelee_SSD_304x304_iter_40000.solverstate
#--weights models/MobileNet/MobileNet_deploy.caffemodel 
#--snapshot models/fisheye/mobilenet_yolov3_deploy_iter_2000.solverstate