## Deploy model

network|mAP|resolution|macc|param|pruned
:---:|:---:|:---:|:---:|:---:|:---:|
MobileNetV2-YOLOv3|0.707|352|1.22G|4.05M|N|
MobileNetV2-YOLOv3|0.702|352|1.01G|2.88M|Y|

The bias_term error was cause from dismatch caffemodel , please make sure your model was merged batchnorm or not 

* yolov3_m2.prototxt : merged 

* yolov3_lite_deploy.prototxt : not merged

* yolov3_lite_deploy_pruned.prototxt : not merged and pruned channel

See issue :

https://github.com/eric612/MobileNet-YOLO/issues/42

https://github.com/eric612/MobileNet-YOLO/issues/50

# Download weights

[yolov3_m2](https://drive.google.com/open?id=18SL6tnGCUap6LCfqX2sT11cZnF6ECZ_j)

[yolov3_lite_deploy](https://drive.google.com/open?id=1FIoG2HSSuVQ-c4JOVZFvCTpp-CPPzJlJ)

[yolov3_lite_deploy_pruned](https://drive.google.com/open?id=1w3j-a7pFiYiTVTtDNRTWGWI84MQ3JewG)

# Pre-trained ilsvrc2012 weights

[MobileNetV2](https://drive.google.com/open?id=1RgT7jrYFJ0Ub_CF5zdfP3ExwHKDftPik)

Converted from [PytorchToCaffe](https://github.com/xxradon/PytorchToCaffe)
