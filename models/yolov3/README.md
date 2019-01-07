## Deploy model

The bias_term error was cause from dismatch caffemodel , please make sure your model was merged batchnorm or not 

* mobilenet_yolov3_lite_deploy.prototxt : merged 

* mobilenet_yolov3_lite_bn_deploy.prototxt : not merged

See issue :

https://github.com/eric612/MobileNet-YOLO/issues/42

https://github.com/eric612/MobileNet-YOLO/issues/50