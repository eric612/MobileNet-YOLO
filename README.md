# MobileNet-YOLO Caffe

## MobileNet-YOLO 

A caffe implementation of MobileNet-YOLO (YOLOv2 base) detection network, with pretrained weights on VOC0712 and mAP=0.718

Network|mAP|Download|Download|NetScope
:---:|:---:|:---:|:---:|:---:
MobileNet-YOLO-Lite|0.667|[train](models/MobileNet/mobilenet_iter_73000.caffemodel)|[deploy](models/yolov2/mobilenet_yolo_deploy_iter_95000.caffemodel)|[graph](http://ethereon.github.io/netscope/#/gist/11229dc092ef68d3b37f37ce4d9cdec8)
MobileNet-YOLO|0.718|[train](models/MobileNet/mobilenet_iter_73000.caffemodel)|[deploy](models/yolov2/mobilenet_yolo_deploy_iter_81000.caffemodel)|[graph](http://ethereon.github.io/netscope/#/gist/52f298d84f8fa4ebb2bb94767fa6ca88)
## Performance

Compare with [YOLOv2](https://pjreddie.com/darknet/yolov2/)

Network|mAP|Weight size|Inference time (GTX 1080)
:---:|:---:|:---:|:---:
MobileNet-YOLO-Lite|0.66|22.8 mb|11 ms
MobileNet-YOLO|0.71|41.4 mb|22 ms
Tiny-YOLO|0.57|60.5 mb|N/A
YOLOv2|0.76|193 mb|N/A

Note :  the yolo_detection_output_layer not be optimization

## Training 

Download [lmdb](https://drive.google.com/open?id=19pBP1NwomDvm43xxgDaRuj_X4KubwuCZ)

Unzip into $caffe_root/ 

Please check the path exist "$caffe_root\examples\VOC0712\VOC0712_trainval_lmdb" and "$caffe_root\examples\VOC0712\VOC0712_test_lmdb"

```
> cd $caffe_root/
> sh train_yolo.sh
```

## Demo

```
> cd $caffe_root/
> sh demo_yolo.sh
```
If load success , you can see the image window like this 

![alt tag](00002.jpg)

## Maintenance

I'll appreciate  if you can help me to 

1. Miragrate to [modivius neural compute stick](https://github.com/eric612/YoloV2-MobileNet-NCS)
2. Mobilenet upgrade to v2 or model tunning

## Caffe 

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Custom distributions

 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, SKX, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
