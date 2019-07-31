CODE UPDATED FOR OPENCV 3

# MobileNet-YOLO Caffe

This project also support ssd framework , and here lists the difference from ssd caffe

* Multi-scale training , you can select input resoluton when inference
* Modified from last update caffe (2018)
* Support multi-task model 
* [pelee + driverable map](data/bdd100k)
 
## MobileNet-YOLO 

A caffe implementation of MobileNet-YOLO detection network , first train on COCO trainval35k then fine-tune on 07+12 , test on VOC2007

Network|mAP|Resolution|Download|NetScope|Inference time (GTX 1080)|Inference time (i5-4440)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
MobileNet-YOLOv3-Lite|74.6|320|[caffemodel](models/yolov3)|[graph](http://ethereon.github.io/netscope/#/gist/8f218ac764fd14c74c5eac1549a088af)|[4.79 ms](benchmark/test-320.log)|150 ms
MobileNet-YOLOv3-Lite|76.3|416|[caffemodel](models/yolov3)|[graph](http://ethereon.github.io/netscope/#/gist/8f218ac764fd14c74c5eac1549a088af)|[6.52 ms](benchmark/test-416.log)|280 ms

* inference time was log from [script](benchmark/test_yolov3_lite.sh) , does not include pre-processing 
* the [benchmark](/benchmark) of cpu performance on Tencent/ncnn  framework
* the deploy model was made by [merge_bn.py](https://github.com/chuanqi305/MobileNet-SSD/blob/master/merge_bn.py) , or you can try my custom [version](examples/merge_bn/)
* bn_model download [here](https://drive.google.com/open?id=15vMupzg4GDATUuNWDQQI1GFQUR8_5puo) 

### Coverted TensorRT models

[TensorRT-Yolov3-models](https://github.com/eric612/TensorRT-Yolov3-models)

[Pelee-Driverable_Maps](https://github.com/eric612/Pelee-Seg-TensorRT), run 89 ms on [jetson nano](https://github.com/eric612/Jetson-nano-benchmark)

### YOLO Segmentation

[How to use](https://github.com/eric612/MobileNet-YOLO/tree/master/data/cityscapes)

## Windows Version

[Caffe-YOLOv3-Windows](https://github.com/eric612/Caffe-YOLOv2-Windows)

### Oringinal darknet-yolov3

[Converter](models/darknet_yolov3) 

test on coco_minival_lmdb (IOU 0.5)

Network|mAP|Resolution|Download|NetScope|
:---:|:---:|:---:|:---:|:---:
yolov3|54.2|416|[caffemodel](https://drive.google.com/file/d/1nYgjOg8o48qQ3Cw47CamERgJVgLlo-Cu/view?usp=sharing)|[graph](http://ethereon.github.io/netscope/#/gist/59c75a50e5b91d6dd80a879df3cfaf55)
yolov3-spp|59.8|608|[caffemodel](https://drive.google.com/file/d/1eEFXWPFnCt6fWtmS6zTsPkAQgW0VFkt7/view?usp=sharing)|[graph](http://ethereon.github.io/netscope/#/gist/71edbfacf4d39c56f2d82cbcb739ae38)

## Performance

Train on  COCO trainval35k (2014) , and  compare with [YOLO](https://pjreddie.com/darknet/yolo/) , (IOU 0.5)

Network|IOU 0.5:0.95|IOU 0.5|IOU 0.75|Weight size|Resolution|NetScope|Resize Mode
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
[MobileNet-YOLOv3-Lite](models/yolov3_coco/)|19.9|35.5|19.6|[22.0 mb](https://drive.google.com/file/d/1rruY8BtS8WVdKPwU0LIT_6FyTnVxvHQl/view?usp=sharing)|320|[graph](http://ethereon.github.io/netscope/#/gist/110f5f5a2edad80c0c9074c7a532347b)|WARP
[MobileNet-YOLOv3-Lite](models/yolov3_coco/)|21.5|38.9|21.2|[22.0 mb](https://drive.google.com/file/d/1rruY8BtS8WVdKPwU0LIT_6FyTnVxvHQl/view?usp=sharing)|416|[graph](http://ethereon.github.io/netscope/#/gist/110f5f5a2edad80c0c9074c7a532347b)|WARP
[MobileNet-YOLOv3](models/yolov3_coco/)|22.7|40.2|22.6|[22.5 mb](https://drive.google.com/file/d/1tVdLzBA5T_HjDQkJv2ldr99X-T_s5UMn/view?usp=sharing)|416|[graph](http://ethereon.github.io/netscope/#/gist/ef69b621d69703be0327836ec9708634)|LetterBox
YOLOv3-Tiny||33.1||33.8 mb|416
[MobileNet-YOLOv3-Lite-trt](https://github.com/eric612/TensorRT-Yolov3-model)||37.5||[23.5 mb](https://github.com/eric612/TensorRT-Yolov3-model/blob/master/mobilenet_yolov3_lite_deploy_iter_21000.caffemodel)|416|[graph](http://ethereon.github.io/netscope/#/gist/cf22541c314a5896abd45b88d430ca1f)|WARP
* (*) testdev-2015 server was closed , here use coco 2014 minival
* MobileNet-YOLOv3-Lite-trt was the fastest model

## Other Models

You can find non-depthwise convolution network here , [Yolo-Model-Zoo](https://github.com/eric612/Yolo-Model-Zoo.git)

network|mAP|resolution|macc|param|
:---:|:---:|:---:|:---:|:---:|
PVA-YOLOv3|0.703|416|2.55G|4.72M|
Pelee-YOLOv3|0.703|416|4.25G|3.85M|

### Model VisulizationTool

Supported on [Netron](https://github.com/lutzroeder/netron) , [browser](https://lutzroeder.github.io/netron/) version

### Build , Run and Training

See [wiki](https://github.com/eric612/MobileNet-YOLO/wiki)


## License and Citation


Please cite MobileNet-YOLO in your publications if it helps your research:

    @article{MobileNet-YOLO,
      Author = {eric612 , Avisonic , ELAN},
      Year = {2018}
    }
    
## Reference

> https://github.com/weiliu89/caffe/tree/ssd

> https://pjreddie.com/darknet/yolo/

> https://github.com/gklz1982/caffe-yolov2

> https://github.com/yonghenglh6/DepthwiseConvolution

> https://github.com/alexgkendall/caffe-segnet

> https://github.com/BVLC/caffe/pull/6384/commits/4d2400e7ae692b25f034f02ff8e8cd3621725f5c

> https://www.cityscapes-dataset.com/

> https://github.com/TuSimple/tusimple-benchmark/wiki

> https://github.com/Robert-JunWang/Pelee

> https://github.com/hujie-frank/SENet

Cudnn convolution

> https://github.com/chuanqi305/MobileNetv2-SSDLite/tree/master/src
