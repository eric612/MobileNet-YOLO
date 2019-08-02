CODE UPDATED FOR OPENCV 3

# MobileNet-YOLO Caffe

This project also support ssd framework , and here lists the difference from ssd caffe

* Multi-scale training , you can select input resoluton when inference
* Modified from last update caffe (2018)
* Support multi-task model 
* [pelee + driverable map](data/bdd100k)
 
## MobileNet-YOLO 

A caffe implementation of MobileNet-YOLO detection network , train on 07+12 , test on VOC2007

Network|mAP|Resolution|Download|NetScope|Inference time (GTX 1080)|Inference time (i5-4440)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
MobileNetV2-YOLOv3|70.4|352|[caffemodel](models/mobilenetv2_voc/yolo_lite)|[graph](http://ethereon.github.io/netscope/#/gist/495618dacbfca0ed2256cce9bf221b1f)|[6.09 ms](benchmark/test.log)|170 ms

* inference time was log from [script](benchmark/test_yolov3_lite.sh) , does not include pre-processing 
* the [benchmark](/benchmark) of cpu performance on Tencent/ncnn  framework
* the deploy model was made by [merge_bn.py](https://github.com/chuanqi305/MobileNet-SSD/blob/master/merge_bn.py) , or you can try my custom [version](examples/merge_bn/)

### CNN Analyzer

Use this [tool](https://dgschwend.github.io/netscope/quickstart.html) to compare macc and param , train on 07+12 , test on VOC2007

network|mAP|resolution|macc|param|
:---:|:---:|:---:|:---:|:---:|
MobileNetV2-YOLOv3|0.704|352|1.22G|4.05M|
[Pelee-SSD](https://github.com/Robert-JunWang/Pelee)|0.709|304|1.2G|5.42M|

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
