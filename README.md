## MobileNet-YOLO Caffe

A caffe implementation of MobileNet-YOLO detection network , train on 07+12 , test on VOC2007

Network|mAP|Resolution|Download|NetScope|Inference time (GTX 1080)|Inference time (i5-7500)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
MobileNetV2-YOLOv3|71.5|352|[caffemodel](models/mobilenetv2_voc/yolo_lite)|[graph](http://ethereon.github.io/netscope/#/gist/495618dacbfca0ed2256cce9bf221b1f)|[6.65 ms](benchmark/test.log)|217 ms

* inference time was log from [script](benchmark/test_yolov3_lite.sh) , does not include pre-processing 
* the [benchmark](/benchmark) of cpu performance on Tencent/ncnn  framework
* the deploy model was made by [merge_bn.py](https://github.com/Robert-JunWang/Pelee/blob/master/tools/gen_merged_model.py), set eps = your prototxt batchnorm eps
* old models please see [here](https://github.com/eric612/MobileNet-YOLO/tree/83827a038efdd891f4d01bf711e520bf2539c036)

This project also support ssd framework , and here lists the difference from ssd caffe

* Multi-scale training , you can select input resoluton when inference
* Modified from last update caffe (2018)
* Support multi-task model 
* [pelee + driverable map](data/bdd100k)

## Update

* The pytorch version was released [Mobilenet-YOLO-Pytorch](https://github.com/eric612/Mobilenet-YOLO-Pytorch)

### CNN Analyzer

Use this [tool](https://dgschwend.github.io/netscope/quickstart.html) to compare macc and param , train on 07+12 , test on VOC2007

network|mAP|resolution|macc|param|pruned|IOU_THRESH|GIOU
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
MobileNetV2-YOLOv3|0.707|352|1.22G|4.05M|N|N|N|
MobileNetV2-YOLOv3|[0.715](https://drive.google.com/open?id=1YBJOhc14qfOALf2R5kWaDTa1p722xD2Z)|352|1.22G|4.05M|N|Y|Y|
MobileNetV2-YOLOv3|0.702|352|1.01G|2.88M|Y|N|N|
[Pelee-SSD](https://github.com/Robert-JunWang/Pelee)|0.709|304|1.2G|5.42M|N|N|N|
[Mobilenet-SSD](https://github.com/chuanqi305/MobileNet-SSD)|0.68|300|1.21G|5.43M|N|N|N|
[MobilenetV2-SSD-lite](models/mobilenetv2_voc/ssd_lite)|0.709|336|1.10G|[5.2M](https://drive.google.com/open?id=1Lb9LJOrl5fYZ7Mp65beBQ44d6cH_vlbv)|N|N|N|

* MobileNetV2-YOLOv3 and MobilenetV2-SSD-lite were not offcial model

### Coverted TensorRT models

[TensorRT-Yolov3-models](https://github.com/eric612/TensorRT-Yolov3-models)

[Pelee-Driverable_Maps](https://youtu.be/nndFtIPMy20), run 89 ms on [jetson nano](https://github.com/eric612/Jetson-nano-benchmark) , [running project](https://github.com/eric612/Pelee-Seg-TensorRT)

### YOLO Segmentation

[How to use](https://github.com/eric612/MobileNet-YOLO/tree/master/data/cityscapes)

## Windows Version

[Caffe-YOLOv3-Windows](https://github.com/eric612/Caffe-YOLOv2-Windows)

### Oringinal darknet-yolov3

[Converter](models/darknet_yolov3) 

test on coco_minival_lmdb (IOU 0.5)

Network|mAP|Resolution|Download|NetScope|
:---:|:---:|:---:|:---:|:---:
yolov3|54.2|416|[caffemodel](https://drive.google.com/file/d/1wfajkXsZcTurXkCUPpftNaOLfxbboOMt/view?usp=sharing)|[graph](http://ethereon.github.io/netscope/#/gist/59c75a50e5b91d6dd80a879df3cfaf55)
yolov3-spp|59.8|608|[caffemodel](https://drive.google.com/file/d/15vRLuvHYdWVux4wdjbVZIPLfdGe-vLVH/view?usp=sharing)|[graph](http://ethereon.github.io/netscope/#/gist/71edbfacf4d39c56f2d82cbcb739ae38)


### Model VisulizationTool

Supported on [Netron](https://github.com/lutzroeder/netron) , [browser](https://lutzroeder.github.io/netron/) version

### Build , Run and Training

See [wiki](https://github.com/eric612/MobileNet-YOLO/wiki)

See [docker](https://hub.docker.com/r/eric612/mobilenet-yolo)

## License and Citation


Please cite MobileNet-YOLO in your publications if it helps your research:

    @article{MobileNet-YOLO,
      Author = {eric612 , Avisonic , ELAN},
      Year = {2018}
    }
    
## Reference

> https://github.com/weiliu89/caffe/tree/ssd

> https://pjreddie.com/darknet/yolo/

> https://github.com/chuanqi305/MobileNet-SSD

> https://github.com/gklz1982/caffe-yolov2

> https://github.com/yonghenglh6/DepthwiseConvolution

> https://github.com/alexgkendall/caffe-segnet

> https://github.com/BVLC/caffe/pull/6384/commits/4d2400e7ae692b25f034f02ff8e8cd3621725f5c

> https://www.cityscapes-dataset.com/

> https://github.com/TuSimple/tusimple-benchmark/wiki

> https://github.com/Robert-JunWang/Pelee

> https://github.com/hujie-frank/SENet

> https://github.com/lusenkong/Caffemodel_Compress

Cudnn convolution

> https://github.com/chuanqi305/MobileNetv2-SSDLite/tree/master/src

## Acknowledgements

https://github.com/AlexeyAB/darknet
