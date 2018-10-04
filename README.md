# MobileNet-YOLO Caffe

## MobileNet-YOLO 

A caffe implementation of MobileNet-YOLO detection network

Network|mAP|Resolution|Download|NetScope|Inference time (GTX 1080)|Inference time (i5-4440)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
MobileNet-YOLO-Lite|0.675|416|[deploy](models/yolov2/)|[graph](http://ethereon.github.io/netscope/#/gist/11229dc092ef68d3b37f37ce4d9cdec8)|N/A|N/A
MobileNet-YOLOv3-Lite|0.717|320|[deploy](models/yolov3/)|[graph](http://ethereon.github.io/netscope/#/gist/f308433ad8ba69e5a4e36d02482f8829)|6 ms|150 ms
MobileNet-YOLOv3-Lite|0.737|416|[deploy](models/yolov3/)|[graph](http://ethereon.github.io/netscope/#/gist/f308433ad8ba69e5a4e36d02482f8829)|11 ms|280 ms

Note :  the yolo_detection_output_layer not be optimization , and the deploy model was made by [merge_bn.py](https://github.com/chuanqi305/MobileNet-SSD/blob/master/merge_bn.py)

## Windows Version

[Caffe-YOLOv2-Windows](https://github.com/eric612/Caffe-YOLOv2-Windows)

### Oringinal darknet-yolov3

[Converter](models/darknet_yolov3) 

test on coco_minival_lmdb (IOU 0.5)

Network|mAP|Resolution|Download|NetScope|
:---:|:---:|:---:|:---:|:---:
yolov3|54.4|416|[caffemodel](https://drive.google.com/file/d/12nLE6GtmwZxDiulwdEmB3Ovj5xx18Nnh/view?usp=sharing)|[graph](http://ethereon.github.io/netscope/#/gist/59c75a50e5b91d6dd80a879df3cfaf55)
yolov3-spp|59.3|608|[caffemodel](https://drive.google.com/file/d/17b5wsR9tzbdrRnyL_iFEvofJ8jCmQ1ff/view?usp=sharing)|[graph](http://ethereon.github.io/netscope/#/gist/71edbfacf4d39c56f2d82cbcb739ae38)

## Performance

Compare with [YOLO](https://pjreddie.com/darknet/yolo/) 

Network|mAP|Weight size|Resolution
:---:|:---:|:---:|:---:
[MobileNet-YOLOv3-Lite](models/yolov3_coco/)|34.0*|[21.5 mb](https://drive.google.com/file/d/1bXZtB_wZBu1kOeagYtZgsjLq2CX0BGFD/view?usp=sharing)|320
[MobileNet-YOLOv3-Lite](models/yolov3_coco/)|37.3*|[21.5 mb](https://drive.google.com/file/d/1bXZtB_wZBu1kOeagYtZgsjLq2CX0BGFD/view?usp=sharing)|416
YOLOv3-Tiny|33.1|33.8 mb|416

* (*) testdev-2015 server was closed , here use coco 2014 minival

## Other models

You can find non-depthwise convolution network here , [Yolo-Model-Zoo](https://github.com/eric612/Yolo-Model-Zoo.git)

network|mAP|resolution|macc|param|
:---:|:---:|:---:|:---:|:---:|
PVA-YOLOv3|0.703|416|2.55G|4.72M|
Pelee-YOLOv3|0.703|416|4.25G|3.85M|

### CMake Build

[Caffe page](http://caffe.berkeleyvision.org/installation.html#compilation) , [dependency](https://docs.google.com/document/d/1n-WVIOrqadoIiRD-PW7RGb5LKOKP0y0Pms7svbZ3Zqw/edit?usp=sharing)

```
> git clone https://github.com/eric612/MobileNet-YOLO.git 
> cd $MobileNet-YOLO_root/
> mkdir build
> cd build
> cmake ..
> make -j4
```

## Training 

Download [lmdb](https://drive.google.com/open?id=19pBP1NwomDvm43xxgDaRuj_X4KubwuCZ)

Unzip into $caffe_root/ 

Please check the path exist "$caffe_root\examples\VOC0712\VOC0712_trainval_lmdb" and "$caffe_root\examples\VOC0712\VOC0712_test_lmdb"

Download [pre-trained weights](https://drive.google.com/file/d/141AVMm_h8nv3RpgylRyhUYb4w8rEguLM/view?usp=sharing) , and save at $caffe_root\model\convert

```
> cd $caffe_root/
> sh train_yolo.sh
```



## Demo

```
> cd $caffe_root/
> sh demo_yolo_lite.sh
```
If load success , you can see the image window like this 

![alt tag](00002.jpg)

### Future work 

* Distillation training 
* Openimages training and testing

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
