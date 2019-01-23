# MobileNet-YOLO Caffe

## MobileNet-YOLO benchmark

By using tencent/[ncnn](https://github.com/Tencent/ncnn) framework 

ubuntu18 , intel i5-7500

```
loop_count = 8
num_threads = 4
powersave = 0
      squeezenet  min =   27.47  max =   27.62  avg =   27.52
       mobilenet  min =   21.06  max =   22.03  avg =   21.60
    mobilenet_v2  min =   15.84  max =   16.63  avg =   16.16
      shufflenet  min =    7.65  max =    8.54  avg =    7.88
       googlenet  min =  132.11  max =  134.41  avg =  132.82
        resnet18  min =  197.74  max =  201.52  avg =  198.61
         alexnet  min =   77.14  max =   79.23  avg =   77.58
           vgg16  min = 1130.89  max = 1135.79  avg = 1132.22
  squeezenet-ssd  min =   92.09  max =   93.14  avg =   92.70
   mobilenet-ssd  min =   49.84  max =   50.99  avg =   50.39
mobilenet-yolov3-320  min =   56.83  max =   57.72  avg =   57.34
mobilenet-yolov3-416  min =   94.59  max =   98.39  avg =   95.84
```

## Model download

[MobileNet-YOLOv3-Lite-VOC](https://github.com/eric612/MobileNet-YOLO/blob/master/models/yolov3/mobilenet_yolov3_lite_deploy.caffemodel)

[Darknet-YOLOv3](https://drive.google.com/file/d/12nLE6GtmwZxDiulwdEmB3Ovj5xx18Nnh/view)

## inference time issue

The most time consuming layer were group convolution , not deconvolution

## int8 inference mAP list

https://github.com/eric612/caffe-int8-convert-tools

Network|mAP|Resolution|Data Type|Framework|Bit
:---:|:---:|:---:|:---:|:---:|:---:
MobileNet-YOLOv3|0.737|416|float|caffe|32
MobileNet-YOLOv3|0.729|416|float|ncnn|32
MobileNet-YOLOv3|0.634|416|int|ncnn|8
MobileNet-YOLOv3|0.668|416|mixed precision : remove conv0|ncnn|8

## To do

* ["A Quantization-Friendly Separable Convolution for MobileNets"](https://arxiv.org/pdf/1803.08607.pdf)
* [Ristretto ](http://lepsucd.com/?page_id=621)
