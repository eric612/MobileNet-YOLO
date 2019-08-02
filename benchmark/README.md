# MobileNet-YOLO Caffe

## MobileNet-YOLO benchmark

By using tencent/[ncnn](https://github.com/Tencent/ncnn) framework 

ubuntu18 , intel i5-7500

```
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
          squeezenet  min =   13.00  max =   15.19  avg =   13.72
     squeezenet_int8  min =   24.32  max =   24.98  avg =   24.55
           mobilenet  min =   38.82  max =   39.51  avg =   39.21
      mobilenet_int8  min =   49.59  max =   52.74  avg =   50.41
        mobilenet_v2  min =   43.54  max =   47.73  avg =   44.38
          shufflenet  min =   21.72  max =   22.24  avg =   21.97
             mnasnet  min =   38.37  max =   41.76  avg =   39.25
     proxylessnasnet  min =   52.03  max =   53.03  avg =   52.42
           googlenet  min =   44.69  max =   46.46  avg =   45.21
      googlenet_int8  min =   74.33  max =   78.75  avg =   75.33
            resnet18  min =   35.36  max =   36.02  avg =   35.60
       resnet18_int8  min =   52.20  max =   55.73  avg =   52.95
             alexnet  min =   52.79  max =   59.08  avg =   53.93
               vgg16  min =  168.00  max =  174.34  avg =  170.31
          vgg16_int8  min =  279.29  max =  289.84  avg =  284.04
            resnet50  min =   92.58  max =   93.89  avg =   93.28
       resnet50_int8  min =  197.26  max =  215.15  avg =  201.12
      squeezenet_ssd  min =   34.18  max =   34.85  avg =   34.33
 squeezenet_ssd_int8  min =   43.44  max =   44.66  avg =   43.92
       mobilenet_ssd  min =   48.47  max =   50.70  avg =   49.56
  mobilenet_ssd_int8  min =   99.36  max =  113.17  avg =  101.96
      mobilenet_yolo  min =  100.28  max =  103.80  avg =  101.04
    mobilenet_yolov3  min =   85.21  max =   90.68  avg =   86.62
  mobilenetv2_yolov3  min =   58.12  max =   60.43  avg =   58.91
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
