# MobileNet-YOLO Caffe

## MobileNet-YOLO benchmark

By using tencent/[ncnn](https://github.com/Tencent/ncnn) framework 

ubuntu18 , intel i5-7500

4 thread

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

1 thread

```

loop_count = 8
num_threads = 1
powersave = 0
gpu_device = -1
          squeezenet  min =   40.83  max =   41.40  avg =   41.07
     squeezenet_int8  min =   82.48  max =   83.96  avg =   83.26
           mobilenet  min =   61.35  max =   62.40  avg =   61.64
      mobilenet_int8  min =  177.86  max =  179.57  avg =  178.61
        mobilenet_v2  min =  147.33  max =  151.00  avg =  149.29
          shufflenet  min =   56.88  max =   58.81  avg =   57.32
             mnasnet  min =  151.08  max =  152.61  avg =  151.85
     proxylessnasnet  min =  175.62  max =  178.84  avg =  177.13
           googlenet  min =  152.17  max =  155.21  avg =  153.19
      googlenet_int8  min =  262.90  max =  265.88  avg =  264.00
            resnet18  min =  110.29  max =  111.74  avg =  111.01
       resnet18_int8  min =  187.22  max =  190.55  avg =  188.93
             alexnet  min =  228.68  max =  232.15  avg =  230.26
               vgg16  min = 1250.01  max = 1262.71  avg = 1256.51
          vgg16_int8  min = 1118.23  max = 1129.76  avg = 1124.70
            resnet50  min =  289.05  max =  293.84  avg =  290.92
       resnet50_int8  min =  696.78  max =  705.94  avg =  702.75
      squeezenet_ssd  min =  219.35  max =  223.17  avg =  221.46
 squeezenet_ssd_int8  min =  141.96  max =  144.01  avg =  142.69
       mobilenet_ssd  min =  215.80  max =  217.38  avg =  216.57
  mobilenet_ssd_int8  min =  348.32  max =  352.69  avg =  349.76
      mobilenet_yolo  min =  287.34  max =  292.10  avg =  290.14
    mobilenet_yolov3  min =  290.87  max =  294.93  avg =  292.03
  mobilenetv2_yolov3  min =  150.22  max =  151.17  avg =  150.80
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
