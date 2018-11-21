# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import sys
import cv2
#import matplotlib.pyplot as plt
# display plots in this notebook

# set display defaults
#plt.rcParams['figure.figsize'] = (10, 10)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = './'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
import math
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

caffe.set_mode_cpu()
model_def = sys.argv[2]
model_weights = sys.argv[3]
#model_def = 'MobileNetSSD_deploy.prototxt'
#model_weights = './gnet_yolo_region_darknet_anchor_iter_32000.caffemodel'
#model_weights = 'MobileNetSSD_deploy.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


mu = np.array([0.5, 0.5, 0.5])
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 1.0)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          320, 320)  # image size is 227x227


def vis_detections(image,result) :
    w = image.shape[1]
    h = image.shape[0]
    for i in range(result.shape[1]):
        left = result[0][i][3] * w
        top = result[0][i][4] * h
        right = result[0][i][5] * w
        bot = result[0][i][6] * h
        score = result[0][i][2]
        label = result[0][i][1]
        if(score>0.5) :
            print(left,right,top,bot,score,label)
            cv2.rectangle(image,(int(left), int(top)),(int(right),int(bot)),(0,255,0), 2)

            label = '{:s} {:.3f}'.format(CLASSES[int(label)], score)
            font = cv2.FONT_HERSHEY_SIMPLEX
            size = cv2.getTextSize(label, font, 0.5, 0)[0]
            cv2.rectangle(image,(int(left), int(top)),
                    (int(left+size[0]),int(top+ size[1])),(0,255,0), -1)

            cv2.putText(image, label,(int(left), int(top+ size[1])),font,0.5,(0,0,0),0)

def det(image, pic):
    
    transformed_image = transformer.preprocess('data', image)
    #plt.imshow(image)

    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()

    res = output['detection_out'][0]  # the output probability vector for the first image in the batch
    #print(res.shape)
    return res


    
pic = sys.argv[1]
image = caffe.io.load_image(pic)
image_show =cv2.imread(pic)  
result = det(image, pic)
vis_detections(image_show,result)
cv2.imshow("Image", image_show)
cv2.waitKey (0)