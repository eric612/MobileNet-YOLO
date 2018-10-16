import sys,os  
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')  
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
from caffe import layers as L
from caffe import params as P
import caffe
net = caffe_pb2.NetParameter()
from google.protobuf import text_format
#fn = 'mobilenet_yolov3_test.prototxt'
def remove_bn_prototxt(fn,filename) :
    with open(fn) as f:
        s = f.read()
        txtf.Merge(s, net)

    net.name = 'remove_bn'
    #layerNames = [l.name for l in net.layer]
    idx = 0
    del_list = list()
    #print(len(net.layer))
    while idx < len(net.layer) :

        if 'bn' in net.layer[idx].name :
            print('delete %s'%net.layer[idx].name)
            del net.layer[idx]
        elif 'scale' in net.layer[idx].name :
            print('delete %s'%net.layer[idx].name)
            del net.layer[idx]
        elif 'conv' in net.layer[idx].name and 'relu' not in net.layer[idx].name and 'sum' not in net.layer[idx].name:
            #print(net.layer[idx].convolution_param.bias_term)
            #net.layer[idx].include.remove(net.layer[idx].param[0])
            #print(net.layer[idx].include.remove)
            net.layer[idx].param[0].lr_mult = int(1)
            net.layer[idx].param[0].decay_mult = int(1)
            net.layer[idx].convolution_param.bias_term = True
            idx+=1
        else :
            idx+=1

    outFn = filename
    print 'writing', outFn
    with open(outFn, 'w') as f:
        f.write(str(net))
    
