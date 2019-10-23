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
net2 = caffe_pb2.NetParameter()
#fn = 'mobilenet_yolov3_test.prototxt'

def Data(name, tops, source, batch_size, phase):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Data'
    layer.top.extend(tops)
    layer.data_param.source = source
    layer.data_param.batch_size = batch_size
    layer.data_param.backend = caffe_pb2.DataParameter.LMDB

    return layer
    
def remove_bn_prototxt(fn,filename) :
    with open(fn) as f:
        s = f.read()
        txtf.Merge(s, net)

    net.name = 'remove_bn'
    net2.name = 'remove_bn'
    data = net2.layer.add()
    
    #layers = []
    #layers.append(Data('data', ['data', 'label'],
    #                   'examples/imagenet/ilsvrc12_train_lmdb', 32, 'train'))
                       
    #data.CopyFrom(layers)
    #layerNames = [l.name for l in net.layer]
    idx = 0
    del_list = list()
    #print(len(net.layer))
    top = 0 
    delflag = False
    while idx < len(net.layer) :
        #print(idx)
        #print(net.layer[idx].type)
        if 'BatchNorm' in net.layer[idx].type :
            #print('delete %s'%net.layer[idx].name)
            #if 'ReLU' not in net.layer[idx+2].type :
                #print(net.layer[idx-1].name)
            #    net.layer[idx].bottom[0] = top
            #else :
            #    top = net.layer[idx].bottom[0]
            #print(net.layer[idx+2].type)
            #top = net.layer[idx].bottom[0]
            del net.layer[idx]
            delflag = True
        elif 'Scale' in net.layer[idx].type :
            #print('delete %s'%net.layer[idx].name)
            #top = net.layer[idx].bottom[0]
            del net.layer[idx]
            #delflag = True
        elif 'Convolution' in net.layer[idx].type and 'ReLU' not in net.layer[idx].type :
            #print(net.layer[idx].convolution_param.bias_term)
            #net.layer[idx].include.remove(net.layer[idx].param[0])
            #if 'ReLU' not in net.layer[idx-1].type :
            #    print(net.layer[idx-1].name)
            if delflag == True :
                net.layer[idx].bottom[0] = top
            #net.layer[idx].param[0].lr_mult = int(1)
            #net.layer[idx].param[0].decay_mult = int(1)
            top = net.layer[idx].top[0]
            #print(net.layer[idx].name)
            net.layer[idx].convolution_param.bias_term = True         
            #if delflag == True :
            #    net.layer[idx].bottom[0] = top
            #print(top)
            idx+=1
            delflag = False
            '''
        elif 'data' in net.layer[idx].name :
            print('delete %s'%net.layer[idx].name)
            #del net.layer[idx]
            #data = net.layer.add()
            net.layer[idx].type = 'data'
            del net.layer[idx].top[1]
            del net.layer[idx].include
            #net.layer[idx].include.remove(net.layer[idx].include[0])
            #net.layer[idx].transform_param.remove()
            #data.top[0] = 'data'
            #net.layer[idx].CopyFrom(L.Data(net,ntop=1))
            idx += 1
            '''
        elif 'Power' in net.layer[idx].type :
            if delflag == True :
                net.layer[idx].bottom[0] = top
            idx+=1
        elif 'Eltwise' in net.layer[idx].type :
            if delflag == True : #for mobilenetv2
                if 'batch_norm_blob' in net.layer[idx].bottom[0] :
                    #print(net.layer[idx].bottom[0])
                    net.layer[idx].bottom[0] = net.layer[idx].bottom[0].replace('batch_norm','conv');
                net.layer[idx].bottom[1] = top
            idx+=1
            delflag = False
        elif 'Concat' in net.layer[idx].type :
            #if delflag == True :
            #    net.layer[idx].bottom[0] = top
            idx+=1
            delflag = False
        else :
            if delflag == True :
                net.layer[idx].bottom[0] = top
            #print(top)
            idx+=1
            delflag = False

    outFn = filename
    print 'writing', outFn
    with open(outFn, 'w') as f:
        f.write(str(net))
    
