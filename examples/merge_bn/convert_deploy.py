import numpy as np  
import sys,os  
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
import merge_bn as converter  
#train_proto = 'mobilenet_yolov3_test.prototxt'  
#train_model = 'mobilenet_yolov3_deploy_iter_55000.caffemodel'  #should be your snapshot caffemodel

#deploy_proto = 'remove_bn.prototxt'  
#save_model = 'remove_bn.caffemodel'
 
import remove_bn 


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('\nUsage:\n\npython convert_deploy.py [bn_model],[bn_weight],[output path]\n')
    else :
        train_proto = sys.argv[1]
        train_model = sys.argv[2]
        deploy_proto = '%s.prototxt'%str(sys.argv[3])
        save_model = '%s.caffemodel'%str(sys.argv[3])
        remove_bn.remove_bn_prototxt(train_proto,deploy_proto)
        net = caffe.Net(train_proto, train_model, caffe.TRAIN)  
        net_deploy = caffe.Net(deploy_proto, caffe.TEST)  

        converter.merge_bn(net, net_deploy)
        net_deploy.save(save_model)