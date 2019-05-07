import sys
sys.path.insert(0,"python")
import caffe
 
#model="models/bvlc_alexnet/deploy.prototxt"
#model="models/yolov3/mobilenet_yolov3_lite_deploy.prototxt"
model="examples/pelee/model/voc/deploy_merged.prototxt"

def main():
    net=caffe.Net(model,caffe.TEST)
    params=0
    flops=0
    blobs=net.blobs
    print("name param flops")
    idx = 0
    '''
    while idx < len(net.layers) :
      layer = net.layers[idx]
      c1=layer[0].count

      idx = idx + 1
    '''
    for item in net.params.items():
        name,layer=item
        print(name)
        size = len(layer)
        param = 0
        for i in range(size):
          param = param + layer[i].count
        b=blobs[name]
        flop=param*b.width*b.height
        print(name+" "+str(param)+" "+str(flop))
        params+=param
        flops+=flop
    print("total params","{0:.2f}".format(params/float(1000000))+"m")
    print("FLOPs:","{0:.2f}".format(flops/float(1000000000))+"g")
    
if __name__ == '__main__':
    main()