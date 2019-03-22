/*
* @Author: Eric612
* @Date:   2019-01-29
* @https://github.com/eric612/Caffe-YOLOv2-Windows
* @https://github.com/eric612/MobileNet-YOLO
* Avisonic , ELAN microelectronics
*/

#include <cmath>
#include <vector>

#include "caffe/layers/yolo_seg_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
template <typename Dtype>
void YoloSegLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
    "YoloSeg layer inputs must have the same count.";
  //LOG(INFO)<<bottom[1]->width()<<","<<bottom[1]->height()<<","<<bottom[1]->num()<<","<<bottom[1]->channels();
  if (diff_.width() != bottom[0]->width()) {
    diff_.ReshapeLike(*bottom[0]);
    swap_.ReshapeLike(*bottom[0]);
  }

  
  const int count = bottom[0]->count();
  const Dtype* input_data = bottom[0]->gpu_data();	
  Dtype* diff = diff_.mutable_gpu_data();
  Dtype* swap = swap_.mutable_gpu_data();
  caffe_gpu_logistic_activate(count,input_data ,swap );
  caffe_copy(count,swap,diff);
  

  //visualization(bottom,top);	
	Forward_cpu(bottom,top);	
}


template <typename Dtype>
void YoloSegLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const Dtype sign(1.);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
    //const Dtype alpha(1.0);
    //LOG(INFO) << "alpha:" << alpha;

    caffe_gpu_axpby(
      bottom[0]->count(),
      alpha,
      diff_.gpu_data(),
      Dtype(0),
      bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(YoloSegLayer);


}  // namespace caffe
