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
  if (diff_.width() != bottom[0]->width()) {
    diff_.ReshapeLike(*bottom[0]);
  }
  const int count = bottom[0]->count();
  const Dtype* input_data = bottom[0]->gpu_data();	
  const Dtype* label_data = bottom[1]->cpu_data(); 
  Dtype* diff = diff_.mutable_gpu_data();
  caffe_gpu_logistic_activate(count,input_data ,diff );

  
  //caffe_gpu_set(diff_.count(), Dtype(0.0), diff);
  Dtype loss(0.0);
  const Dtype alpha = object_scale_;
  //caffe_gpu_mul(count,label,diff);
  caffe_gpu_axpby(bottom[0]->count(),-alpha,label_data,alpha,diff);
  
  diff = diff_.mutable_cpu_data();
  Dtype obj(0.0),no_obj(0.0);
  
  for (int i = 0; i < diff_.count(); ++i) {
    //if(diff[i]>0) 
    //  no_obj += diff[i];
    //else
    //  obj += 1.0 - diff[i];
    loss += diff[i] * diff[i];
  }
  if(iter_%16==0) {
    //LOG(INFO)  << "avg_no_obj : " << no_obj_score_/16 << " , avg_obj : " << obj_score_/16;
    //obj_score_ = 0;
    //no_obj_score_ = 0;
  }
  else {
    //obj_score_ += obj/count;
    //no_obj_score_ += no_obj/count;
  }
  iter_++;
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
  //visualization(bottom,top);	
	//Forward_cpu(bottom,top);	
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
