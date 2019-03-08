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
  //LOG(INFO)<<bottom[1]->width()<<","<<bottom[1]->height()<<","<<bottom[1]->num()<<","<<bottom[1]->channels();
  if (diff_.width() != bottom[0]->width()) {
    diff_.ReshapeLike(*bottom[0]);
    swap_.ReshapeLike(*bottom[0]);
  }
  const int count = bottom[0]->count();
  const Dtype* input_data = bottom[0]->gpu_data();	
  const Dtype* label_data = bottom[1]->gpu_data(); 
  Dtype* diff = diff_.mutable_gpu_data();
  Dtype* swap = swap_.mutable_gpu_data();
  caffe_gpu_logistic_activate(count,input_data ,swap );
  caffe_copy(count,swap,diff);
  
  //caffe_gpu_set(diff_.count(), Dtype(0.0), diff);
  Dtype loss(0.0);
  const Dtype alpha = object_scale_;
  //caffe_gpu_mul(count,label,diff);
  caffe_gpu_axpby(bottom[0]->count(),-alpha,label_data,alpha,diff);
  
  diff = diff_.mutable_cpu_data();
  Dtype obj(0.0),no_obj(0.0);
  label_data = bottom[1]->cpu_data(); 
  const Dtype* swap_cpu = swap_.cpu_data();
  int pos_count = 0;
  int neg_count = 0;
  int nii_count = 0;
  int eval_count = 0;
  for (int i = 0; i < diff_.count(); ++i) {
    if(label_data[i]>0.1) {
      obj += swap_cpu[i];
      pos_count++;
      if(swap_cpu[i]>0.1) {
        nii_count++;
      }
    }
    else {
      no_obj += swap_cpu[i];
      neg_count++;
    }
    if(swap_cpu[i]>0.1) {
        eval_count++;
    }
    loss += diff[i] * diff[i];
  }
  if(iter_%16==0) {
    LOG(INFO)  << "no_obj : " << no_obj_score_/16 << " , obj : " << obj_score_/16 << " , IOU : "<<IOU_score_/16 << " , gt_pixel : " << pos_count<< " , match_pixel : " << nii_count<< " , eval_pixel : " << eval_count;
    obj_score_ = 0;
    no_obj_score_ = 0;
    IOU_score_ = 0;
  }
  else {
    if(pos_count)
      obj_score_ += obj/pos_count;
    if(neg_count)
      no_obj_score_ += no_obj/neg_count;
   if(nii_count)
      IOU_score_ += (float)nii_count/(float)(pos_count+eval_count-nii_count);   
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
