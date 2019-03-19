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
  const Dtype* class_weighting_data = bottom[2]->cpu_data(); 
  int classes_num = bottom[2]->channels();
  vector<Dtype> scale_data ;
  Dtype total(0.0);
  scale_data.clear();
  for (int i = 0; i < classes_num; i++) {
    Dtype scale_val = 0;   
    for (int j = 0; j < bottom[2]->num(); j++) {
      //LOG(INFO) << class_weighting_data[i];
      scale_val += class_weighting_data[j*classes_num + i];     
    }
    total += scale_val;
    scale_data.push_back(scale_val);
  }
  for (int i = 0; i < classes_num; i++) {
    scale_data[i] /= total;   
    //LOG(INFO) << "classes : "<< i << " , " <<scale_data[i];
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
  int image_size = bottom[0]->width()*bottom[0]->height();
  //caffe_gpu_mul(count,label,diff);
  for (int i = 0; i < classes_num; i++) {
    const Dtype beta = alpha * (2. - scale_data[i]);
    for (int j = 0; j < bottom[0]->num(); j++) {
      int offset = bottom[0]->offset(j) + i*image_size;
      //LOG(INFO) << offset <<"," << bottom[0]->width() << "," << bottom[0]->height();
      caffe_gpu_axpby(image_size,-beta,label_data+offset,beta,diff+offset);
    }
  }
  //caffe_gpu_axpby(bottom[0]->count(),-alpha,label_data,alpha,diff);
  
  diff = diff_.mutable_cpu_data();
  Dtype obj(0.0),no_obj(0.0);
  label_data = bottom[1]->cpu_data(); 
  const Dtype* swap_cpu = swap_.cpu_data();
  float mIOU = 0.0;
  int pos_count_total = 0;
  int neg_count_total = 0;
  int nii_count_total = 0;
  int eval_count_total = 0;
  for (int i = 0; i < classes_num; i++) {
    int pos_count = 0;
    int neg_count = 0;
    int nii_count = 0;
    int eval_count = 0;
    scale_data[i] = 0;
    float IOU_class = 0.0;
    for (int j = 0; j < bottom[0]->num(); j++) {
      int offset = bottom[0]->offset(j) + i*image_size;
      for (int k = 0; k < image_size; k++) {
        int index = offset + k;
        if(label_data[index]>0.5) {
          obj += swap_cpu[index];
          pos_count++;
          if(swap_cpu[index]>0.5) {
            nii_count++;
          }
        }
        else {
          no_obj += swap_cpu[index];
          neg_count++;
        }
        if(swap_cpu[index]>0.5) {
            eval_count++;
        }
        loss += diff[index] * diff[index];
      }
    }
    pos_count_total += pos_count;
    neg_count_total += neg_count;
    nii_count_total += nii_count;
    eval_count_total += eval_count;
    if(nii_count)
      IOU_class = (float)nii_count/(float)(pos_count+eval_count-nii_count);   
    mIOU += IOU_class;
  }
  mIOU /= (float)classes_num;
  //LOG(INFO)  <<mIOU;
  /*for (int i = 0; i < diff_.count(); ++i) {
    if(label_data[i]>0.5) {
      obj += swap_cpu[i];
      pos_count++;
      if(swap_cpu[i]>0.5) {
        nii_count++;
      }
    }
    else {
      no_obj += swap_cpu[i];
      neg_count++;
    }
    if(swap_cpu[i]>0.5) {
        eval_count++;
    }
    loss += diff[i] * diff[i];
  }*/
  if(iter_%16==0) {
    LOG(INFO)  << "no_obj : " << no_obj_score_/16 << " , obj : " << obj_score_/16 << " , mIOU : "<<IOU_score_/16 ;
    obj_score_ = 0;
    no_obj_score_ = 0;
    IOU_score_ = 0;
  }
  else {
    if(pos_count_total)
      obj_score_ += obj/pos_count_total;
    if(neg_count_total)
      no_obj_score_ += no_obj/neg_count_total;
    if(nii_count_total)
      IOU_score_ += mIOU;   
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
