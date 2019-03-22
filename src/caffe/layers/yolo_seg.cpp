/*
* @Author: Eric612
* @Date:   2019-01-29
* @https://github.com/eric612/Caffe-YOLOv2-Windows
* @https://github.com/eric612/MobileNet-YOLO
* Avisonic , ELAN microelectronics
*/

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#endif  // USE_OPENCV

#include <cmath>
#include <vector>
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/yolo_seg_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/region_loss_layer.hpp"
#include <iostream>
#include <algorithm> 
namespace caffe {

template <typename Dtype>
void YoloSegLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "YoloSeg layer inputs must have the same count.";
  YoloSegParameter param = this->layer_param_.yolo_seg_param();
  use_logic_gradient_ = param.use_logic_gradient();
  use_hardsigmoid_ = param.use_hardsigmoid();
  object_scale_ = param.object_scale();
  class_scale_ = param.class_scale();
  num_class_ = param.num_class();
  enable_weighting_ = false;
  if(bottom.size()==3) {
    enable_weighting_ = true;
  }
  iter_ = 0;
}
  
template <typename Dtype>
void YoloSegLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[0]->count()) <<
      "YoloSeg layer inputs must have the same count.";
  diff_.ReshapeLike(*bottom[0]);
  swap_.ReshapeLike(*bottom[0]);
}
template <typename Dtype>
void YoloSegLayer<Dtype>::visualization(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
  int w = bottom[0]->width();
  int h = bottom[0]->height();
  cv::Mat img2(w, h, CV_8UC1);
  uchar* ptr2;
  int img_index1 = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data(); 
  for (int y = 0; y < h; y++) {
    uchar* ptr2 = img2.ptr<uchar>(y);
    int img_index2 = 0;
    for (int j = 0; j < w; j++)
    {
      //LOG(INFO)<<(int)(bottom_data[img_index1] * 255);
      ptr2[img_index2] = (unsigned char)(sigmoid(bottom_data[img_index1]) * 255);
      
      //ptr2[img_index2] = (unsigned char)((label_data[img_index1]) * 255);
      img_index1++;
      img_index2++;
    }
  }
  //cv::imwrite("test.jpg",img2);
  cv::namedWindow("show", cv::WINDOW_NORMAL);
  cv::resizeWindow("show", 600, 600);
  cv::imshow("show", img2);
  cv::waitKey(1);
}
template <typename Dtype>
void YoloSegLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //LOG(INFO)<<bottom[1]->channels()<<","<<bottom[1]->num()<<","<<bottom[1]->width()<<","<<bottom[1]->height();

  Dtype* diff;
  
#ifdef CPU_ONLY
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const Dtype* label_data = bottom[1]->cpu_data(); //[label,x,y,w,h]
  if (diff_.width() != bottom[0]->width()) {
    diff_.ReshapeLike(*bottom[0]);
    swap_.ReshapeLike(*bottom[0]);
  }
  diff = diff_.mutable_cpu_data();
  //caffe_set(diff_.count(), Dtype(0.0), diff);
  Dtype loss(0.0);
  for (int i = 0; i < count; ++i) {
    swap[i] = sigmoid(bottom_data[i]);
  }
  caffe_copy(count,swap,diff);
#endif  
  diff = diff_.mutable_cpu_data();
  //caffe_gpu_set(diff_.count(), Dtype(0.0), diff);
  Dtype loss(0.0);
  const Dtype alpha = object_scale_;
  int classes_num = bottom[1]->channels();
  const Dtype* class_weighting_data; 
  vector<Dtype> scale_data ;
  const Dtype* label_data ; 
  scale_data.clear();
  int image_size = bottom[0]->width()*bottom[0]->height();
  if(enable_weighting_){
    class_weighting_data = bottom[2]->cpu_data(); 
    for (int i = 0; i < classes_num; i++) {
      Dtype scale_val = 0;   
      Dtype total(0.0);
      for (int j = 0; j < bottom[2]->num(); j++) {
        //LOG(INFO) << class_weighting_data[i];
        scale_val += class_weighting_data[j*classes_num + i]; 
        total += image_size - class_weighting_data[j*classes_num + i]; 
        //LOG(INFO) << scale_val << "," << image_size - class_weighting_data[j*classes_num + i];
      }
      
      scale_data.push_back(scale_val/total);
    }
  
  }
  if(enable_weighting_){
    diff = diff_.mutable_cpu_data();
    label_data = bottom[1]->cpu_data(); 
    
    for (int i = 0; i < classes_num; i++) {
      
      const Dtype beta = alpha * (2.0 - 1.*scale_data[i]);
      //LOG(INFO)<<beta;
      for (int j = 0; j < bottom[0]->num(); j++) {
        int offset = bottom[0]->offset(j) + i*image_size;
        //LOG(INFO) << offset <<"," << bottom[0]->width() << "," << bottom[0]->height();
        //caffe_gpu_axpby(image_size,-beta,label_data+offset,beta,diff+offset);
        for (int k = 0; k < image_size; k++) {
          int idx = offset + k;
          if(label_data[idx]>0.5) {
            diff[idx] = (-1.0) * (label_data[idx]- (diff[idx]))*beta;
          }
          else {
            diff[idx] = (-1.0) * (label_data[idx]- (diff[idx]))*object_scale_;
          }
        }
      }
    }
  }
  else {
    
    caffe_cpu_axpby(bottom[0]->count(),-alpha,label_data,alpha,diff);
  }
  
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
  visualization(bottom,top);	
}

template <typename Dtype>
void YoloSegLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
			const Dtype sign(1.);
			const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
			//const Dtype alpha(1.0);
			//LOG(INFO) << "alpha:" << alpha;

			caffe_cpu_axpby(
				bottom[0]->count(),
				alpha,
				diff_.cpu_data(),
				Dtype(0),
				bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(YoloSegLayer);
#endif

INSTANTIATE_CLASS(YoloSegLayer);
REGISTER_LAYER_CLASS(YoloSeg);

}  // namespace caffe
