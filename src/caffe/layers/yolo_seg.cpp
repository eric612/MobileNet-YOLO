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
  CHECK_EQ(bottom[0]->count(), bottom[0]->count()) <<
      "YoloSeg layer inputs must have the same count.";
  YoloSegParameter param = this->layer_param_.yolo_seg_param();
  use_logic_gradient_ = param.use_logic_gradient();
  use_hardsigmoid_ = param.use_hardsigmoid();
  object_scale_ = param.object_scale();
  class_scale_ = param.class_scale();
  num_class_ = param.num_class();
}
  
template <typename Dtype>
void YoloSegLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[0]->count()) <<
      "YoloSeg layer inputs must have the same count.";
  diff_.ReshapeLike(*bottom[0]);
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
  for (int y = 0; y < h; y++) {
    uchar* ptr2 = img2.ptr<uchar>(y);
    int img_index2 = 0;
    for (int j = 0; j < w; j++)
    {
      ptr2[img_index2] = (unsigned char)(sigmoid(bottom_data[img_index1]) * 255);

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
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const Dtype* label_data = bottom[1]->cpu_data(); //[label,x,y,w,h]
  if (diff_.width() != bottom[0]->width()) {
    diff_.ReshapeLike(*bottom[0]);
  }
  Dtype* diff = diff_.mutable_cpu_data();
  caffe_set(diff_.count(), Dtype(0.0), diff);
  Dtype loss(0.0);
  //LOG(INFO) << object_scale_;
  for (int i = 0; i < count; ++i) {
    diff[i] = (-1.0) * (label_data[i]- sigmoid(bottom_data[i]))*object_scale_;
	//LOG(INFO) << sigmoid(bottom_data[i]);
  }
  

  for (int i = 0; i < diff_.count(); ++i) {
    loss += diff[i] * diff[i];
  }
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
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
