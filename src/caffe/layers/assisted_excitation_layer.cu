

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
#include "caffe/layers/assisted_excitation_layer.hpp"
#include "caffe/layers/region_loss_layer.hpp"
#include <iostream>
#include <algorithm> 
#define BOUND(a,min_val,max_val)           ( (a < min_val) ? min_val : (a >= max_val) ? (max_val) : a )
#define M_PI 3.14159265358979323846/* pi */
namespace caffe {


template <typename Dtype>
void AssistedExcitationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  caffe_copy(count, bottom_data, top_data);
  const Dtype* label_data = bottom[1]->cpu_data(); //[label,x,y,w,h]
  Dtype* swap_data = swap_.mutable_cpu_data();
  
  int width = bottom[0]->width();
  int height = bottom[0]->height();
  const Dtype* mean_data = bottom[2]->cpu_data();
  //cv::Mat img(width, height, CV_8UC1);
  //img = cv::Scalar(0);

  float alpha =  cos ((this->iter_ / (float) this->max_iter_)*M_PI/2.0f);
  //LOG(INFO) << alpha ;
  for (int b = 0; b < bottom[0]->num(); b++) {
	Dtype* mask_data = mask_.mutable_cpu_data();
    caffe_set(bottom[2]->count(), Dtype(0), mask_data);
    for (int t = 0; t < 300; ++t) {
      Dtype x = label_data[b * 300 * 5 + t * 5 + 1];
      Dtype y = label_data[b * 300 * 5 + t * 5 + 2];
      Dtype w = label_data[b * 300 * 5 + t * 5 + 3];
      Dtype h = label_data[b * 300 * 5 + t * 5 + 4];
      if (!x)
        break;
      
      int lb_x = BOUND((int) ((x - w/2) * width),0,width-1);
      int lb_y = BOUND((int) ((y - h/2) * height),0,height-1);
      int rt_x = BOUND((int) ((x + w/2) * width + 0.95),lb_x+1,width);
      int rt_y = BOUND((int) ((y + h/2) * height + 0.95),lb_y+1,height);
      //LOG(INFO) << lb_x << "," << lb_y<< ","<< rt_x << "," << rt_y;
      
      for (int i = lb_y;i < rt_y; i++) {
        //uchar* ptr2 = img.ptr<uchar>(i);
        //int img_index2 = 0;
        for (int j = lb_x; j < rt_x; j++)
        {
          int index = i * width + j + b*bottom[2]->count(1);
          mask_data[index] = alpha*mean_data[index];
          //mask_data[index] = alpha;
          //LOG(INFO) << index;
          //caffe_set(count, alpha*mean_data[index], swap_data);
          //ptr2[j] = (unsigned char) BOUND(255,0,255);
        }
      }
    }
    mask_data = mask_.mutable_gpu_data();
    for (int c=0;c<bottom[0]->channels();c++) {
      int offset = bottom[0]->offset(b) + c*bottom[2]->count(1);
      caffe_gpu_axpy(width*height, Dtype(1) , mask_data, &top_data[offset]);
    }
    /*if(b==0) {
      cv::namedWindow("show", cv::WINDOW_NORMAL);
      cv::resizeWindow("show", 600, 600);
      cv::imshow("show", img);
      cv::waitKey(1);
    }*/
  }
}

template <typename Dtype>
void AssistedExcitationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int count = top[0]->count();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AssistedExcitationLayer);

}  // namespace caffe
