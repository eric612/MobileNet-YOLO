/*
* @Author: Eric612
* @Date:   2019-03-11
* @https://github.com/eric612/Caffe-YOLOv3-Windows
* @https://github.com/eric612/MobileNet-YOLO
* Avisonic , ELAN microelectronics
*/

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/layers/segmentation_evaluate_layer.hpp"
#include "caffe/util/bbox_util.hpp"
#define BOUND(a,min_val,max_val)           ( (a < min_val) ? min_val : (a >= max_val) ? (max_val) : a )
namespace caffe {

template <typename Dtype>
void SegmentationEvaluateLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const SegmentationEvaluateParameter& segmentation_evaluate_param =
      this->layer_param_.segmentation_evaluate_param();
  CHECK(segmentation_evaluate_param.has_num_classes())
      << "Must provide num_classes.";
  num_classes_ = segmentation_evaluate_param.num_classes();
  threshold_ = segmentation_evaluate_param.threshold();
  iter_ = 0;
}

template <typename Dtype>
void SegmentationEvaluateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(2, 1);
  top_shape.push_back(1);
  int width = bottom[1]->width();
  int height = bottom[1]->height();
  int size = bottom[1]->channels();
  top_shape.push_back(size);
  top[0]->Reshape(top_shape);
  
}
template <typename Dtype>
void SegmentationEvaluateLayer<Dtype>::visualization(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
  int w = bottom[0]->width();
  int h = bottom[0]->height();
  cv::Mat img2(h, w, CV_8UC1);
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
      ptr2[img_index2] = (unsigned char)((bottom_data[img_index1]) * 255);
      
      //ptr2[img_index2] = (unsigned char)((label_data[img_index1]) * 255);
      img_index1++;
      img_index2++;
    }
  }
  //cv::imwrite("test.jpg",img2);
  cv::namedWindow("show", cv::WINDOW_NORMAL);
  cv::resizeWindow("show", 800, 400);
  cv::imshow("show", img2);
  cv::waitKey(1);
}
template <typename Dtype>
void SegmentationEvaluateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const Dtype* seg_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int size = bottom[0]->channels();
  
  int width = bottom[0]->width();
  int height = bottom[0]->height();
  int img_index1 = 0;
  int eval_width = bottom[1]->width();
  int eval_height = bottom[1]->height();
  float iou = 0;
  //visualization(bottom,top);
  if(width == eval_width && height == eval_height) {

    int count = bottom[0]->count();
    int len = width*height;
    for(int c = 0; c<size;c++) {   
      int gt_pixel_num = 0;
      int match_pixel_num = 0;
      int eval_pixel_num = 0;    
      for (int i = 0; i < height ; i++) {
        for (int j = 0; j < width ; j++) {
          int index = c*len + i*width + j;
          if(gt_data[index]>threshold_) {
            gt_pixel_num++;
            if(seg_data[index]>threshold_) {
              match_pixel_num++;
            }
          }
          if(seg_data[index]>threshold_) {
            eval_pixel_num++;
          }
        }
      }  
      if(match_pixel_num)
        iou = (float) match_pixel_num / (float)(gt_pixel_num + eval_pixel_num - match_pixel_num);
      else
        iou = 0;
      top_data[c] = iou;
      //LOG(INFO) << "class" << c << " : "<<iou;      
    }

  }
  else {
    cv::Mat tmp_img(height, width, CV_8UC1);
    cv::Mat eval_img(eval_height, eval_width, CV_8UC1);
    int len1,len2;
    
    for(int c = 0; c<size;c++) {   
      int img_index1=0;      
      len1 = width*height;
      for (int i = 0; i < height; i++) {
        uchar* ptr2 = tmp_img.ptr<uchar>(i);
        
        int img_index2 = 0;
        for (int j = 0; j < width; j++) {
          ptr2[img_index2] = (unsigned char)BOUND((unsigned char)((seg_data[img_index1+c*len1]) * 255),0,255);
          img_index1++;
          img_index2++;
          //LOG(INFO)<<img_index1;
        }
      }
      
      cv::resize(tmp_img, eval_img, cv::Size(eval_width, eval_height),cv::INTER_AREA);
      int gt_pixel_num = 0;
      int match_pixel_num = 0;
      int eval_pixel_num = 0;  
      int th = threshold_*255;
      len2 = eval_width*eval_height;
      for (int i = 0; i < eval_height; i++) {
        const unsigned char* ptr = eval_img.ptr<unsigned char>(i);
        int img_index = 0;
        for (int j = 0; j < eval_width; j++) {
          int index = c*len2 + i*eval_width + j;
          //LOG(INFO)<<(int)gt_data[index];
          if(gt_data[index]>threshold_) {
            gt_pixel_num++;
            if(ptr[img_index]>th) {
              match_pixel_num++;
            }
          }
          if(ptr[img_index]>th) {
            eval_pixel_num++;
          }
          img_index++;
        }
      }
      if(match_pixel_num) {
        iou = (float) match_pixel_num / (float)(gt_pixel_num + eval_pixel_num - match_pixel_num);
      }
      else if (gt_pixel_num==0) {
        iou = -1;
      }
      else {
        iou = 0;
      }
      top_data[c] = iou;
      //LOG(INFO)  <<"gt_pixel : " << gt_pixel_num<< " , match_pixel : " << match_pixel_num<< " , eval_pixel : " << eval_pixel_num;
    }
  }
  //cv::imwrite("test.jpg",seg_img_[0]);
  //LOG(INFO)<<bottom[0]->num()<<" , "<<bottom[0]->channels()<<" , "<<bottom[0]->width()<<" , "<<bottom[0]->height();
  //LOG(INFO)<<bottom[1]->num()<<" , "<<bottom[1]->channels()<<" , "<<bottom[1]->width()<<" , "<<bottom[1]->height();
  iter_++;
}

INSTANTIATE_CLASS(SegmentationEvaluateLayer);
REGISTER_LAYER_CLASS(SegmentationEvaluate);

}  // namespace caffe
