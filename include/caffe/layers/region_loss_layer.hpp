#ifndef CAFFE_REGION_LOSS_LAYER_HPP_
#define CAFFE_REGION_LOSS_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <string>
#include "caffe/layers/loss_layer.hpp"
#include <map>

namespace caffe {
template <typename Dtype>
Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2);

template <typename Dtype>
Dtype Calc_iou(const vector<Dtype>& box, const vector<Dtype>& truth);

template <typename Dtype>
void disp(Blob<Dtype>& swap);

template <typename Dtype>
inline Dtype sigmoid(Dtype x)
{
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
Dtype softmax_region(Dtype* input, int classes);
//template <typename Dtype>
//Dtype softmax_region(Dtype* input, int n, float temp, Dtype* output);



template <typename Dtype>
vector<Dtype> get_region_box(Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int w, int h);

template <typename Dtype>
Dtype delta_region_box(vector<Dtype> truth, Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int w, int h, Dtype* delta, float scale);

template <typename Dtype>
void delta_region_class(Dtype* input_data, Dtype* &diff, int index, int class_label, int classes, float scale, Dtype* avg_cat);

template <typename Dtype>
class PredictionResult {
public:
	Dtype x;
	Dtype y;
	Dtype w;
	Dtype h;
	Dtype objScore;
	Dtype classScore;
	Dtype confidence;
	int classType;
};

template <typename Dtype>
class RegionLossLayer : public LossLayer<Dtype> {
 public:
  explicit RegionLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RegionLoss"; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  int side_;
  int bias_match_;
  int num_class_;
  int coords_;
  int num_;
  int softmax_;
  float jitter_;
  int rescore_;
  
  float object_scale_;
  float class_scale_;
  float noobject_scale_;
  float coord_scale_;
  
  int absolute_;
  float thresh_;
  int random_;
  vector<Dtype> biases_;

  Blob<Dtype> diff_;
  Blob<Dtype> real_diff_;

  string class_map_;
  map<int, int> cls_map_;
};

}  // namespace caffe

#endif  // CAFFE_REGION_LOSS_LAYER_HPP_
