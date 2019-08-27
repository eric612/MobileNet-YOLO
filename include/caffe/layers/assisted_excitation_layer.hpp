/*
* Reference : https://pdfs.semanticscholar.org/ec96/b6ae95e1ebbe4f7c0252301ede26dfc79467.pdf
* @Author: Eric612
* @Date:   2019-08-22
* @https://github.com/eric612/Caffe-YOLOv2-Windows
* @https://github.com/eric612/MobileNet-YOLO
* Avisonic , ELAN microelectronics
*/

#ifndef CAFFE_Assisted_Excitation_LAYER_HPP_
#define CAFFE_Assisted_Excitation_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class AssistedExcitationLayer : public Layer<Dtype> {
 public:
  explicit AssistedExcitationLayer(const LayerParameter& param)
            :Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "AssistedExcitationLayer"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
 protected:
 
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void visualization(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

 protected:
  //Blob<Dtype> diff_;  
  Blob<Dtype> swap_;  
  Blob<Dtype> mask_;  

};

}  // namespace caffe

#endif  // CAFFE_Assisted_Excitation_LAYER_HPP_
