/*
* @Author: Eric612
* @Date:   2019-03-11
* @https://github.com/eric612/Caffe-YOLOv3-Windows
* @https://github.com/eric612/MobileNet-YOLO
* Avisonic , ELAN microelectronics
*/
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#ifndef CAFFE_SEGMENTATION_EVALUATE_LAYER_HPP_
#define CAFFE_SEGMENTATION_EVALUATE_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Generate the detection evaluation based on DetectionOutputLayer and
 * ground truth bounding box labels.
 *
 * Intended for use with MultiBox detection method.
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class SegmentationEvaluateLayer : public Layer<Dtype> {
 public:
  explicit SegmentationEvaluateLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetectionEvaluate"; }
  virtual inline int ExactBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @brief Evaluate the detection output.
   *
   * @param bottom input Blob vector (exact 2)
   *   -# @f$ (1 \times 1 \times N \times 7) @f$
   *      N detection results.
   *   -# @f$ (1 \times 1 \times M \times 7) @f$
   *      M ground truth.
   * @param top Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times N \times 4) @f$
   *      N is the number of detections, and each row is:
   *      [image_id, label, confidence, true_pos, false_pos]
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  void visualization(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  int num_classes_;
  //std::vector<cv::Mat> seg_img_;
  float threshold_;
  int iter_;
};

}  // namespace caffe

#endif  // CAFFE_SEGMENTATION_EVALUATE_LAYER_HPP_
