#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/lane_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"
#include "caffe/util/im_transforms.hpp"
#include "caffe/util/bbox_util.hpp"
//#define draw 
const float prob_eps = 0.01;
namespace caffe {

template <typename Dtype>
LaneDataLayer<Dtype>::LaneDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
LaneDataLayer<Dtype>::~LaneDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void LaneDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
    this->layer_param_.annotated_data_param();
  yolo_data_jitter_ = anno_data_param.yolo_data_jitter();
  yolo_data_type_ = anno_data_param.yolo_data_type();
  seg_scales_ = anno_data_param.seg_scales();
  seg_resize_width_ = anno_data_param.seg_resize_width();
  seg_resize_height_ = anno_data_param.seg_resize_height();
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }

  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  if (transform_param.resize_param_size()) {
    if (transform_param.resize_param(0).resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }
  iters_ = 0;
  policy_num_ = 0;
  // Read a data point, and use it to initialize the top blob.
  AnnotatedDatum& anno_datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum(),0);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    vector<int> label_shape(4, 1);
    label_shape[0] = batch_size;
    label_shape[1] = 1;
    label_shape[2] = 1;
    label_shape[3] = 1;
    top[1]->Reshape(label_shape);

    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
    
  }
  if (this->output_seg_labels_) {
    
	  vector<int> seg_label_shape(4, 1);
	  seg_label_shape[0] = batch_size;
	  seg_label_shape[1] = 1;
    if(seg_resize_width_==0 || seg_resize_height_==0) {
      seg_label_shape[2] = top_shape[2] / seg_scales_;
      seg_label_shape[3] = top_shape[3] / seg_scales_;
    }      
    else {
      seg_label_shape[2] = seg_resize_width_;
      seg_label_shape[3] = seg_resize_height_;  
    }
    LOG(INFO)<<seg_label_shape[0]<<","<<seg_label_shape[1]<<","<<seg_label_shape[2]<<","<<seg_label_shape[3];
	  top[2]->Reshape(seg_label_shape);
	  for (int i = 0; i < this->prefetch_.size(); ++i) {
		  this->prefetch_[i]->seg_label_.Reshape(seg_label_shape);
	  }
	  this->transformed_label_.Reshape(seg_label_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}


// This function is called on prefetch thread
template<typename Dtype>
void LaneDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  AnnotatedDatum& anno_datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from anno_datum.
  int num_resize_policies = transform_param.resize_param_size();
  bool size_change = false;
  

  if (num_resize_policies > 0 && iters_%10 == 0) {
	  std::vector<float> probabilities;
	  float prob_sum = 0;
	  for (int i = 0; i < num_resize_policies; i++) {
		  const float prob = transform_param.resize_param(i).prob();
		  CHECK_GE(prob, 0);
		  CHECK_LE(prob, 1);
		  prob_sum += prob;
		  probabilities.push_back(prob);
	  }
	  CHECK_NEAR(prob_sum, 1.0, prob_eps);
	  policy_num_ = roll_weighted_die(probabilities);
	  size_change = true;
  }
  else {

  }
  
  vector<int> top_shape =
	  this->data_transformer_->InferBlobShape(anno_datum.datum(), policy_num_);
  // Reshape batch according to the batch_size.
  this->transformed_data_.Reshape(top_shape);
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);
  int num_bboxes = 0;
  NormalizedBBox crop_box;
  //LOG(INFO) << this->prefetch_[i].data_.width() << "," << transform_param.resize_param(policy_num_).width() << "," << iters_;
  //this->prefetch_[0].data_.Reshape(top_shape);
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  Dtype* top_seg_label = NULL;  // suppress warnings about uninitialized variables
  vector<int> label_shape(4, 1);
  if (this->output_labels_) {   
    label_shape[0] = batch_size;
    label_shape[1] = 5; // maxima lane 
    label_shape[2] = 64; // maxima node
    label_shape[3] = 3; // coor. x and y
    batch->label_.Reshape(label_shape);
    top_label = batch->label_.mutable_cpu_data();
  }
  if (this->output_seg_labels_ ) {
	  top_seg_label = batch->seg_label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a anno_datum
    AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    AnnotatedDatum distort_datum;
    AnnotatedDatum* expand_datum = NULL;
    
    if (transform_param.has_distort_param()) {
      distort_datum.CopyFrom(anno_datum);
      this->data_transformer_->DistortImage(anno_datum.datum(),
                                            distort_datum.mutable_datum());
      expand_datum = &distort_datum;
    } else {
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(anno_datum, expand_datum);
      } else {
        expand_datum = &anno_datum;
      }
    }
    AnnotatedDatum* sampled_datum = NULL;
    
    bool has_sampled = false;
    if (batch_samplers_.size() > 0 || yolo_data_type_== 1) {
      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      
      if (batch_samplers_.size()>0) {
        GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
      }
      else {
        bool keep = transform_param.resize_param(policy_num_).resize_mode() == ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD;
        GenerateJitterSamples(yolo_data_jitter_, &sampled_bboxes , keep);
      }
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        sampled_datum = new AnnotatedDatum();
        crop_box = sampled_bboxes[rand_idx];
        this->data_transformer_->CropImage(*expand_datum,
                                           sampled_bboxes[rand_idx],
                                           sampled_datum,false);
        has_sampled = true;
      } else {
        sampled_datum = expand_datum;
      }
    } else {
      sampled_datum = expand_datum;
    }
    CHECK(sampled_datum != NULL);

    vector<int> shape =
        this->data_transformer_->InferBlobShape(sampled_datum->datum(), policy_num_);

    //LOG(INFO) << shape[2] << "," << shape[3];
    if (transform_param.resize_param_size()) {
      if (transform_param.resize_param(policy_num_).resize_mode() ==
          ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
        this->transformed_data_.Reshape(shape);
        batch->data_.Reshape(shape);
        top_data = batch->data_.mutable_cpu_data();
      } else {
		  //LOG(INFO) << top_shape;
        //CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
        //      shape.begin() + 1));
      }
    } else {
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
            shape.begin() + 1));
    }
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    vector<AnnotationGroup> transformed_anno_vec;

    this->data_transformer_->Transform(sampled_datum->datum(),
                                       &(this->transformed_data_),policy_num_); // entry point 1
    
    
    //LOG(INFO)<< "packing " << anno_datum.annotation_group().size() << " lanes annotation into prefetch buffer";
    if (this->output_labels_) {

      int total = anno_datum.annotation_group().size();
#ifdef draw
      cv::Mat cv_img = DecodeDatumToCVMatNative(sampled_datum->datum());
      if (this->data_transformer_->get_mirror()) {
        cv::flip(cv_img, cv_img, 1);
      }
#endif
      //LOG(INFO) << crop_box.xmin() << crop_box.xmax() << crop_box.ymin() << crop_box.ymax();
      float crop_w = crop_box.xmax() - crop_box.xmin();
      float crop_h = crop_box.ymax() - crop_box.ymin();
      cv::Mat cv_lab;
      vector<int> seg_label_shape(4);
      if (this->output_seg_labels_) {
        
        seg_label_shape[0] = batch_size;
        seg_label_shape[1] = 1;
        
        if(seg_resize_width_==0 || seg_resize_height_==0) {
          seg_label_shape[2] = top_shape[2] / seg_scales_;
          seg_label_shape[3] = top_shape[3] / seg_scales_;
        }
        else {
          seg_label_shape[2] = seg_resize_height_;
          seg_label_shape[3] = seg_resize_width_;        
        }
        cv_lab = cv::Mat(seg_label_shape[2],seg_label_shape[3], CV_8UC1, cvScalar(0));
        batch->seg_label_.Reshape(seg_label_shape);
        
      }
      for (int i=0;i<total;i++) {
        const AnnotationGroup &anno_group = anno_datum.annotation_group(i);        
        int point_size = anno_group.annotation().size();
        float px , py;
        int count = 0;
        int index = i*label_shape[2]*label_shape[3];
        for(int j=0;j<point_size;j++) {
          const Annotation &anno_lane = anno_group.annotation(point_size-1-j);
          float x = anno_lane.lanes().x();
          float y = anno_lane.lanes().y();
          x = (x - crop_box.xmin())/crop_w;
          y = (y - crop_box.ymin())/crop_h;
          if (this->data_transformer_->get_mirror()) {
            //x = 1.0 - x;
          }
          int label_offset = batch->label_.offset(item_id);
          int idx = label_offset + index + j*label_shape[3];
          top_label[idx] = x;
          top_label[idx+1] = y;
          top_label[idx+2] = i;
          if( !(x>=0 && x<=1 && y>=0 && y<=1)) {
            top_label[idx+2] = -1; // out of crop box
          }
          //LOG(INFO)<<idx;

          if(count>0 && x>=0 && x<=1 && y>=0 && y<=1) {
            

            int ix = x*cv_lab.cols;
            int iy = y*cv_lab.rows;
            int ipx = px*cv_lab.cols;
            int ipy = py*cv_lab.rows;
            cv::line(cv_lab,cv::Point(ix,iy),cv::Point(ipx,ipy),cv::Scalar(255,255,255),1); //test draw           
#ifdef draw
            cv::line(cv_img,cv::Point(ix,iy),cv::Point(ipx,ipy),cv::Scalar(128,255,128),2); //test draw
            cv::circle(cv_img,cv::Point(ix,iy),3,cv::Scalar(128,128,255),4);
#endif            
          }

          if(x>=0 && x<=1 && y>=0 && y<=1) {
            count++;
            px = x;
            py = y;
          }
        }
      }
      std::vector<cv::Mat> channels;

      this->transformed_label_.Reshape(seg_label_shape);
      int offset = batch->seg_label_.offset(item_id);
      this->transformed_label_.set_cpu_data(top_seg_label + offset);
      channels.push_back(cv_lab);
      this->data_transformer_->Transform2(channels, &this->transformed_label_, true);
      //char buf[1000];
      //sprintf(buf, "input/input_%05d.jpg",iters_*batch_size+item_id);
      //cv::imwrite(buf,cv_lab);
#ifdef draw
      char buf[1000];
      sprintf(buf, "input/input_%05d.jpg",iters_*batch_size+item_id);
      cv::imwrite(buf,cv_img);
#endif
    }

    // clear memory
    if (has_sampled) {
      delete sampled_datum;
    }
    if (transform_param.has_expand_param()) {
      delete expand_datum;
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));


  }

  iters_++;
}

INSTANTIATE_CLASS(LaneDataLayer);
REGISTER_LAYER_CLASS(LaneData);

}  // namespace caffe
