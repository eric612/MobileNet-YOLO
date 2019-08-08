/*
* @Author: Eric612
* @Date:   2018-08-20 
* @https://github.com/eric612/Caffe-YOLOv2-Windows
* @https://github.com/eric612/MobileNet-YOLO
* Avisonic
*/
#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/layers/yolov3_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

#include <algorithm>
#include <cfloat>
#include <vector>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV


namespace caffe {
  

  template <typename Dtype>
  dxrep dx_box_iou(vector<Dtype> pred, vector<Dtype> truth, IOU_LOSS iou_loss) {
    boxabs pred_tblr = to_tblr(pred);
    float pred_t = fmin(pred_tblr.top, pred_tblr.bot);
    float pred_b = fmax(pred_tblr.top, pred_tblr.bot);
    float pred_l = fmin(pred_tblr.left, pred_tblr.right);
    float pred_r = fmax(pred_tblr.left, pred_tblr.right);

    boxabs truth_tblr = to_tblr(truth);
  #ifdef DEBUG_PRINTS
    printf("\niou: %f, giou: %f\n", box_iou(pred, truth), box_giou(pred, truth));
    printf("pred: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n", pred.x, pred.y, pred.w, pred.h, pred_tblr.top, pred_tblr.bot, pred_tblr.left, pred_tblr.right);
    printf("truth: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n", truth.x, truth.y, truth.w, truth.h, truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
  #endif
    //printf("pred (t,b,l,r): (%f, %f, %f, %f)\n", pred_t, pred_b, pred_l, pred_r);
    //printf("trut (t,b,l,r): (%f, %f, %f, %f)\n", truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
    dxrep dx = { 0 };
    float X = (pred_b - pred_t) * (pred_r - pred_l);
    float Xhat = (truth_tblr.bot - truth_tblr.top) * (truth_tblr.right - truth_tblr.left);
    float Ih = fmin(pred_b, truth_tblr.bot) - fmax(pred_t, truth_tblr.top);
    float Iw = fmin(pred_r, truth_tblr.right) - fmax(pred_l, truth_tblr.left);
    float I = Iw * Ih;
    float U = X + Xhat - I;

    float Cw = fmax(pred_r, truth_tblr.right) - fmin(pred_l, truth_tblr.left);
    float Ch = fmax(pred_b, truth_tblr.bot) - fmin(pred_t, truth_tblr.top);
    float C = Cw * Ch;

    // float IoU = I / U;
    // Partial Derivatives, derivatives
    float dX_wrt_t = -1 * (pred_r - pred_l);
    float dX_wrt_b = pred_r - pred_l;
    float dX_wrt_l = -1 * (pred_b - pred_t);
    float dX_wrt_r = pred_b - pred_t;

    // gradient of I min/max in IoU calc (prediction)
    float dI_wrt_t = pred_t > truth_tblr.top ? (-1 * Iw) : 0;
    float dI_wrt_b = pred_b < truth_tblr.bot ? Iw : 0;
    float dI_wrt_l = pred_l > truth_tblr.left ? (-1 * Ih) : 0;
    float dI_wrt_r = pred_r < truth_tblr.right ? Ih : 0;
    // derivative of U with regard to x
    float dU_wrt_t = dX_wrt_t - dI_wrt_t;
    float dU_wrt_b = dX_wrt_b - dI_wrt_b;
    float dU_wrt_l = dX_wrt_l - dI_wrt_l;
    float dU_wrt_r = dX_wrt_r - dI_wrt_r;
    // gradient of C min/max in IoU calc (prediction)
    float dC_wrt_t = pred_t < truth_tblr.top ? (-1 * Cw) : 0;
    float dC_wrt_b = pred_b > truth_tblr.bot ? Cw : 0;
    float dC_wrt_l = pred_l < truth_tblr.left ? (-1 * Ch) : 0;
    float dC_wrt_r = pred_r > truth_tblr.right ? Ch : 0;

    // Final IOU loss (prediction) (negative of IOU gradient, we want the negative loss)
    float p_dt = 0;
    float p_db = 0;
    float p_dl = 0;
    float p_dr = 0;
    if (U > 0) {
        p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / (U * U);
        p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / (U * U);
        p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / (U * U);
        p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / (U * U);
    }

    if (iou_loss == GIOU) {
        if (C > 0) {
            // apply "C" term from gIOU
            p_dt += ((C * dU_wrt_t) - (U * dC_wrt_t)) / (C * C);
            p_db += ((C * dU_wrt_b) - (U * dC_wrt_b)) / (C * C);
            p_dl += ((C * dU_wrt_l) - (U * dC_wrt_l)) / (C * C);
            p_dr += ((C * dU_wrt_r) - (U * dC_wrt_r)) / (C * C);
        }
    }

    // apply grad from prediction min/max for correct corner selection
    dx.dt = pred_tblr.top < pred_tblr.bot ? p_dt : p_db;
    dx.db = pred_tblr.top < pred_tblr.bot ? p_db : p_dt;
    dx.dl = pred_tblr.left < pred_tblr.right ? p_dl : p_dr;
    dx.dr = pred_tblr.left < pred_tblr.right ? p_dr : p_dl;

    return dx;
  }

  template <typename Dtype>
  void delta_region_class_v3(Dtype* input_data, Dtype* &diff, int index, int class_label, int classes, float scale, Dtype* avg_cat, int stride, bool use_focal_loss)
  {
    if (diff[index]) {
      diff[index + stride*class_label] = (-1.0) * (1 - input_data[index + stride*class_label]);
      *avg_cat += input_data[index + stride*class_label];
      //LOG(INFO)<<"test";
      return;
    }
    if (use_focal_loss) {
      //Reference : https://github.com/AlexeyAB/darknet/blob/master/src/yolo_layer.c
      float alpha = 0.5;    // 0.25 or 0.5
                  //float gamma = 2;    // hardcoded in many places of the grad-formula

      int ti = index + stride*class_label;
      float pt = input_data[ti] + 0.000000000000001F;
      // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
      float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
                                  //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

      for (int n = 0; n < classes; ++n) {
        diff[index + stride*n] = (-1.0) * scale * (((n == class_label) ? 1 : 0) - input_data[index + n*stride]);

        diff[index + stride*n] *= alpha*grad;

        if (n == class_label) {
          *avg_cat += input_data[index + stride*n];
        }
      }

    }
    else {
      for (int n = 0; n < classes; ++n) {
        diff[index + n*stride] = (-1.0) * scale * (((n == class_label) ? 1 : 0) - input_data[index + n*stride]);
        //std::cout<<diff[index+n]<<",";
        if (n == class_label) {
          *avg_cat += input_data[index + n*stride];
          //std::cout<<"avg_cat:"<<input_data[index+n]<<std::endl; 
        }
      }
    }

  }

  template <typename Dtype>
  Dtype delta_region_box(vector<Dtype> truth, Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int lw, int lh, int w, int h, Dtype* delta, float scale, int stride,IOU_LOSS iou_loss,float iou_normalizer) {
    vector<Dtype> pred;
    pred.clear();
    
    get_region_box(pred, x, biases, n, index, i, j,lw,lh, w, h, stride);

    if (iou_loss == MSE)    // old loss
    {
      float iou = box_iou(pred, truth);
      //LOG(INFO) << pred[0] << "," << pred[1] << "," << pred[2] << "," << pred[3] << ";"<< truth[0] << "," << truth[1] << "," << truth[2] << "," << truth[3];
      float tx = truth[0] * lw - i; //0.5
      float ty = truth[1] * lh - j; //0.5
      float tw = log(truth[2] * w / biases[2 * n]); //truth[2]=biases/w tw = 0
      float th = log(truth[3] * h / biases[2 * n + 1]); //th = 0

      //delta[index + 0] = (-1) * scale * (tx - sigmoid(x[index + 0 * stride])) * sigmoid(x[index + 0 * stride]) * (1 - sigmoid(x[index + 0 * stride]));
      //delta[index + 1 * stride] = (-1) * scale * (ty - sigmoid(x[index + 1 * stride])) * sigmoid(x[index + 1 * stride]) * (1 - sigmoid(x[index + 1 * stride]));
      delta[index + 0 * stride] = (-1) * scale * (tx - x[index + 0 * stride]);
      delta[index + 1 * stride] = (-1) * scale * (ty - x[index + 1 * stride]);
      delta[index + 2 * stride] = (-1) * scale * (tw - x[index + 2 * stride]);
      delta[index + 3 * stride] = (-1) * scale * (th - x[index + 3 * stride]);
      return iou;
    }
    else {
      // Reference code : https://github.com/AlexeyAB/darknet/blob/master/src/yolo_layer.c
      
      // https://github.com/generalized-iou/g-darknet
      // https://arxiv.org/abs/1902.09630v2
      // https://giou.stanford.edu/
      float giou = box_giou(pred, truth);
      ious all_ious = { 0 };
      if (pred[2] == 0) { pred[2] = 1.0; }
      if (pred[3] == 0) { pred[3] = 1.0; }
      // i - step in layer width
      // j - step in layer height
      //  Returns a box in absolute coordinates
      //box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
      all_ious.iou = box_iou(pred, truth);
      all_ious.giou = box_giou(pred, truth);
    
      all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

      // jacobian^t (transpose)
      delta[index + 0 * stride] = (-1) * (all_ious.dx_iou.dl + all_ious.dx_iou.dr);
      delta[index + 1 * stride] = (-1) * (all_ious.dx_iou.dt + all_ious.dx_iou.db);
      delta[index + 2 * stride] = (-1) * ((-0.5 * all_ious.dx_iou.dl) + (0.5 * all_ious.dx_iou.dr));
      delta[index + 3 * stride] = (-1) * ((-0.5 * all_ious.dx_iou.dt) + (0.5 * all_ious.dx_iou.db));

      // predict exponential, apply gradient of e^delta_t ONLY for w,h
      delta[index + 2 * stride] *= exp(x[index + 2 * stride]);
      delta[index + 3 * stride] *= exp(x[index + 3 * stride]);

      // normalize iou weight
      delta[index + 0 * stride] *= iou_normalizer;
      delta[index + 1 * stride] *= iou_normalizer;
      delta[index + 2 * stride] *= iou_normalizer;
      delta[index + 3 * stride] *= iou_normalizer;
      return giou;
    }
    
  }
  template <typename Dtype>
  void Yolov3Layer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    Yolov3Parameter param = this->layer_param_.yolov3_param();
    iter_ = 0;
    num_class_ = param.num_class(); //20
    num_ = param.num(); //5
    side_w_ = bottom[0]->width();
    side_h_ = bottom[0]->height();
    anchors_scale_ = param.anchors_scale();
    object_scale_ = param.object_scale(); //5.0
    noobject_scale_ = param.noobject_scale(); //1.0
    class_scale_ = param.class_scale(); //1.0
    coord_scale_ = param.coord_scale(); //1.0
    thresh_ = param.thresh(); //0.6
    use_logic_gradient_ = param.use_logic_gradient();
    use_focal_loss_  = param.use_focal_loss();
    iou_loss_ = (IOU_LOSS) param.iou_loss();
    
    iou_normalizer_ = param.iou_normalizer();
    for (int c = 0; c < param.biases_size(); ++c) {
      biases_.push_back(param.biases(c));
    } 
    for (int c = 0; c < param.mask_size(); ++c) {
      mask_.push_back(param.mask(c));
    }
    biases_size_ = param.biases_size()/2;
    int input_count = bottom[0]->count(1); //h*w*n*(classes+coords+1) = 13*13*5*(20+4+1)
    int label_count = bottom[1]->count(1); //30*5-
                         // outputs: classes, iou, coordinates
    int tmp_input_count = side_w_ * side_h_ * num_ * (4 + num_class_ + 1); //13*13*5*(20+4+1) label: isobj, class_label, coordinates
    int tmp_label_count = 300 * num_;
    CHECK_EQ(input_count, tmp_input_count);
    //CHECK_EQ(label_count, tmp_label_count);
  }
  typedef struct {
    float x, y, w, h;
  } box;


  template <typename Dtype>
  void Yolov3Layer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    diff_.ReshapeLike(*bottom[0]);
    real_diff_.ReshapeLike(*bottom[0]);
  }
  template <typename Dtype>
  int int_index(vector<Dtype> a, int val, int n)
  {
    int i;
    for (i = 0; i < n; ++i) {
      if (a[i] == val) return i;
    }
    return -1;
  }
  template <typename Dtype>
  void Yolov3Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    side_w_ = bottom[0]->width();
    side_h_ = bottom[0]->height();
    //LOG(INFO)<<"iou loss" << iou_loss_<<","<<GIOU;
    //LOG(INFO) << side_*anchors_scale_;
    const Dtype* label_data = bottom[1]->cpu_data(); //[label,x,y,w,h]
    if (diff_.width() != bottom[0]->width()) {
      diff_.ReshapeLike(*bottom[0]);
    }
    Dtype* diff = diff_.mutable_cpu_data();
    caffe_set(diff_.count(), Dtype(0.0), diff);

    Dtype avg_anyobj(0.0), avg_obj(0.0), avg_iou(0.0), avg_cat(0.0), recall(0.0), recall75(0.0), loss(0.0), avg_iou_loss(0.0);
    int count = 0;
    
    const Dtype* input_data = bottom[0]->cpu_data();
    //const Dtype* label_data = bottom[1]->cpu_data();		
    if (swap_.width() != bottom[0]->width()) {
      swap_.ReshapeLike(*bottom[0]);
    }
    Dtype* swap_data = swap_.mutable_cpu_data();
    int len = 4 + num_class_ + 1;
    int stride = side_w_*side_h_;
    /*for (int i = 0; i < 81; i++) {
      char label[100];
      sprintf(label, "%d,%s\n",i, CLASSES[static_cast<int>(i )]);
      printf(label);
    }*/
    for (int b = 0; b < bottom[0]->num(); b++) {
      /*//if (b == 0) {
        char buf[100];
        int idx = iter_*bottom[0]->num() + b;
        sprintf(buf, "input/input_%05d.jpg", idx+1 );
        //int idx = (iter*swap.num() % 200) + b;
        cv::Mat cv_img = cv::imread(buf);
        for (int t = 0; t < 300; ++t) {
          vector<Dtype> truth;
          Dtype c = label_data[b * 300 * 5 + t * 5 + 0];
          Dtype x = label_data[b * 300 * 5 + t * 5 + 1];

          Dtype y = label_data[b * 300 * 5 + t * 5 + 2];
          Dtype w = label_data[b * 300 * 5 + t * 5 + 3];
          Dtype h = label_data[b * 300 * 5 + t * 5 + 4];
          if (!x) break;
          float left = (x - w / 2.);
          float right = (x + w / 2.);
          float top = (y - h / 2.);
          float bot = (y + h / 2.);

          cv::Point pt1, pt2;
          pt1.x = left*cv_img.cols;
          pt1.y = top*cv_img.rows;
          pt2.x = right*cv_img.cols;
          pt2.y = bot*cv_img.rows;

          cv::rectangle(cv_img, pt1, pt2, cvScalar(0, 255, 0), 1, 8, 0);
          char label[100];
          sprintf(label, "%s", CLASSES[static_cast<int>(c + 1)]);
          int baseline;
          cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 0, &baseline);
          cv::Point pt3;
          pt3.x = pt1.x + size.width;
          pt3.y = pt1.y - size.height;
          cv::rectangle(cv_img, pt1, pt3, cvScalar(0, 255, 0), -1);


          cv::putText(cv_img, label, pt1, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
          //LOG(INFO) << "Truth box" << "," << c << "," << x << "," << y << "," << w << "," << h;
        }
        sprintf(buf, "out/out_%05d.jpg", idx);
        cv::imwrite(buf, cv_img);
      //}*/
      for (int s = 0; s < stride; s++) {
        for (int n = 0; n < num_; n++) {
          int index = n*len*stride + s + b*bottom[0]->count(1);
          //LOG(INFO)<<index;
          vector<Dtype> pred;
          float best_iou = 0;
          int best_class = -1;
          vector<Dtype> best_truth;
#ifdef CPU_ONLY
          for (int c = 0; c < len; ++c) {
            int index2 = c*stride + index;
            //LOG(INFO)<<index2;
            if (c == 2 || c==3) {
              swap_data[index2] = (input_data[index2 + 0]);
            }
            else {						
              swap_data[index2] = logistic_activate(input_data[index2 + 0]);
            }
          }
#endif
          int y2 = s / side_w_;
          int x2 = s % side_w_;
          get_region_box(pred, swap_data, biases_, mask_[n], index, x2, y2, side_w_, side_h_, side_w_*anchors_scale_, side_h_*anchors_scale_, stride);
          for (int t = 0; t < 300; ++t) {
            vector<Dtype> truth;
            Dtype x = label_data[b * 300 * 5 + t * 5 + 1];
            Dtype y = label_data[b * 300 * 5 + t * 5 + 2];
            Dtype w = label_data[b * 300 * 5 + t * 5 + 3];
            Dtype h = label_data[b * 300 * 5 + t * 5 + 4];

            if (!x)
              break;

            truth.push_back(x);
            truth.push_back(y);
            truth.push_back(w);
            truth.push_back(h);
            Dtype iou = box_iou(pred, truth);
            if (iou > best_iou) {
              best_class = label_data[b * 300 * 5 + t * 5];
              best_iou = iou;
              best_truth = truth;
            }
          }
          avg_anyobj += swap_data[index + 4 * stride];
          diff[index + 4 * stride] = (-1) * (0 - swap_data[index + 4 * stride]);
          //diff[index + 4 * stride] = (-1) * (0 - exp(input_data[index + 4 * stride]-exp(input_data[index + 4 * stride])));
          //diff[index + 4 * stride] = (-1) * noobject_scale_ * (0 - swap_data[index + 4 * stride]) *logistic_gradient(swap_data[index + 4 * stride]);
          if (best_iou > thresh_) {
            diff[index + 4 * stride] = 0;
          }
          if (best_iou > 1) {
            LOG(INFO) << "best_iou > 1"; // plz tell me ..
            diff[index + 4 * stride] = (-1) * (1 - swap_data[index + 4 * stride]);

            delta_region_class_v3(swap_data, diff, index + 5 * stride, best_class, num_class_, class_scale_, &avg_cat, stride, use_focal_loss_);
            delta_region_box(best_truth, swap_data, biases_, mask_[n], index, x2, y2, side_w_, side_h_,
              side_w_*anchors_scale_, side_h_*anchors_scale_, diff, coord_scale_*(2 - best_truth[2] * best_truth[3]), stride,iou_loss_,iou_normalizer_);
          }
        }
      }
      //vector<Dtype> used;
      //used.clear();
      for (int t = 0; t < 300; ++t) {
        vector<Dtype> truth;
        truth.clear();
        int class_label = label_data[t * 5 + b * 300 * 5 + 0];
        float x = label_data[t * 5 + b * 300 * 5 + 1];
        float y = label_data[t * 5 + b * 300 * 5 + 2];
        float w = label_data[t * 5 + b * 300 * 5 + 3];
        float h = label_data[t * 5 + b * 300 * 5 + 4];

        if (!w)
          break;
        truth.push_back(x);
        truth.push_back(y);
        truth.push_back(w);
        truth.push_back(h);
        float best_iou = 0;
        int best_index = 0;
        int best_n = -1;
        int i = truth[0] * side_w_;
        int j = truth[1] * side_h_;
        int pos = j * side_w_ + i;
        vector<Dtype> truth_shift;
        truth_shift.clear();
        truth_shift.push_back(0);
        truth_shift.push_back(0);
        truth_shift.push_back(w);
        truth_shift.push_back(h);

        //LOG(INFO) << j << "," << i << "," << anchors_scale_;

        for (int n = 0; n < biases_size_; ++n) {
          vector<Dtype> pred(4);
          pred[2] = biases_[2 * n] / (float)(side_w_*anchors_scale_);
          pred[3] = biases_[2 * n + 1] / (float)(side_h_*anchors_scale_);

          pred[0] = 0;
          pred[1] = 0;
          float iou = box_iou(pred, truth_shift);
          if (iou > best_iou) {
            best_n = n;
            best_iou = iou;
          }
        }
        int mask_n = int_index(mask_, best_n, num_);			
        if (mask_n >= 0) {
          bool overlap = false;
          float iou;
          best_n = mask_n;
          //LOG(INFO) << best_n;
          best_index = best_n*len*stride + pos + b * bottom[0]->count(1);
          
          iou = delta_region_box(truth, swap_data, biases_,mask_[best_n], best_index, i, j, side_w_, side_h_, side_w_*anchors_scale_, side_h_*anchors_scale_, diff, coord_scale_*(2 - truth[2] * truth[3]), stride,iou_loss_,iou_normalizer_);

          if (iou > 0.5)
            recall += 1;
          if (iou > 0.75)
            recall75 += 1;
          avg_iou += iou;
          avg_iou_loss += (1 - iou);
          avg_obj += swap_data[best_index + 4 * stride];
          if (use_logic_gradient_) {
            diff[best_index + 4 * stride] = (-1.0) * (1 - swap_data[best_index + 4 * stride]) * object_scale_;
          }
          else {
            diff[best_index + 4 * stride] = (-1.0) * (1 - swap_data[best_index + 4 * stride]);
            //diff[best_index + 4 * stride] = (-1) * (1 - exp(input_data[best_index + 4 * stride] - exp(input_data[best_index + 4 * stride])));
          }

          //diff[best_index + 4 * stride] = (-1.0) * (1 - swap_data[best_index + 4 * stride]) ;

          delta_region_class_v3(swap_data, diff, best_index + 5 * stride, class_label, num_class_, class_scale_, &avg_cat, stride, use_focal_loss_); //softmax_tree_

          ++count;
          ++class_count_;
        }

      }
    }
    //LOG(INFO) << " ===================================================== " ;
    if(iou_loss_ == MSE) {
      for (int i = 0; i < diff_.count(); ++i) {
        loss += diff[i] * diff[i];
      }
      top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
    }
    else {
      for (int b = 0; b < bottom[0]->num(); b++) {
        for (int s = 0; s < stride; s++) {
          for (int n = 0; n < num_; n++) {
            int index = n*len*stride + s + b*bottom[0]->count(1);
            for (int c = 0; c < len; ++c) {
              int index2 = c*stride + index;
              //LOG(INFO)<<index2;
              if (c < 4) {
                swap_data[index2] = (input_data[index2 + 0]);
              }
              else {						
                loss += diff[index2] * diff[index2];
              }
            }
          }
        }
      }
      //LOG(INFO) << avg_iou_loss;
      loss += iou_normalizer_*avg_iou_loss;
      top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
    }
    //LOG(INFO) << "avg_noobj: " << avg_anyobj / (side_ * side_ * num_ * bottom[0]->num());	
    iter_++;
    //LOG(INFO) << "iter: " << iter <<" loss: " << loss;
    if (!(iter_ % 16))
    {
      if(time_count_>0 ) {
        LOG(INFO) << "noobj: " << score_.avg_anyobj / 10. << " obj: " << score_.avg_obj / time_count_ <<
          " iou: " << score_.avg_iou / time_count_ << " cat: " << score_.avg_cat / time_count_ << " recall: " << score_.recall / time_count_ << " recall75: " << score_.recall75 / time_count_<< " count: " << class_count_/time_count_;
        //LOG(INFO) << "avg_noobj: "<< avg_anyobj/(side_*side_*num_*bottom[0]->num()) << " avg_obj: " << avg_obj/count <<" avg_iou: " << avg_iou/count << " avg_cat: " << avg_cat/class_count << " recall: " << recall/count << " recall75: " << recall75 / count;
        score_.avg_anyobj = score_.avg_obj = score_.avg_iou = score_.avg_cat = score_.recall = score_.recall75 = 0;
        class_count_ = 0;
        time_count_ = 0;
      }
    }
    else {
      score_.avg_anyobj += avg_anyobj / (side_w_*side_h_*num_*bottom[0]->num());
      if (count > 0) {
        score_.avg_obj += avg_obj / count;
        score_.avg_iou += avg_iou / count;
        score_.avg_cat += avg_cat / count;
        score_.recall += recall / count;
        score_.recall75 += recall75 / count;
        time_count_++;
      }

    }
  }

  template <typename Dtype>
  void Yolov3Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    //LOG(INFO) <<" propagate_down: "<< propagate_down[1] << " " << propagate_down[0];
    if (propagate_down[1]) {
      LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
      if (use_logic_gradient_) {
        const Dtype* top_data = swap_.cpu_data();
        Dtype* diff = diff_.mutable_cpu_data();
        side_w_ = bottom[0]->width();
        side_h_ = bottom[0]->height();
        int len = 4 + num_class_ + 1;
        int stride = side_w_*side_h_;
        //LOG(INFO)<<swap.count(1);
        for (int b = 0; b < bottom[0]->num(); b++) {
          for (int s = 0; s < stride; s++) {
            for (int n = 0; n < num_; n++) {
              int index = n*len*stride + s + b*bottom[0]->count(1);
              //LOG(INFO)<<index;
              vector<Dtype> pred;
              float best_iou = 0;
              int best_class = -1;
              vector<Dtype> best_truth;
              for (int c = 0; c < len; ++c) {
                int index2 = c*stride + index;
                //LOG(INFO)<<index2;
                if (c == 2 || c == 3) {
                  diff[index2] = diff[index2 + 0];
                }
                else {
                  diff[index2] = diff[index2 + 0] * logistic_gradient(top_data[index2 + 0]);

                }
              }
            }
          }
        }
      }
      else {
        // non-logic_gradient formula
        // https://blog.csdn.net/yanzi6969/article/details/80505421
        // https://xmfbit.github.io/2018/03/21/cs229-supervised-learning/
        // https://zlatankr.github.io/posts/2017/03/06/mle-gradient-descent
      }
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
  STUB_GPU(Yolov3Layer);
#endif

  INSTANTIATE_CLASS(Yolov3Layer);
  REGISTER_LAYER_CLASS(Yolov3);

}  // namespace caffe
