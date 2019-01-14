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
#include "caffe/util/bbox_util.hpp"
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV


namespace caffe {
	


	template <typename Dtype>
	Dtype overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2)
	{
		float l1 = x1 - w1 / 2;
		float l2 = x2 - w2 / 2;
		float left = l1 > l2 ? l1 : l2;
		float r1 = x1 + w1 / 2;
		float r2 = x2 + w2 / 2;
		float right = r1 < r2 ? r1 : r2;
		return right - left;
	}
	template <typename Dtype>
	Dtype box_intersection(vector<Dtype> a, vector<Dtype> b)
	{
		float w = overlap(a[0], a[2], b[0], b[2]);
		float h = overlap(a[1], a[3], b[1], b[3]);
		if (w < 0 || h < 0) return 0;
		float area = w*h;
		return area;
	}
	template <typename Dtype>
	Dtype box_union(vector<Dtype> a, vector<Dtype> b)
	{
		float i = box_intersection(a, b);
		float u = a[2] * a[3] + b[2] * b[3] - i;
		return u;
	}
	template <typename Dtype>
	Dtype box_iou(vector<Dtype> a, vector<Dtype> b)
	{
		return box_intersection(a, b) / box_union(a, b);
	}
	template <typename Dtype>
	void delta_region_class_v3(Dtype* input_data, Dtype* &diff, int index, int class_label, int classes, float scale, Dtype* avg_cat, int stride, bool use_focal_loss)
	{
		if (diff[index]) {
			diff[index + stride*class_label] = 1 - input_data[index + stride*class_label];
			*avg_cat += input_data[index + stride*class_label];
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
	void get_region_box(vector<Dtype> &b, Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {

		b.clear();
		b.push_back((i + (x[index + 0 * stride])) / lw);
		b.push_back((j + (x[index + 1 * stride])) / lh);
		b.push_back(exp(x[index + 2 * stride]) * biases[2 * n] / (w));
		b.push_back(exp(x[index + 3 * stride]) * biases[2 * n + 1] / (h));
	}
	template <typename Dtype>
	Dtype delta_region_box(vector<Dtype> truth, Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int lw, int lh, int w, int h, Dtype* delta, float scale, int stride) {
		vector<Dtype> pred;
		pred.clear();
		get_region_box(pred, x, biases, n, index, i, j,lw,lh, w, h, stride);

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
	template <typename Dtype>
	void Yolov3Layer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		Yolov3Parameter param = this->layer_param_.yolov3_param();
		iter_ = 0;
		num_class_ = param.num_class(); //20
		num_ = param.num(); //5
		side_ = bottom[0]->width();
		anchors_scale_ = param.anchors_scale();
		object_scale_ = param.object_scale(); //5.0
		noobject_scale_ = param.noobject_scale(); //1.0
		class_scale_ = param.class_scale(); //1.0
		coord_scale_ = param.coord_scale(); //1.0
		thresh_ = param.thresh(); //0.6
		use_logic_gradient_ = param.use_logic_gradient();
		use_focal_loss_  = param.use_focal_loss();
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
		int tmp_input_count = side_ * side_ * num_ * (4 + num_class_ + 1); //13*13*5*(20+4+1) label: isobj, class_label, coordinates
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
		side_ = bottom[0]->width();
		const Dtype* label_data = bottom[1]->cpu_data(); //[label,x,y,w,h]
		if (diff_.width() != bottom[0]->width()) {
			diff_.ReshapeLike(*bottom[0]);
		}
		Dtype* diff = diff_.mutable_cpu_data();
		caffe_set(diff_.count(), Dtype(0.0), diff);

		Dtype avg_anyobj(0.0), avg_obj(0.0), avg_iou(0.0), avg_cat(0.0), recall(0.0), recall75(0.0), loss(0.0);
		int count = 0;
		
		const Dtype* input_data = bottom[0]->cpu_data();
		//const Dtype* label_data = bottom[1]->cpu_data();		
		if (swap_.width() != bottom[0]->width()) {
			swap_.ReshapeLike(*bottom[0]);
		}
		Dtype* swap_data = swap_.mutable_cpu_data();
		int len = 4 + num_class_ + 1;
		int stride = side_*side_;
		for (int b = 0; b < bottom[0]->num(); b++) {
			for (int s = 0; s < side_*side_; s++) {
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
					int y2 = s / side_;
					int x2 = s % side_;
					get_region_box(pred, swap_data, biases_, mask_[n], index, x2, y2, side_, side_, side_*anchors_scale_, side_*anchors_scale_, stride);
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
						delta_region_box(best_truth, swap_data, biases_, mask_[n], index, x2, y2, side_, side_,
							side_*anchors_scale_, side_*anchors_scale_, diff, coord_scale_*(2 - best_truth[2] * best_truth[3]), stride);
					}
				}
			}
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
				int i = truth[0] * side_;
				int j = truth[1] * side_;
				int pos = j * side_ + i;
				vector<Dtype> truth_shift;
				truth_shift.clear();
				truth_shift.push_back(0);
				truth_shift.push_back(0);
				truth_shift.push_back(w);
				truth_shift.push_back(h);
				//int size = coords_ + num_class_ + 1;
				//LOG(INFO) << biases_size_;
				for (int n = 0; n < biases_size_; ++n) {
					
					//LOG(INFO) << index2;
					vector<Dtype> pred(4);
					pred[2] = biases_[2 * n] / (float)(side_*anchors_scale_);
					pred[3] = biases_[2 * n + 1] / (float)(side_*anchors_scale_);

					pred[0] = 0;
					pred[1] = 0;
					float iou = box_iou(pred, truth_shift);
					if (iou > best_iou) {
						//best_index = index2;
						best_n = n;
						best_iou = iou;
					}
				}
				//LOG(INFO) << best_n;
				int mask_n = int_index(mask_, best_n, num_);
				
				//LOG(INFO) << mask_n;
				
				if (mask_n >= 0) {
					float iou;
					best_n = mask_n;
					//LOG(INFO) << best_n;
					best_index = best_n*len*stride + pos + b * bottom[0]->count(1);
					
					iou = delta_region_box(truth, swap_data, biases_,mask_[best_n], best_index, i, j, side_, side_, side_*anchors_scale_, side_*anchors_scale_, diff, coord_scale_*(2 - truth[2] * truth[3]), stride);

					if (iou > 0.5)
						recall += 1;
					if (iou > 0.75)
						recall75 += 1;
					avg_iou += iou;
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

		for (int i = 0; i < diff_.count(); ++i) {
			loss += diff[i] * diff[i];
		}
		top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
		//LOG(INFO) << "avg_noobj: " << avg_anyobj / (side_ * side_ * num_ * bottom[0]->num());	
		iter_++;
		//LOG(INFO) << "iter: " << iter <<" loss: " << loss;
		if (!(iter_ % 10))
		{
			if(time_count_>0 ) {
				LOG(INFO) << "avg_noobj: " << score_.avg_anyobj / 10. << " avg_obj: " << score_.avg_obj / time_count_ <<
					" avg_iou: " << score_.avg_iou / time_count_ << " avg_cat: " << score_.avg_cat / time_count_ << " recall: " << score_.recall / time_count_ << " recall75: " << score_.recall75 / time_count_<< " count: " << class_count_/time_count_;
				//LOG(INFO) << "avg_noobj: "<< avg_anyobj/(side_*side_*num_*bottom[0]->num()) << " avg_obj: " << avg_obj/count <<" avg_iou: " << avg_iou/count << " avg_cat: " << avg_cat/class_count << " recall: " << recall/count << " recall75: " << recall75 / count;
				score_.avg_anyobj = score_.avg_obj = score_.avg_iou = score_.avg_cat = score_.recall = score_.recall75 = 0;
				class_count_ = 0;
				time_count_ = 0;
			}
		}
		else {
			score_.avg_anyobj += avg_anyobj / (side_*side_*num_*bottom[0]->num());
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
				side_ = bottom[0]->width();
				int len = 4 + num_class_ + 1;
				int stride = side_*side_;
				//LOG(INFO)<<swap.count(1);
				for (int b = 0; b < bottom[0]->num(); b++) {
					for (int s = 0; s < side_*side_; s++) {
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
