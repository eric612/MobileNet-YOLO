#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

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
	
	static inline float logistic_activate(float x) { return 1. / (1. + exp(-x)); }
	static inline float logistic_gradient(float x) { return (1 - x)*x; }

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
	void Yolov3Layer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		Yolov3Parameter param = this->layer_param_.yolov3_param();
		iter_ = 0;
		num_class_ = param.num_class(); //20
		num_ = param.num(); //5
		side_ = bottom[0]->width();

		object_scale_ = param.object_scale(); //5.0
		noobject_scale_ = param.noobject_scale(); //1.0
		class_scale_ = param.class_scale(); //1.0
		coord_scale_ = param.coord_scale(); //1.0
		thresh_ = param.thresh(); //0.6

		for (int c = 0; c < param.anchors_size(); ++c) {
			anchors_.push_back(param.anchors(c));
		} 
		for (int c = 0; c < param.mask_size(); ++c) {
			mask_.push_back(param.mask(c));
		}
		int input_count = bottom[0]->count(1); //h*w*n*(classes+coords+1) = 13*13*5*(20+4+1)
		int label_count = bottom[1]->count(1); //30*5-
											   // outputs: classes, iou, coordinates
		int tmp_input_count = side_ * side_ * num_ * (4 + num_class_ + 1); //13*13*5*(20+4+1) label: isobj, class_label, coordinates
		int tmp_label_count = 30 * num_;
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
	void Yolov3Layer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		side_ = bottom[0]->width();
		const Dtype* label_data = bottom[1]->cpu_data(); //[label,x,y,w,h]
		diff_.ReshapeLike(*bottom[0]);
		Dtype* diff = diff_.mutable_cpu_data();
		caffe_set(diff_.count(), Dtype(0.0), diff);

		Dtype avg_anyobj(0.0), avg_obj(0.0), avg_iou(0.0), avg_cat(0.0), recall(0.0), recall75(0.0), loss(0.0);
		int count = 0;
		int class_count = 0;
		const Dtype* input_data = bottom[0]->cpu_data();
		//const Dtype* label_data = bottom[1]->cpu_data();
		Blob<Dtype> swap;
		swap.ReshapeLike(*bottom[0]);
		Dtype* swap_data = bottom[0]->mutable_cpu_data();
		//LOG(INFO) << diff_.channels() << "," << diff_.height();
		//LOG(INFO)<<bottom[0]->count(1)*bottom[0]->num();
		//LOG(INFO) << bottom[0]->num()<<","<< bottom[0]->channels() << "," << bottom[0]->height() << "," << bottom[0]->width();
		//int index = 0;
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
						if (c == 4) {
							swap_data[index2] = logistic_activate(input_data[index2 + 0]);
						}
						else {
							swap_data[index2] = (input_data[index2 + 0]);
						}
					}
					softmax_region(swap_data + index + 5 * stride, num_class_, stride);
					//LOG(INFO) << index + 5;
					int y2 = s / side_;
					int x2 = s % side_;
					//LOG(INFO) << side_;
					get_region_box(pred, swap_data, anchors_, mask_[n], index, x2, y2, side_, side_, stride);
					for (int t = 0; t < 30; ++t) {
						vector<Dtype> truth;
						Dtype x = label_data[b * 30 * 5 + t * 5 + 1];
						Dtype y = label_data[b * 30 * 5 + t * 5 + 2];
						Dtype w = label_data[b * 30 * 5 + t * 5 + 3];
						Dtype h = label_data[b * 30 * 5 + t * 5 + 4];

						if (!x)
							break;

						truth.push_back(x);
						truth.push_back(y);
						truth.push_back(w);
						truth.push_back(h);
						Dtype iou = box_iou(pred, truth);
						if (iou > best_iou) {
							best_class = label_data[b * 30 * 5 + t * 5];
							best_iou = iou;
							best_truth = truth;
						}
					}
					avg_anyobj += swap_data[index + 4 * stride];
					//diff[index + 4] = (-1) * noobject_scale_* (0 - swap_data[index + 4]);
					diff[index + 4 * stride] = (-1) * noobject_scale_ * (0 - swap_data[index + 4 * stride]) *logistic_gradient(swap_data[index + 4 * stride]);
					if (best_iou > thresh_) {
						diff[index + 4 * stride] = 0;
					}
				}
			}
			for (int t = 0; t < 30; ++t) {
				vector<Dtype> truth;
				truth.clear();
				int class_label = label_data[t * 5 + b * 30 * 5 + 0];
				float x = label_data[t * 5 + b * 30 * 5 + 1];
				float y = label_data[t * 5 + b * 30 * 5 + 2];
				float w = label_data[t * 5 + b * 30 * 5 + 3];
				float h = label_data[t * 5 + b * 30 * 5 + 4];

				if (!w)
					break;
				truth.push_back(x);
				truth.push_back(y);
				truth.push_back(w);
				truth.push_back(h);
				float best_iou = 0;
				int best_index = 0;
				int best_n = 0;
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

				for (int n = 0; n < num_; ++n) {
					int index2 = n*len*stride + pos + b * bottom[0]->count(1);
					//LOG(INFO) << index2;
					vector<Dtype> pred;
					get_region_box(pred, swap_data, anchors_, mask_[n], index2, i, j, side_, side_, stride);
					pred[2] = anchors_[2 * mask_[n]] / (float)side_;
					pred[3] = anchors_[2 * mask_[n] + 1] / (float)side_;

					pred[0] = 0;
					pred[1] = 0;
					float iou = box_iou(pred, truth_shift);
					if (iou > best_iou) {
						best_index = index2;
						best_iou = iou;
						best_n = n;
					}
				}
				float iou;
				iou = delta_region_box(truth, swap_data, anchors_, mask_[best_n], best_index, i, j, side_, side_, diff, coord_scale_*(2 - truth[2] * truth[3]), stride);

				if (iou > 0.5)
					recall += 1;
				if (iou > 0.75)
					recall75 += 1;
				avg_iou += iou;
				avg_obj += swap_data[best_index + 4 * stride];

				diff[best_index + 4 * stride] = (-1.0) * object_scale_ * (1 - swap_data[best_index + 4 * stride]) * logistic_gradient(swap_data[best_index + 4 * stride]);


				delta_region_class(swap_data, diff, best_index + 5 * stride, class_label, num_class_, class_scale_, &avg_cat, stride); //softmax_tree_

				++count;
				++class_count;
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
			LOG(INFO) << "avg_noobj: " << score_.avg_anyobj / 10. << " avg_obj: " << score_.avg_obj / 10. <<
				" avg_iou: " << score_.avg_iou / 10. << " avg_cat: " << score_.avg_cat / 10. << " recall: " << score_.recall / 10. << " recall75: " << score_.recall75 / 10.;
			//LOG(INFO) << "avg_noobj: "<< avg_anyobj/(side_*side_*num_*bottom[0]->num()) << " avg_obj: " << avg_obj/count <<" avg_iou: " << avg_iou/count << " avg_cat: " << avg_cat/class_count << " recall: " << recall/count << " recall75: " << recall75 / count;
			score_.avg_anyobj = score_.avg_obj = score_.avg_iou = score_.avg_cat = score_.recall = score_.recall75 = 0;
		}
		else {
			score_.avg_anyobj += avg_anyobj / (side_*side_*num_*bottom[0]->num());
			score_.avg_obj += avg_obj / count;
			score_.avg_iou += avg_iou / count;
			score_.avg_cat += avg_cat / class_count;
			score_.recall += recall / count;
			score_.recall75 += recall75 / count;
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
	//STUB_GPU(DetectionLossLayer);
#endif

	INSTANTIATE_CLASS(Yolov3Layer);
	REGISTER_LAYER_CLASS(Yolov3);

}  // namespace caffe
