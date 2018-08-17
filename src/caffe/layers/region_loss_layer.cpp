#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/util/bbox_util.hpp"
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

int iter = 0;
namespace caffe {
template <typename Dtype>
Dtype softmax_region(Dtype* input, int classes , int stride)
{
  Dtype sum = 0;
  Dtype large = input[0];
  for (int i = 0; i < classes; ++i){
    if (input[i*stride] > large)
      large = input[i*stride];
  }
  for (int i = 0; i < classes; ++i){
    Dtype e = exp(input[i*stride] - large);
    sum += e;
    input[i*stride] = e;
  }
  for (int i = 0; i < classes; ++i){
    input[i*stride] = input[i*stride] / sum;
  }
  return 0;
}

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

static inline float logistic_activate(float x) { return 1. / (1. + exp(-x)); }
static inline float logistic_gradient(float x) { return (1 - x)*x; }
template <typename Dtype>
void get_region_box(vector<Dtype> &b, Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int w, int h, int stride) {

	b.clear();
	b.push_back((i + sigmoid(x[index + 0 * stride])) / w);
	b.push_back((j + sigmoid(x[index + 1*stride])) / h);
	b.push_back(exp(x[index + 2 * stride]) * biases[2 * n] / (w));
	b.push_back(exp(x[index + 3 * stride]) * biases[2 * n + 1] / (h));
}
template <typename Dtype>
Dtype delta_region_box(vector<Dtype> truth, Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int w, int h, Dtype* delta, float scale,int stride) {
	vector<Dtype> pred;
	pred.clear();
	get_region_box(pred, x, biases, n, index, i, j, w, h,stride);

	float iou = box_iou(pred, truth);
	//LOG(INFO) << pred[0] << "," << pred[1] << "," << pred[2] << "," << pred[3] << ";"<< truth[0] << "," << truth[1] << "," << truth[2] << "," << truth[3];
	float tx = truth[0] * w - i; //0.5
	float ty = truth[1] * h - j; //0.5
	float tw = log(truth[2] * w / biases[2 * n]); //truth[2]=biases/w tw = 0
	float th = log(truth[3] * h / biases[2 * n + 1]); //th = 0

	delta[index + 0] =(-1) * scale * (tx - sigmoid(x[index + 0 * stride])) * sigmoid(x[index + 0 * stride]) * (1 - sigmoid(x[index+ 0*stride ]));
	delta[index + 1 * stride] =(-1) * scale * (ty - sigmoid(x[index + 1 * stride])) * sigmoid(x[index + 1 * stride]) * (1 - sigmoid(x[index + 1*stride]));
	//delta[index + 0] = (-1) * scale * (tx - x[index + 0]);
	//delta[index + 1] = (-1) * scale * (ty - x[index + 1]);
	delta[index + 2* stride] = (-1) * scale * (tw - x[index + 2 * stride]);
	delta[index + 3* stride] = (-1) * scale * (th - x[index + 3 * stride]);

	return iou;
}

template <typename Dtype>
void delta_region_class(Dtype* input_data, Dtype* &diff, int index, int class_label, int classes, float scale, Dtype* avg_cat,int stride)
{
	for (int n = 0; n < classes; ++n) {
		diff[index + n*stride] = (-1.0) * scale * (((n == class_label) ? 1 : 0) - input_data[index + n*stride]);
		//std::cout<<diff[index+n]<<",";
		if (n == class_label) {
			*avg_cat += input_data[index + n*stride];
			//std::cout<<"avg_cat:"<<input_data[index+n]<<std::endl; 
		}
	}
}



template <typename Dtype>
void RegionLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	RegionLossParameter param = this->layer_param_.region_loss_param();
  
	side_ = param.side(); //13
	bias_match_ = param.bias_match(); //
	num_class_ = param.num_class(); //20
	coords_ = param.coords(); //4
	num_ = param.num(); //5
	softmax_ = param.softmax(); //
	side_ = bottom[0]->width();
	rescore_ = param.rescore(); //
	class_map_ = param.class_map();
	if (class_map_ != "") {
		string line;
		std::fstream fin(class_map_.c_str());
		if (!fin){
			LOG(INFO) << "no map file";
		}
		int index = 0;
		int id = 0;
		while (getline(fin, line)){
			stringstream ss;
			ss << line;
			ss >> id;
      
			cls_map_[index] = id;
			index ++;
		}
		fin.close();
	}  
	object_scale_ = param.object_scale(); //5.0
	noobject_scale_ = param.noobject_scale(); //1.0
	class_scale_ = param.class_scale(); //1.0
	coord_scale_ = param.coord_scale(); //1.0
  
	//absolute_ = param.absolute();
	thresh_ = param.thresh(); //0.6
	//random_ = param.random();  
	if (param.biases_size() < 10)
	{
		for (int c = 0; c < 10; ++c) {
			biases_.push_back(0);
		} //0.73 0.87;2.42 2.65;4.30 7.04;10.24 4.59;12.68 11.87;
	}
	else {
		for (int c = 0; c < param.biases_size(); ++c) {
			biases_.push_back(param.biases(c));
		} //0.73 0.87;2.42 2.65;4.30 7.04;10.24 4.59;12.68 11.87;
	}


	int input_count = bottom[0]->count(1); //h*w*n*(classes+coords+1) = 13*13*5*(20+4+1)
	int label_count = bottom[1]->count(1); //30*5-
	// outputs: classes, iou, coordinates
	int tmp_input_count = side_ * side_ * num_ * (coords_ + num_class_ + 1); //13*13*5*(20+4+1) label: isobj, class_label, coordinates
	int tmp_label_count = 30 * num_;
	//CHECK_EQ(input_count, tmp_input_count);
	//CHECK_EQ(label_count, tmp_label_count);
}
typedef struct {
	float x, y, w, h;
} box;



template <typename Dtype>
void RegionLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//LossLayer<Dtype>::Reshape(bottom, top);
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
	diff_.ReshapeLike(*bottom[0]);
	//real_diff_.ReshapeLike(*bottom[0]); 
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	side_ = bottom[0]->width();
	const Dtype* label_data = bottom[1]->cpu_data(); //[label,x,y,w,h]
	diff_.ReshapeLike(*bottom[0]);
	Dtype* diff = diff_.mutable_cpu_data();
	caffe_set(diff_.count(), Dtype(0.0), diff);
	
	Dtype avg_anyobj(0.0), avg_obj(0.0), avg_iou(0.0), avg_cat(0.0), recall(0.0), recall75(0.0) , loss(0.0);
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
	int len = coords_ + num_class_ + 1;
	int stride = side_*side_;
	//LOG(INFO)<<swap.count(1);
	for (int b = 0; b < bottom[0]->num(); b++) {
		for (int s = 0; s < side_*side_; s++) {
			for (int n = 0; n < num_; n++) {
				int index =  n*len*stride + s + b*bottom[0]->count(1);
				//LOG(INFO)<<index;
				vector<Dtype> pred;
				float best_iou = 0;
				int best_class = -1;
				vector<Dtype> best_truth;
				for (int c = 0; c < len; ++c) {
					int index2 = c*stride + index;
					//LOG(INFO)<<index2;
					if(c==4) {
						swap_data[index2] = logistic_activate(input_data[index2 + 0]);
					}
					else {
						swap_data[index2] = (input_data[index2 + 0]);
					}
				}			
				softmax_region(swap_data + index + 5*stride, num_class_,stride);
				//LOG(INFO) << index + 5;
				int y2 = s / side_;
				int x2 = s % side_; 
				//LOG(INFO) << side_;
				get_region_box(pred, swap_data, biases_, n, index, x2, y2, side_, side_, stride);
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
				avg_anyobj += swap_data[index + 4*stride];
				//diff[index + 4] = (-1) * noobject_scale_* (0 - swap_data[index + 4]);
				diff[index + 4*stride] = (-1) * noobject_scale_ * (0 - swap_data[index + 4 * stride]) *logistic_gradient(swap_data[index + 4 * stride]);
				if (best_iou > thresh_) {
					diff[index + 4 * stride] = 0;
				}

				if (iter < 12800 / bottom[0]->num()) {
					vector<Dtype> truth;
					truth.clear();
					truth.push_back((x2 + .5) / side_); //center of i,j
					truth.push_back((y2 + .5) / side_);
					truth.push_back((biases_[2 * n]) / side_); //anchor boxes
					truth.push_back((biases_[2 * n + 1]) / side_);
					//LOG(INFO)<<truth[2]<<","<<truth[3];
					//LOG(INFO)<<index;
					delta_region_box(truth, swap_data, biases_, n, index, x2, y2, side_, side_, diff, .01, stride);
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
				get_region_box(pred, swap_data, biases_, n, index2, i, j, side_, side_,stride);
				if (bias_match_) {
					pred[2] = biases_[2 * n] / (float)side_;
					pred[3] = biases_[2 * n + 1] / (float)side_;
				}
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
			if (rescore_) {
				iou = delta_region_box(truth, swap_data, biases_, best_n, best_index, i, j, side_, side_, diff, coord_scale_, stride);
			}
			else {
				iou = delta_region_box(truth, swap_data, biases_, best_n, best_index, i, j, side_, side_, diff, coord_scale_*(2 - truth[2] * truth[3]), stride);
			}
			if (iou > 0.5)
				recall += 1;
			if (iou > 0.75)
				recall75 += 1;
			avg_iou += iou;
			avg_obj += swap_data[best_index + 4 * stride];
			if (rescore_) {
				diff[best_index + 4 * stride] = (-1.0)* object_scale_ * (iou - swap_data[best_index + 4 * stride])* logistic_gradient(swap_data[best_index + 4 * stride]);
			}
			else {
				//LOG(INFO)<<"test";
				diff[best_index + 4 * stride] = (-1.0) * object_scale_ * (1 - swap_data[best_index + 4 * stride]) * logistic_gradient(swap_data[best_index + 4 * stride]);
			}


			delta_region_class(swap_data, diff, best_index + 5*stride, class_label, num_class_, class_scale_, &avg_cat, stride); //softmax_tree_

			++count;
			++class_count;
		}
	}

	for (int i = 0; i < diff_.count(); ++i) {
		loss += diff[i] * diff[i];
	}
	top[0]->mutable_cpu_data()[0] = loss/ bottom[0]->num();
	//LOG(INFO) << "avg_noobj: " << avg_anyobj / (side_ * side_ * num_ * bottom[0]->num());	
	iter ++;
	//LOG(INFO) << "iter: " << iter <<" loss: " << loss;
	if (!(iter % 10))
	{
		LOG(INFO) << "avg_noobj: " << score_.avg_anyobj/10. << " avg_obj: " << score_.avg_obj /10. << 
			" avg_iou: " << score_.avg_iou/10. << " avg_cat: " << score_.avg_cat/10.  << " recall: " << score_.recall/10.  << " recall75: " << score_.recall75/10. ;
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
void RegionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

INSTANTIATE_CLASS(RegionLossLayer);
REGISTER_LAYER_CLASS(RegionLoss);

}  // namespace caffe
