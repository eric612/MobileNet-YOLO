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
#include "caffe/util/bbox_util.hpp"
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

int iter = 0;
namespace caffe {
template <typename Dtype>
Dtype softmax_region(Dtype* input, int classes)
{
  Dtype sum = 0;
  Dtype large = input[0];
  for (int i = 0; i < classes; ++i){
    if (input[i] > large)
      large = input[i];
  }
  for (int i = 0; i < classes; ++i){
    Dtype e = exp(input[i] - large);
    sum += e;
    input[i] = e;
  }
  for (int i = 0; i < classes; ++i){
    input[i] = input[i] / sum;
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
void get_region_box(vector<Dtype> &b, Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int w, int h) {

	b.clear();
	b.push_back((i + sigmoid(x[index + 0])) / w);
	b.push_back((j + sigmoid(x[index + 1])) / h);
	b.push_back(exp(x[index + 2]) * biases[2 * n] / (w));
	b.push_back(exp(x[index + 3]) * biases[2 * n + 1] / (h));
}
template <typename Dtype>
Dtype delta_region_box(vector<Dtype> truth, Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int w, int h, Dtype* delta, float scale) {
	vector<Dtype> pred;
	pred.clear();
	get_region_box(pred, x, biases, n, index, i, j, w, h);

	float iou = box_iou(pred, truth);
	//LOG(INFO) << pred[0] << "," << pred[1] << "," << pred[2] << "," << pred[3] << ";"<< truth[0] << "," << truth[1] << "," << truth[2] << "," << truth[3];
	float tx = truth[0] * w - i; //0.5
	float ty = truth[1] * h - j; //0.5
	float tw = log(truth[2] * w / biases[2 * n]); //truth[2]=biases/w tw = 0
	float th = log(truth[3] * h / biases[2 * n + 1]); //th = 0

	//delta[index + 0] =(-1) * scale * (tx - sigmoid(x[index + 0])) * sigmoid(x[index + 0]) * (1 - sigmoid(x[index + 0]));
	//delta[index + 1] =(-1) * scale * (ty - sigmoid(x[index + 1])) * sigmoid(x[index + 1]) * (1 - sigmoid(x[index + 1]));
	delta[index + 0] =(-1) * scale * (tx - sigmoid(x[index + 0])) * sigmoid(x[index + 0]) * (1 - sigmoid(x[index + 0]));
	delta[index + 1] =(-1) * scale * (ty - sigmoid(x[index + 1])) * sigmoid(x[index + 1]) * (1 - sigmoid(x[index + 1]));
	//delta[index + 0] = (-1) * scale * (tx - x[index + 0]);
	//delta[index + 1] = (-1) * scale * (ty - x[index + 1]);
	delta[index + 2] = (-1) * scale * (tw - x[index + 2]);
	delta[index + 3] = (-1) * scale * (th - x[index + 3]);

	return iou;
}

template <typename Dtype>
void delta_region_class(Dtype* input_data, Dtype* &diff, int index, int class_label, int classes, float scale, Dtype* avg_cat)
{
	//float grad = -2 * (1 - input_data[index + class_label])*logf(fmaxf(input_data[index + class_label], 0.0000001))*input_data[index + class_label]
	//	+ (1 - input_data[index + class_label])*(1 - input_data[index + class_label]);
	//LOG(INFO) << grad;
	for (int n = 0; n < classes; ++n) {
		diff[index + n] = (-1.0) * scale * (((n == class_label) ? 1 : 0) - input_data[index + n]);
		//[index + n] *= 0.5*grad;
		//std::cout<<diff[index+n]<<",";
		if (n == class_label) {
			*avg_cat += input_data[index + n];
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
	CHECK_EQ(input_count, tmp_input_count);
	CHECK_EQ(label_count, tmp_label_count);
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
	real_diff_.ReshapeLike(*bottom[0]); 
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	side_ = bottom[0]->width();
	const Dtype* label_data = bottom[1]->cpu_data(); //[label,x,y,w,h]
	Dtype* diff = diff_.mutable_cpu_data();
	caffe_set(diff_.count(), Dtype(0.0), diff);
	Dtype avg_anyobj(0.0), avg_obj(0.0), avg_iou(0.0), avg_cat(0.0), recall(0.0), loss(0.0);
	int count = 0;
	int class_count = 0;
	//*********************************************************Reshape********************************************************//
	Blob<Dtype> swap;
	swap.Reshape(bottom[0]->num(), bottom[0]->height()*bottom[0]->width(), num_, bottom[0]->channels() / num_);
	diff_.Reshape(bottom[0]->num(), bottom[0]->height()*bottom[0]->width(), num_, bottom[0]->channels() / num_);
	Dtype* swap_data = swap.mutable_cpu_data();
	int index = 0;
	for (int b = 0; b < bottom[0]->num(); ++b) {
		for (int h = 0; h < bottom[0]->height(); ++h) {
			for (int w = 0; w < bottom[0]->width(); ++w) {
				for (int c = 0; c < bottom[0]->channels(); ++c) {
					swap_data[index++] = bottom[0]->data_at(b, c, h, w);
				}
			}
		}
	}
	for (int b = 0; b < swap.num(); ++b) {
		for (int c = 0; c < swap.channels(); ++c) {
			for (int h = 0; h < swap.height(); ++h) {
				int index = b * swap.channels() * swap.height() * swap.width() + c * swap.height() * swap.width() + h * swap.width();
				//swap_data[index + 0] = logistic_activate(swap_data[index + 0]);
				//swap_data[index + 1] = logistic_activate(swap_data[index + 1]);
				swap_data[index + 4] = logistic_activate(swap_data[index + 4]);
				//CHECK_GE(swap_data[index], 0);
			}
		}
	}

	for (int b = 0; b < swap.num(); ++b) {
		for (int c = 0; c < swap.channels(); ++c) {
			for (int h = 0; h < swap.height(); ++h) {
				int index = b * swap.channels() * swap.height() * swap.width() + c * swap.height() * swap.width() + h * swap.width() + 5;

				softmax_region(swap_data + index, num_class_);
				for (int i = 0; i < num_class_; ++i)
					CHECK_GE(swap_data[index + i], 0);
			}
		}
	}

	int best_num = 0;
	for (int b = 0; b < swap.num(); ++b) {
		for (int j = 0; j < side_; ++j) {
			for (int i = 0; i < side_; ++i) {
				for (int n = 0; n < num_; ++n) {
					int index = b * swap.channels() * swap.height() * swap.width() + (j * side_ + i) * swap.height() * swap.width() + n * swap.width();
					CHECK_EQ(swap_data[index], swap.data_at(b, j * side_ + i, n, 0));

					//std::cout<<index<<std::endl;
					vector<Dtype> pred;
					get_region_box(pred, swap_data, biases_, n, index, i, j, side_, side_);
					//fprintf(fp, "%f\n", swap_data[index]);
					float best_iou = 0;
					int best_class = -1;
					vector<Dtype> best_truth;
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
					avg_anyobj += swap_data[index + 4];
					//diff[index + 4] = (-1) * noobject_scale_* (0 - swap_data[index + 4]);
					diff[index + 4] =  (-1) * noobject_scale_ * (0 - swap_data[index + 4]) *logistic_gradient(swap_data[index + 4]) ;
					if (best_iou > thresh_) {
						best_num++;
						diff[index + 4] = 0;
					}
					if (iter < 12800 / bottom[0]->num()) {
						vector<Dtype> truth;
						truth.clear();
						truth.push_back((i + .5) / (float)side_); //center of i,j
						truth.push_back((j + .5) / (float)side_);
						truth.push_back((biases_[2 * n]) / (float)side_); //anchor boxes
						truth.push_back((biases_[2 * n + 1]) / (float)side_);
						delta_region_box(truth, swap_data, biases_, n, index, i, j, side_, side_, diff, .01);
					}
				}
			}
		}
		for (int t = 0; t < 30; ++t){
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
			int size = coords_ + num_class_ + 1; 

			for (int n = 0; n < num_; ++ n){ 
				int index = b * bottom[0]->count(1) + pos * size * num_ + n * size;
				vector<Dtype> pred;
				get_region_box(pred,swap_data, biases_, n, index, i, j, side_, side_); 
				if (bias_match_){
					pred[2] = biases_[2 * n] / (float)side_;
					pred[3] = biases_[2 * n + 1] / (float)side_;
				}
				pred[0] = 0;
				pred[1] = 0;
				float iou = box_iou(pred, truth_shift);
				if (iou > best_iou){
					best_index = index;
					best_iou = iou;
					best_n = n;
				}
			}
			float iou;
			if (rescore_) {
				iou = delta_region_box(truth, swap_data, biases_, best_n, best_index, i, j, side_, side_, diff, coord_scale_);
			}
			else {
				iou = delta_region_box(truth, swap_data, biases_, best_n, best_index, i, j, side_, side_, diff, coord_scale_*( 2 - truth[2]*truth[3]));
			}
			if (iou > 0.5)
				recall += 1;
			avg_iou += iou;
			avg_obj += swap_data[best_index + 4];
			if (rescore_) {
				diff[best_index + 4] = (-1.0)* object_scale_ * (iou - swap_data[best_index + 4])* logistic_gradient(swap_data[best_index + 4]);
			}
			else {
				
				diff[best_index + 4] = (-1.0) * object_scale_ * (1 - swap_data[best_index + 4]) * logistic_gradient(swap_data[best_index + 4]);
			}
			

			delta_region_class(swap_data, diff, best_index + 5, class_label, num_class_, class_scale_, &avg_cat); //softmax_tree_

			++count;
			++class_count;	
		}
	}
	
	Dtype* real_diff = real_diff_.mutable_cpu_data();    
	int sindex = 0;

	for (int b = 0; b < real_diff_.num(); ++b) {
		for (int h = 0; h < real_diff_.height(); ++h) {
			for (int w = 0; w < real_diff_.width(); ++w) {
				for (int c = 0; c < real_diff_.channels(); ++c) {
					int rindex = b * real_diff_.height() * real_diff_.width() * real_diff_.channels() + c * real_diff_.height() * real_diff_.width() + h * real_diff_.width() + w;
					Dtype e = diff[sindex];
					real_diff[rindex] = e;
					sindex++;
				}
			}
		}
	}
	for (int i = 0; i < real_diff_.count(); ++i) {
	loss += real_diff[i] * real_diff[i];
	}
	top[0]->mutable_cpu_data()[0] = loss;
	//LOG(INFO) << "avg_noobj: " << avg_anyobj / (side_ * side_ * num_ * bottom[0]->num());	
	iter ++;
	//LOG(INFO) << "iter: " << iter <<" loss: " << loss;
	if (!(iter % 10))
	{
		LOG(INFO) << "avg_noobj: "<< avg_anyobj/(side_*side_*num_*bottom[0]->num()) << " avg_obj: " << avg_obj/count <<" avg_iou: " << avg_iou/count << " avg_cat: " << avg_cat/class_count << " recall: " << recall/count << " class_count: "<< class_count;
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
			real_diff_.cpu_data(),
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
