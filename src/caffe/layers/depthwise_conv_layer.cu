#include <vector>
#include <algorithm>
#include <cfloat>
#include "caffe/layers/depthwise_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"


/*
 * The depthwise layer for mobilenet.   only for stride 1
 */

namespace caffe {

template <typename Dtype>
__global__ void ConvForward(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width,const int conved_height,
		const int conved_width,const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* const top_data,const Dtype* const weight,const Dtype* const bias,const bool bias_term_) {
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int pw = index % conved_width;
		const int ph = (index / conved_width) % conved_height;
		const int c = (index / conved_width / conved_height) % channels;
		const int n = index / conved_width / conved_height / channels;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height + pad_h);
		int wend = min(wstart + kernel_w, width + pad_w);
//		const int pool_size = (hend - hstart) * (wend - wstart);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height);
		wend = min(wend, width);
		Dtype aveval = 0;
		const Dtype* const bottom_slice =
		bottom_data + (n * channels + c) * height * width;
		const Dtype* const weight_slice =
		weight + c * kernel_h * kernel_w;
//		if (index==1) {
//			printf("pw%d ph%d c%d n%d \n",pw,ph,c,n);
//			printf("hstart%d wstart%d hend%d wend%d \n",hstart,wstart,hend,wend);
//		}

		int khstart=hend<kernel_h?kernel_h-hend:0;
		int kwstart=wend<kernel_w?kernel_w-wend:0;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {

				aveval += bottom_slice[h * width + w]*weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)];
//				if (index==1) {
//					printf("pos:h%d w%d\n",h,w);
//					printf("cal:bottom%f weight%f\n",bottom_slice[h * width + w],weight_slice[(h-hstart) * kernel_w + (w-wstart)]);
//				}
			}
		}
		if(bias_term_) {
			aveval+=bias[c];
		}
		top_data[index] = aveval;
	}
}

template<typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//	std::cout << "fp" << std::endl;
	const Dtype* weight = this->blobs_[0]->gpu_data();
	int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
	int* stride_data = this->stride_.mutable_cpu_data();
	int* pad_data = this->pad_.mutable_cpu_data();

	for (int i = 0; i < bottom.size(); ++i) {
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* top_data = top[i]->mutable_gpu_data();
		const int count = top[i]->count();
		vector<int> shape_ = bottom[i]->shape();
		const int channels_ = shape_[1];
		const int height_ = shape_[2];
		const int width_ = shape_[3];

		const int kernel_h_ = kernel_shape_data[0];
		const int kernel_w_ = kernel_shape_data[1];
		const int stride_h_ = stride_data[0];
		const int stride_w_ = stride_data[1];
		const int pad_h_ = pad_data[0];
		const int pad_w_ = pad_data[1];

		const int conved_height = this->output_shape_[0];
		const int conved_weight = this->output_shape_[1];

		const bool bias_term_ = this->bias_term_;

		if (bias_term_) {
			const Dtype* const bias = this->blobs_[1]->gpu_data();
			ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,bias,bias_term_);
		} else {
			ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,0,bias_term_);
		}
	}
}

template <typename Dtype>
__global__ void ConvBackward(const int nthreads,
const Dtype* const top_diff,
const int num, const int channels, const int height,
const int width, const int conved_height, const int conved_width,
const int kernel_h, const int kernel_w, const int stride_h,
const int stride_w, const int pad_h, const int pad_w,
Dtype* const bottom_diff,
const Dtype* const weight) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int w = index % width + pad_w;
		const int h = (index / width) % height + pad_h;
		const int c = (index / width / height) % channels;
		const int n = index / width / height / channels;
		
		const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
		const int phend = min(h / stride_h + 1, conved_height);
		const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
		const int pwend = min(w / stride_w + 1, conved_width);
		
		const int khstart=(h >= kernel_h) ? ((h-kernel_h)%stride_h)+(kernel_h-stride_h): h;
		const int kwstart=(w >= kernel_w) ? ((w-kernel_w)%stride_w)+(kernel_w-stride_w) : w;
		
		Dtype gradient = 0;
		const Dtype* const top_diff_slice =
		top_diff + (n * channels + c) * conved_height * conved_width;
		
		const Dtype* const weight_slice =weight + c * kernel_h * kernel_w;
		
//		if (index==2) {
//			printf("w%d h%d c%d n%d \n",w,h,c,n);
//			printf("phstart%d phend%d pwstart%d pwend%d \n",phstart,phend,pwstart,pwend);
//		}
		
		for (int ph = phstart; ph < phend; ++ph) {
			for (int pw = pwstart; pw < pwend; ++pw) {
				int kh=khstart-(ph-phstart)*stride_h;
				int kw=kwstart-(pw-pwstart)*stride_w;
				gradient += top_diff_slice[ph * conved_width + pw] *weight_slice[kh*kernel_w+kw];
				
//						if (index==2) {
//							printf("pos:ph%d pw%d kh%d kw%d\n",ph,pw,kh,kw);
//							printf("cal:top_diff%f weight%f\n",top_diff_slice[ph * conved_width + pw],weight_slice[kh*kernel_w+kw]);
//				//			printf("cal:top_diff%f weight%f\n",top_diff_slice[ph * conved_width + pw],weight_slice[kh*kernel_w+kw]);
//						}
			}
		}
		bottom_diff[index] = gradient;
	}
}

__device__ float atomicAddme(float* address, float val)
{
    return atomicAdd(address,val);
}

__device__ double atomicAddme(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}



#define DIVIDE_CEIL(a,b) a/b+((a/b*b)<a)


template <typename Dtype>
__global__ void ConvBackwardWeight(const int nthreads,
const Dtype* const top_diff,
const int num, const int channels, const int height,
const int width, const int conved_height, const int conved_width,
const int kernel_h, const int kernel_w, const int stride_h,
const int stride_w, const int pad_h, const int pad_w,
Dtype* const weight_diff,
const Dtype* const bottom_data) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int kw=index % kernel_w;
		const int kh= (index /kernel_w)%kernel_h;
		const int c=index /kernel_w/kernel_h;
		
//		if (index==5) {
//			printf("kh%d kw%d kc%d\n",kh,kw,c);
//		}
		Dtype gradient = 0;
		for( int n=0;n<num;n++) {
			
			const Dtype* const top_diff_slice = top_diff + (n * channels + c) * conved_height * conved_width;
			const Dtype* const bottom_data_slice = bottom_data + (n * channels + c) * height * width;
		
			
			const int phstart=max(DIVIDE_CEIL((pad_h-kh),stride_h),0);
			const int phend=min(DIVIDE_CEIL((height+pad_h-kh),stride_h),conved_height);
		
			const int pwstart=max(DIVIDE_CEIL((pad_w-kw),stride_w),0);
			
			const int pwend=min(DIVIDE_CEIL((width+pad_w-kw),stride_w),conved_width);
//			if (index==5) {
//				printf("phstart%d phend%d pwstart%d pwend%d \n",phstart,phend,pwstart,pwend);
//			}
//			
			for(int ph=phstart;ph<phend;ph++){
				for (int pw=pwstart;pw<pwend;pw++){
					const int h=ph*stride_h+kh-pad_h;
					const int w=pw*stride_w+kw-pad_w;
					gradient+=top_diff_slice[ph * conved_width + pw]*bottom_data_slice[h*width+w];
//					if (index==5) {
//						printf("n%d h%d w%d ph%d pw%d topdiff%f bottomdata%f\n",n,h,w,ph,pw,top_diff_slice[ph * conved_width + pw],bottom_data_slice[h*width+w]);
//			//			printf("phstart%d phend%d pwstart%d pwend%d \n",phstart,phend,pwstart,pwend);
//					}
				}
			}
		}
		weight_diff[c * kernel_h * kernel_w+kh*kernel_w+kw]+=gradient;
	}
}

template <typename Dtype>
__global__ void ConvBackwardBias(const int nthreads,
const Dtype* const top_diff,
const int num, const int channels, const int height,
const int width, const int conved_height, const int conved_width,
const int kernel_h, const int kernel_w, const int stride_h,
const int stride_w, const int pad_h, const int pad_w,
Dtype* const bias_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int c = index;
		Dtype gradient=0;
		for( int n=0;n<num;n++) {
			const Dtype* const top_diff_slice =
			top_diff + (n * channels + c) * conved_height * conved_width;
			for(int ph=0;ph<conved_height;ph++) {
				for (int pw=0;pw<conved_width;pw++) {
					gradient+=top_diff_slice[ph * conved_width + pw];
				}
			}
		}
		bias_diff[c]+=gradient;
	}
}
template<typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Backward_gpu(
const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
const vector<Blob<Dtype>*>& bottom) {


	int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
	int* stride_data = this->stride_.mutable_cpu_data();
	int* pad_data = this->pad_.mutable_cpu_data();

	const Dtype* weight = this->blobs_[0]->gpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

	const bool bias_term_ = this->bias_term_;
	Dtype* bias_diff = bias_term_ ? this->blobs_[1]->mutable_gpu_diff() : 0;
	const bool bias_propagate_down_ = this->param_propagate_down_[1];
	const bool weight_propagate_down_ = this->param_propagate_down_[0];


	const int kernel_h_ = kernel_shape_data[0];
	const int kernel_w_ = kernel_shape_data[1];
	const int stride_h_ = stride_data[0];
	const int stride_w_ = stride_data[1];
	const int pad_h_ = pad_data[0];
	const int pad_w_ = pad_data[1];

	const int conved_height = this->output_shape_[0];
	const int conved_weight = this->output_shape_[1];

//	CHECK_EQ(stride_h_, 1)
//	        << "The backward of the net whose stride is bigger than 1 is not implemented now. ";
//	CHECK_EQ(stride_w_, 1)
//	        << "The backward of the net whose stride is bigger than 1 is not implemented now. ";


	for (int i = 0; i < top.size(); ++i) {

		const Dtype* top_diff = top[i]->gpu_diff();
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

		vector<int> shape_ = bottom[i]->shape();
		const int channels_ = shape_[1];
		const int height_ = shape_[2];
		const int width_ = shape_[3];

		// Bias gradient, if necessary.
		if (bias_term_ && bias_propagate_down_) {
			const int count_bias = channels_;
			ConvBackwardBias<Dtype><<<CAFFE_GET_BLOCKS(count_bias), CAFFE_CUDA_NUM_THREADS>>>(
				count_bias, top_diff, bottom[i]->num(), channels_,
				height_, width_,conved_height,conved_weight,kernel_h_,
				kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
				bias_diff);
		}
		// gradient w.r.t. weight. Note that we will accumulate diffs.
		if (weight_propagate_down_) {
			const int count_weight = channels_ * kernel_h_ * kernel_w_;
			ConvBackwardWeight<Dtype><<<CAFFE_GET_BLOCKS(count_weight), CAFFE_CUDA_NUM_THREADS>>>(
					count_weight, top_diff, bottom[i]->num(), channels_,
				height_, width_,conved_height,conved_weight,kernel_h_,
				kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
				weight_diff,
				bottom_data);
		}
		// gradient w.r.t. bottom data, if necessary.
		if (propagate_down[i]) {
			const int count_bottom=bottom[i]->count();
			ConvBackward<Dtype><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
				count_bottom, top_diff, bottom[i]->num(), channels_,
				height_, width_,conved_height,conved_weight,kernel_h_,
				kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, 
				bottom_diff,
				weight);
		}
	}

}

INSTANTIATE_LAYER_GPU_FUNCS (DepthwiseConvolutionLayer);

}  // namespace caffe
