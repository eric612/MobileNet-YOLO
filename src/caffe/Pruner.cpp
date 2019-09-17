#include "caffe/Pruner.h"
#include "caffe/util/io.hpp"
#include <cmath>
#include <fstream>
#include "caffe/util/math_functions.hpp"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
using namespace caffe;
using namespace std;


Pruner::Pruner(const Pruner &p) :
xml_Path(p.xml_Path)
{	}

Pruner& Pruner::operator=(const Pruner &rhs){
	xml_Path = rhs.xml_Path;
	return *this;
}

Pruner::Pruner(const string xml_path){
	xml_Path = xml_path;
	utility_ = boost::shared_ptr <Utility>(new Utility());
}


void Pruner::start(){
	read_XML(xml_Path);
	import();
	pruning();
	writePrototxt(pruning_proto_path, pruned_proto_path);
	writeModel();

}

void Pruner::read_XML(const string xml_path){
	read_xml(xml_path, configure);
	pruning_caffemodel_path = configure.get<string>("caffemodelpath");
	pruning_proto_path = configure.get<string>("protopath");
	pruned_caffemodel_path = configure.get<string>("prunedcaffemodelpath");
	pruned_proto_path = configure.get<string>("prunedprotopath");
	txt_proto_path = configure.get<string>("txtprotopath");
	pruningMode = atoi(configure.get<string>("PruningMode.mode").c_str());
	convCalculateMode = atoi(configure.get<string>("ConvCalculateMode.mode").c_str());

	ReadProtoFromBinaryFile(pruning_caffemodel_path, &proto);
	layer = proto.mutable_layer();
	//importing vanilla convolution layers'parameters from xml
	boost::property_tree::ptree layers = configure.get_child("filterpruning");
	for (auto it1 = layers.begin(); it1 != layers.end(); it1++){
		auto clayers = it1->second;
		if (clayers.empty()){
			continue;
		}
		string name = clayers.get<string>("<xmlattr>.name");
		if (!checkIsConv(name)){
			cout << "input incorrect: " + name + " is not a convolution layer , ignoring..." << endl;
			sleep(1000);
			continue;
		}
		double ratio = atof(clayers.get<string>("<xmlattr>.cut").c_str());
		for (it = layer->begin(); it != layer->end(); it++){
			if (name == it->name()){
				pruning_ratio.push_back(convParam(name, param(ratio, it->blobs(0).shape().dim(0))));
				convNeedRewriteOnPrototxt.push_back(convParam(it->name(), param(ratio, it->blobs(0).shape().dim(0))));
				break;
			}
		}
	}

	//Read Eltwise Prop from XML file
	layers = configure.get_child("eltwise");
	for (auto it1 = layers.begin(); it1 != layers.end(); it1++){
		convParams conv_channels_temps;
		convParams conv_filters_temps;
		auto clayers = it1->second;
		if (clayers.empty()){
			continue;
		}
		string name = clayers.get<string>("<xmlattr>.name");
		double ratio = atof(clayers.get<string>("<xmlattr>.cut").c_str());

		//Getting eltwise layers 'parameters
		//Getting channels'paramters
		pair<vector<string>, vector<string>> pair = eltwiseTravel(name);
		vector<string> channels = pair.first;
		vector<string> filters = pair.second;
		for (auto it2 = channels.begin(); it2 != channels.end(); it2++){
			convParam conv_temp;
			for (it = layer->begin(); it != layer->end(); it++){
				if (*it2 == it->name()){
					conv_temp.first = *it2;
					conv_temp.second.first = ratio;
					conv_temp.second.second = it->blobs(0).shape().dim(0);
					conv_channels_temps.push_back(conv_temp);
					break;
				}
			}
		}
		//Getting filters' parameters 
		for (auto it3 = filters.begin(); it3 != filters.end(); it3++){
			convParam conv_temp;
			for (it = layer->begin(); it != layer->end(); it++){
				if (*it3 == it->name()){
					conv_temp.first = *it3;
					conv_temp.second.first = ratio;
          //cout << ratio << "----" << endl;
					conv_temp.second.second = it->blobs(0).shape().dim(0);
					conv_filters_temps.push_back(conv_temp);
					convNeedRewriteOnPrototxt.push_back(convParam(it->name(), param(ratio, it->blobs(0).shape().dim(0))));
					break;
				}
			}
		}
		eltwiseConv.push_back(make_pair(conv_channels_temps, conv_filters_temps));
	}
}

void Pruner::import(){
	/*for (it = layer->begin(); it->name() != "conv1_bn"; ++it);
	it->blobs(0).num();
	string x1 = it->name();
	int x = it->blobs(0).shape().dim(0);*/

	auto iter1 = pruning_ratio.begin();
	while (iter1 != pruning_ratio.end()){
		it = layer->begin();
		double ratio = iter1->second.first;
		convParams b1;
		string prunedConvName = iter1->first;
		string poolName = "konglusen";
		for (; it != layer->end(); it++){
			string n = it->name();
			if (it->bottom_size() != 0){
				for (int i = 0; i < it->bottom_size(); i++){
					if (prunedConvName == it->bottom(i)){
            
						if (it->type() == "Convolution"){
							if (prunedConvName == it->bottom(0)){
								b1.push_back(convParam(it->name(), param(ratio, it->blobs(0).shape().dim(0))));
								break;
							}
						}
						else if (it->type() == "ConvolutionDepthwise"|| it->type() == "DepthwiseConvolution"){
              cout<<it->name()<<"\n";
							if (prunedConvName == it->bottom(0)){
								b1.push_back(convParam(it->name(), param(ratio, it->blobs(0).shape().dim(0))));
								convNeedRewriteOnPrototxt.push_back(convParam(it->name(), param(ratio, it->blobs(0).shape().dim(0))));
								break;
							}
						}

						else if (it->type() == "Pooling"){
							it++;
							vector<string> top_names_;
							if (it->type() == "Split"){
								for (size_t i = 0; i < it->top_size(); i++){
									top_names_.push_back(it->top(i));
								}
							}
							else{
								top_names_.push_back(it->top(i));
							}

							for (auto it1 = it; it1 != layer->end(); it1++){
								if (it1->type() == "Convolution" || it1->type() == "ConvolutionDepthwise"|| it->type() == "DepthwiseConvolution"){
									if (find(top_names_.begin(), top_names_.end(), it1->bottom(0)) != top_names_.end()){
										b1.push_back(convParam(it1->name(), param(ratio, it1->blobs(0).shape().dim(0))));
									}
								}
							}
						}
					}
				}
			}
			else{
				continue;
			}

		}
		conv.push_back(record(*iter1, b1));
		iter1++;
	}
}

void Pruner::pruningByratio(){
	for (string::size_type i = 0; i < conv.size(); i++){
		vector<int> channelNeedPrune;
		pruningConvByratio(&conv.at(i), &channelNeedPrune);
		pruningBottomByratio(&conv.at(i), &channelNeedPrune);
	}
	for (string::size_type i = 0; i < eltwiseConv.size(); i++){
		vector<int> channelNeedPrune;
		eltwiseCaculate(&eltwiseConv.at(i), &channelNeedPrune);
		pruningEltwiseByratio(&eltwiseConv.at(i), &channelNeedPrune);
	}
}

void Pruner::pruningBySize(){

}

void Utility::hS(vector<atom>* a, int l, int r){
	int k;
	int N = r - l + 1;
	for (k = N / 2; k >= 1; k--){
		fixDown(a, k, N);
	}
	while (N > 1){
		swap(a->at(1), a->at(N));
		fixDown(a, 1, --N);
	}
}

void Utility::fixUp(vector<atom>* a, int k){
	while (k > 1 && ([a, k]() -> bool {return (a->at(k / 2).second) < (a->at(k).second); })()){
		swap(a->at(k), a->at(k / 2));
		k = k / 2;
	}
}

void Utility::fixDown(vector<atom>* a, int k, int N){
	int j;
	while (2 * k <= N){
		j = 2 * k;
		if (j < N && ([a, j]() -> bool {return (a->at(j).second) < (a->at(j + 1).second); })()){
			j++;
		}
		if (([a, j, k]() -> bool {return (a->at(k).second) > (a->at(j).second); })()){
			break;
		}
		swap(a->at(k), a->at(j));
		k = j;
	}
}

void Pruner::writeModel(){
	WriteProtoToTextFile(proto, txt_proto_path);
	WriteProtoToBinaryFile(proto, pruned_caffemodel_path);
}

bool Pruner::checkIsConv(const string layerName){
	int count = 0;
	for (auto it1 = layer->begin(); it1 != layer->end(); it1++){
		if (it1->name() == layerName)
      cout<<it1->type();
			if (it1->type() == "Convolution"){
        
				it1++;
        
				if (it1->type() == "BatchNorm"){
					it1++;
					it1++;
					if (it1->type() == "Split" || it1->type() == "Eltwise")
					{
						break;
					}
					else if (isNonLinear(it1->type())){
						it1++;
						if (it1->type() == "Split" || it1->type() == "Eltwise")
						{
							break;
						}
						else{
							count++;
							break;
						}
					}
					else
					{
						count++;
						break;
					}
				}
				else if (it1->type() == "Split" || it1->type() == "Eltwise"){

					break;
				}
				else{
					count++;
					break;
				}


			}

	}
	return (count == 1) ? true : false;
}

void Pruner::pruningConvByratio(const precord r, vector<int>* pchannelNeedPrune){

	for (it = layer->begin(); it != layer->end(); it++){
		if (r->first.first == it->name()){
			vector<atom> convlayervalue;
			convlayervalue.push_back(make_pair(-1, 1));
			int num = it->blobs(0).shape().dim(0);
			int channels = it->blobs(0).shape().dim(1);
			int height = it->blobs(0).shape().dim(2);
			int width = it->blobs(0).shape().dim(3);
			int spatial_dim = channels * width * height;
			int data_count = num * spatial_dim;
			int cutNum = (r->first.second.second)*(r->first.second.first);

			// oblas calculate
			Blob<double> mean_, variance_, temp_;
			Blob<double> spatial_sum_multiplier_, filter_data_;
			Blob<double> num_temp_, num_temp_1, num_temp_2;
			//BlobProto

			vector<int> sz;
			sz.push_back(spatial_dim);
			spatial_sum_multiplier_.Reshape(sz);

			double* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
			caffe_set(spatial_sum_multiplier_.count(), double(1), multiplier_data);
			//Blob<float> filter_data_ = 

			//Modifying the kernel heap by traveling through the computed-average-kernel's -size then sort 
			BlobProto blobData = it->blobs(0);
			BlobProto blobData1;
			double maxData = 0.0;
			int k = blobData.data_size();
			filter_data_.FromProto(blobData, true);
			sz[0] = num;
			mean_.Reshape(sz);
			temp_.Reshape(sz);
			variance_.Reshape(sz);
			sz[0] = num*spatial_dim;
			num_temp_.Reshape(sz);
			num_temp_1.FromProto(blobData, true);
			num_temp_2.Reshape(sz);
			switch (convCalculateMode)
			{
			case Pruner::Variance:
				caffe_cpu_gemv<double>(CblasNoTrans, num, spatial_dim, 1. / spatial_dim, filter_data_.cpu_data(),
					spatial_sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
				caffe_cpu_gemm<double>(CblasNoTrans, CblasNoTrans, spatial_dim, num, 1, -1,
					spatial_sum_multiplier_.cpu_data(), mean_.cpu_data(), 1, num_temp_1.mutable_cpu_data());
				caffe_powx(num_temp_1.count(), num_temp_1.cpu_data(), double(2), num_temp_2.mutable_cpu_data());
				caffe_cpu_gemv<double>(CblasNoTrans, num, spatial_dim, 1. / spatial_dim, num_temp_2.cpu_data(),
					spatial_sum_multiplier_.cpu_data(), 0., variance_.mutable_cpu_data());
				variance_.ToProto(&blobData1, false);
				for (size_t i = 0; i < variance_.count(); i++){
					atom a = make_pair(i, blobData1.double_data(i));
					convlayervalue.push_back(a);
				}
				break;
			case Pruner::Norm:
				for (size_t i = 0; i < blobData.data_size(); i++){
					if (maxData < abs(blobData.data(i))){
						maxData = abs(blobData.data(i));
					}
				}
				for (int i = 0; i < num; i++){
					double value = 0.0;
					for (int j = 0; j < spatial_dim; j++){
						value += abs(blobData.data(i*spatial_dim + j)) / maxData;
					}
					atom a = make_pair(i, value / spatial_dim);
					convlayervalue.push_back(a);
				}

				break;
			case Pruner::L1:
				caffe_abs(filter_data_.count(), filter_data_.cpu_data(), filter_data_.mutable_cpu_data());
				caffe_cpu_gemv<double>(CblasNoTrans, num, spatial_dim, 1. / spatial_dim, filter_data_.cpu_data(),
					spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
				temp_.ToProto(&blobData1, false);
				for (size_t i = 0; i < temp_.count(); i++){
					atom a = make_pair(i, blobData1.double_data(i));
					convlayervalue.push_back(a);
				}
				break;
			case Pruner::L2:
				caffe_powx(filter_data_.count(), filter_data_.cpu_data(), double(2), filter_data_.mutable_cpu_data());
				caffe_cpu_gemv<double>(CblasNoTrans, num, spatial_dim, 1. / spatial_dim, filter_data_.cpu_data(),
					spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
				temp_.ToProto(&blobData1, false);
				for (size_t i = 0; i < temp_.count(); i++){
					atom a = make_pair(i, blobData1.double_data(i));
					convlayervalue.push_back(a);
				}
				break;
			default:
				break;
			}
			utility_->hS(&convlayervalue, 1, num);
			for (int i = 0; i < cutNum; i++){
				pchannelNeedPrune->push_back(convlayervalue.at(i + 1).first);
			}
			//start prune
			this->filterPruning(it, pchannelNeedPrune);

		}
	}
}

void Pruner::pruningBottomByratio(const precord r, vector<int>* pchannelNeedPrune){
	//preform pruning on next layer 
	int num = r->first.second.second;
	int cutNum = (r->first.second.second)*(r->first.second.first);
	string::size_type i1 = r->second.size();
	for (string::size_type k = 0; k < r->second.size(); k++){
		convParam conv1 = r->second[k];
		string n = conv1.first;
		for (it = layer->begin(); it != layer->end(); it++){
			if (it->name() == conv1.first){
				if (it->type() == "Convolution"){
					this->channelPruning(it, pchannelNeedPrune);
					break;
				}
				else if (it->type() == "ConvolutionDepthwise" || it->type() == "DepthwiseConvolution"){
          cout<<it->name()<<"\n";
					this->filterPruning(it, pchannelNeedPrune);
					it++;

					while (it->type() != "Convolution"){
						it++;
					}

					//start prune pointwise conv layer which subsequent to depthwiseConv
					string name1 = it->name();
					if (it->type() == "Convolution"){
						this->channelPruning(it, pchannelNeedPrune);
					}

					break;
				}
			}

		}

	}
}

void Pruner::filterPruning(::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator iter_, vector<int>* pchannelNeedPrune) const{
	int_64 filter_count = iter_->blobs(0).shape().dim(0);
	int_64 channels = iter_->blobs(0).shape().dim(1);
	int_64 height = iter_->blobs(0).shape().dim(2);
	int_64 width = iter_->blobs(0).shape().dim(3);
	int count = channels * width * height;
	int cutNum = pchannelNeedPrune->size();
	BlobProto *blob_ = iter_->mutable_blobs(0);
	BlobProto blob = iter_->blobs(0);
	blob_->clear_data();
	vector<int>::const_iterator beg = pchannelNeedPrune->cbegin();
	vector<int>::const_iterator end = pchannelNeedPrune->cend();
	for (int j = 0; j < filter_count; j++){
		if (find(beg, end, j) == pchannelNeedPrune->cend()){
			for (int g = 0; g < count; g++){
				blob_->add_data(blob.data(j*count + g));
			}
		}
	}
	BlobShape shape;
	shape.add_dim(filter_count - cutNum);
	shape.add_dim(channels);
	shape.add_dim(height);
	shape.add_dim(width);
	blob_->mutable_shape()->CopyFrom(shape);

	// We will perform bias update based if the bias of conv existed.
	if (iter_->blobs_size() > 1){
		BlobProto *blob_ = iter_->mutable_blobs(1);
		BlobProto blob = iter_->blobs(1);
		blob_->clear_data();
		for (int j = 0; j < filter_count; j++){
			if (find(beg, end, j) == pchannelNeedPrune->cend()){
				blob_->add_data(blob.data(j));
			}
		}
		BlobShape shape;
		shape.add_dim(filter_count - cutNum);
		blob_->mutable_shape()->CopyFrom(shape);
	}

	iter_->mutable_convolution_param()->set_num_output(filter_count - cutNum);

	if ((++iter_)->type() == "BatchNorm"){
		int_64 bn_count = iter_->blobs(0).shape().dim(0);
		BlobProto *bnBlob0_ = iter_->mutable_blobs(0);
		BlobProto bnBlob0 = iter_->blobs(0);
		bnBlob0_->clear_data();
		for (int j = 0; j < bn_count; j++){
			if (find(beg, end, j) == pchannelNeedPrune->cend()){
				bnBlob0_->add_data(bnBlob0.data(j));
			}
		}
		BlobShape shape0;
		shape0.add_dim(bn_count - cutNum);
		bnBlob0_->mutable_shape()->CopyFrom(shape0);

		BlobProto *bnBlob1_ = iter_->mutable_blobs(1);
		BlobProto bnBlob1 = iter_->blobs(1);
		bnBlob1_->clear_data();
		for (int j = 0; j < bn_count; j++){
			if (find(beg, end, j) == pchannelNeedPrune->cend()){
				bnBlob1_->add_data(bnBlob1.data(j));
			}
		}
		BlobShape shape1;
		shape1.add_dim(bn_count - cutNum);
		bnBlob1_->mutable_shape()->CopyFrom(shape1);
		iter_++;

		BlobProto *sBlob0_ = iter_->mutable_blobs(0);
		BlobProto sBlob0 = iter_->blobs(0);
		sBlob0_->clear_data();
		for (int j = 0; j < bn_count; j++){
			if (find(beg, end, j) == pchannelNeedPrune->cend()){
				sBlob0_->add_data(sBlob0.data(j));
			}
		}
		BlobShape shape2;
		shape2.add_dim(bn_count - cutNum);
		sBlob0_->mutable_shape()->CopyFrom(shape2);

		BlobProto *sBlob1_ = iter_->mutable_blobs(1);
		BlobProto sBlob1 = iter_->blobs(1);
		sBlob1_->clear_data();
		for (int j = 0; j < bn_count; j++){
			if (find(beg, end, j) == pchannelNeedPrune->cend()){
				sBlob1_->add_data(sBlob1.data(j));
			}
		}
		BlobShape shape3;
		shape3.add_dim(bn_count - cutNum);
		sBlob1_->mutable_shape()->CopyFrom(shape3);
	}
}

void Pruner::channelPruning(::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator iter_, vector<int>* pchannelNeedPrune) const{

	int_64 nextLayKerNum = iter_->blobs(0).shape().dim(0);
	int_64 nextLayChannel = iter_->blobs(0).shape().dim(1);
	int_64 nextLayKerH = iter_->blobs(0).shape().dim(2);
	int_64 nextLayKerW = iter_->blobs(0).shape().dim(3);
	int cutNum = pchannelNeedPrune->size();
	int counts = nextLayChannel * nextLayKerH * nextLayKerW;
	int dimSize = nextLayKerH * nextLayKerW;
	BlobProto *blob1_ = iter_->mutable_blobs(0);
	BlobProto blob1 = iter_->blobs(0);
	blob1_->clear_data();
	vector<int>::const_iterator beg = pchannelNeedPrune->cbegin();
	vector<int>::const_iterator end = pchannelNeedPrune->cend();
	for (int j = 0; j < nextLayKerNum; j++){
		for (int g = 0; g < nextLayChannel; g++){
			if (find(beg, end, g) == pchannelNeedPrune->cend()){
				for (int m = 0; m < dimSize; m++){
					blob1_->add_data(blob1.data(j * counts + g * dimSize + m));
				}
			}
		}
	}
	BlobShape shape1;
	shape1.add_dim(nextLayKerNum);
	shape1.add_dim(nextLayChannel - cutNum);
	shape1.add_dim(nextLayKerH);
	shape1.add_dim(nextLayKerW);
	blob1_->mutable_shape()->CopyFrom(shape1);

}

int  Pruner::writePrototxt(const string prototxt1, const string prototxt2){
	fstream fin_in(pruning_proto_path, ios::in | ios::binary);
	fstream fin_out(pruned_proto_path, ios::out | ios::binary);
	if (!fin_in || !fin_out)
		return 0;
	string str1 = "name";
	string str2 = "num_output";
	string str3 = "type";
	string str;
	string nametemp;
	bool final_flag = false;
	bool nor_flag = false;
	int prunedNum;
	while (getline(fin_in, str)){
		if (str.find("prob") != -1){
			final_flag = true;
		}
		if (final_flag == true){
			fin_out << str << '\n';
			continue;
		}
		int index = -1;
		if (str.find(str1) != -1){
			for (auto& r : convNeedRewriteOnPrototxt){
				string s = '"' + r.first + '"';
				index = str.find(s);
				if (index != -1){
					int num = r.second.second;
					int cut = r.second.first*r.second.second;
					prunedNum = num - cut;
					nor_flag = true;
					break;
				}
			}
		}
		if (str.find(str2) != -1){
			if (!nor_flag){
				fin_out << str << '\n';
			}
			else{
				fin_out << "    num_output: " + to_string(prunedNum) << '\n';
				nor_flag = false;
			}
		}
		else{
			fin_out << str << '\n';
		}
	}
	return 1;
}

void Pruner::eltwiseCaculate(const peltwiserecord r, vector<int>* channelNeedPrune){

	unsigned blob_size = r->second.at(0).second.second;
	unsigned cutNum = (r->second.at(0).second.second) * (r->second.at(0).second.first);
	vector<atom> convlayervalue;
	convlayervalue.push_back(make_pair(-1, 1));
	double *p_arr = new double[blob_size];
	for (int i = 0; i < blob_size; i++){
		p_arr[i] = 0;
	}
	for (string::size_type i = 0; i < r->second.size(); i++){
		for (it = layer->begin(); it != layer->end(); it++){
			int_64 num, channels, height, width;
			int count, cutNum;
			if (it->name() == r->second.at(i).first){
				num = it->blobs(0).shape().dim(0);
				channels = it->blobs(0).shape().dim(1);
				height = it->blobs(0).shape().dim(2);
				width = it->blobs(0).shape().dim(3);
				count = channels * width * height;
				cutNum = r->second.at(0).second.first * r->second.at(0).second.second;
				BlobProto blobData = it->blobs(0);
				for (int_64 j = 0; j < num; j++){
					double value = 0.0;
					for (int k = 0; k < count; k++){
						value += abs(blobData.data(j*count + k));
					}
					p_arr[i] = p_arr[i] + value / count;
				}
			}
		}
	}
	for (int i = 0; i < blob_size; i++){
		atom a = make_pair(i, p_arr[i]);
		convlayervalue.push_back(a);
	}
	utility_->hS(&convlayervalue, 1, blob_size);
	for (int i = 0; i < cutNum; i++){
		channelNeedPrune->push_back(convlayervalue.at(i + 1).first);
	}
}

void Pruner::pruningEltwiseByratio(const peltwiserecord r, vector<int>* channelNeedPrune){
	for (auto iter = r->second.begin(); iter != r->second.end(); iter++){
		for (it = layer->begin(); it != layer->end(); it++){
			if (it->name() == iter->first){
				this->filterPruning(it, channelNeedPrune);
			}
		}
	}
	for (auto iter = r->first.begin(); iter != r->first.end(); iter++){
		for (it = layer->begin(); it != layer->end(); it++){
			if (it->name() == iter->first){
				this->channelPruning(it, channelNeedPrune);
				break;
			}
		}
	}
}

pair<vector<string>, vector<string>> Pruner::eltwiseTravel(const string eltwiseName){
	//Check 
	auto it = layer->begin();
	for (; it != layer->end(); it++){
		if (it->name() == eltwiseName && it->type() == "Eltwise"){
			break;
		}
		else if (it->name() == eltwiseName && (it->type() != "Eltwise" && it->type() != "Split"))
		{
			cout << eltwiseName << " is not an eltwise or split layer" <<endl;
			system("pause");
		}
	}
	pair<vector<string>, vector<string>> p;
	vector<string> eltwiseLayers = { eltwiseName };
	vector<string> splitLayers;
	vector<string> filters;
	vector<string> channels;
	this->findDown(eltwiseName, &eltwiseLayers, &splitLayers);
	string conv_temp = this->findUp(eltwiseName, &eltwiseLayers, &splitLayers);
	filters = this->findUpFilters(&eltwiseLayers, &splitLayers);
	channels = this->findUpChannels(&eltwiseLayers, &splitLayers);
	if ("stop" != conv_temp){
		channels.push_back(conv_temp);
	}
	p.first = channels;
	p.second = filters;
	return p;
}

vector<string> Pruner::findUpChannels(const vector<string>* eltwiseLayers, const vector<string>* splitLayers){
	vector<string> Channels;
	for (string::size_type k = 0; k < splitLayers->size(); k++){
		auto it = layer->begin();
		while (it->name() != splitLayers->at(k))it++;
		if (it->top_size() != 0){
      
			for (int i = 0; i < it->top_size(); i++){
				string temp = hasBottom(it->top(i));
        
				if (temp != ""){
					Channels.push_back(temp);
          
				}
			}
		}
	}
	return Channels;
}

vector<string> Pruner::findUpFilters(const vector<string>* eltwiseLayers, const vector<string>* splitLayers){
	vector<string> Filters;
	for (string::size_type k = 0; k < eltwiseLayers->size(); k++){
		auto it = layer->begin();
    
		while (it->name() != eltwiseLayers->at(k))it++;
		if (it->bottom_size() != 0){
      if ( it->type() == "Split" ) 
        continue;
			for (int i = 0; i < it->bottom_size(); i++){
				if (it->bottom(i).find("split") != string::npos){
					continue;
				}
				Filters.push_back(it->bottom(i));
			}
		}
	}
	for (string::size_type k = 0; k < splitLayers->size(); k++){
		auto it = layer->begin();
		while (it->name() != splitLayers->at(k))it++;
		it--;
		if (it->type() == "Eltwise" || it->type() == "Pooling"){
			continue;
		}
		else if (isNonLinear(it->type())){
			it--;
			if (it->type() == "Eltwise"){
				continue;
			}
		}
		while (it->type() != "Convolution")
		{
			it--;
		}
		Filters.push_back(it->name());
	}
	return Filters;
}

string Pruner::findDown(const string layerName, vector<string>* eltwiseLayers, vector<string>* splitLayers){
	auto it = layer->begin();
	for (; it != layer->end(); it++){
		if (it->name() == layerName){
			string n = it->name();
			break;
		}
	}
	if (it->bottom_size() != 0){
		for (int i = 0; i < it->bottom_size(); i++){
			if (it->bottom(i).find("split") != string::npos){
				string splitLayerTopName = it->bottom(i);
				string splitLayerName = splitLayerTopName.substr(0, splitLayerTopName.find_last_of("_"));
				splitLayers->push_back(splitLayerName);
				it--;
				while (it->name() != splitLayerName){
					it--;
				}
				it--;
				if (isNonLinear(it->type())){
					it--;
					if (it->type() == "Eltwise"){
						eltwiseLayers->push_back(it->name());
						return this->findDown(it->name(), eltwiseLayers, splitLayers);
					}
				}
				else if (it->type() == "Eltwise"){
					eltwiseLayers->push_back(it->name());
					return this->findDown(it->name(), eltwiseLayers, splitLayers);
				}
				else
				{
					break;
				}
			}
		}
	}
	return "dytto";
}

string Pruner::findUp(const string layerName, vector<string>* eltwiseLayers, vector<string>* splitLayers){
	auto it = layer->begin();
	for (; it != layer->end(); it++){
		if (it->name() == layerName){
			break;
		}
	}
	while (it->type() != "Split" && it->type() != "Convolution" && it->type() != "Pooling"){
		it++;
	}
	if (it->type() == "Pooling"){
		it++;
		if (it->type() == "Convolution"){
			return it->name();
		}
		else if (it->type() == "Split")
		{
			splitLayers->push_back(it->name());
			return "stop";
		}
	}
	else if (it->type() == "Split"){
    //cout <<it->name()<<"--------------\n";
		splitLayers->push_back(it->name());
		return "stop";
	}
	else if (it->type() == "Convolution"){
    
		return it->name();
	}
}

string Pruner::hasBottom(const string layerName){
	auto it = layer->begin();
	for (; it != layer->end(); it++){
		if (it->type() == "Convolution"){
			if (it->bottom_size() > 0){
				for (int k = 0; k < it->bottom_size(); k++){
					if (it->bottom(k) == layerName){
						return it->name();
					}
				}
			}
		}
	}
	return "";
}

string Pruner::hasTop(const string layerName){
	auto it = layer->begin();
	for (; it != layer->end(); it++){
		if (it->type() == "Convolution"){
			if (it->top_size() > 0){
				for (int k = 0; k < it->top_size(); k++){
					if (it->top(k) == layerName){
						return it->name();
					}
				}
			}
		}
	}
	return "";
}
