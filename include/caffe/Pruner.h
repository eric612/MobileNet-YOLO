#pragma once
#include<string>
#include<map>
#include<vector>
#include<iostream>
#include <boost/shared_ptr.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"

typedef google::protobuf::int64 int_64;
typedef std::pair<int, double> atom;
typedef atom* Patom;



class Utility{
public:
	Utility() = default;
	~Utility(){};
	inline std::string doubleToString(double num)
	{
		char str[256];
		sprintf(str, "%lf", num);
		std::string result = str;
		return result;
	};
	inline std::string intToString(int_64 i){
		std::stringstream stream;
		stream << i;
		return stream.str();
	};
	inline std::vector<std::string> split(const std::string& str, const std::string& devide){
		std::vector<std::string> res;
		if ("" == str) return res;
		char * strs = new char[str.length() + 1];
		strcpy(strs, str.c_str());
		char * d = new char[devide.length() + 1];
		strcpy(d, devide.c_str());
		char *p = strtok(strs, d);
		while (p) {
			std::string s = p;
			res.push_back(s);
			p = strtok(NULL, d);
		}
		return res;
	};
	void hS(std::vector<atom>* a, int l, int r);
	void fixUp(std::vector<atom>* a, int k);
	void fixDown(std::vector<atom>* a, int k, int N);
};

class Pruner
{
public:
#define _RELU_ "ReLU"
#define _PRELU_ "PReLU"
#define _SIGMOID_ "Sigmoid"
#define _TANH_ "Tanh"
#define _CONVOLUTION_ "Convolution"
#define _POOLING_ "Pooling"

	typedef std::pair<double, int_64> param;
	typedef std::pair<std::string, param> convParam;
	typedef std::vector<convParam> convParams;
	typedef std::pair<convParam, convParams> record;
	typedef std::pair<convParams, convParams> eltwiseRecord;
	typedef record* precord;
	typedef eltwiseRecord* peltwiserecord;
	typedef ::google::protobuf::RepeatedField< double > caffe_double_;
	typedef const ::google::protobuf::RepeatedField< double >& caffe_double_data_;

	Pruner() = default;
	Pruner(const Pruner&);
	Pruner(const std::string xml_path);
	Pruner& operator=(const Pruner&);
	void start(void);
	void read_XML(const std::string xml_path);
	void import(void);
	inline void pruning(void){
		switch (pruningMode){
		case ratio:
			pruningByratio();
			break;
		case size:
			pruningBySize();
			break;
		default:
			break;
		}
	};
	inline bool isNonLinear(std::string layerType){
		return layerType == _RELU_ || layerType == _PRELU_ || layerType == _SIGMOID_ || layerType == _TANH_ ? true : false;
	}
	std::pair<std::vector<std::string>, std::vector<std::string>> eltwiseTravel(const std::string eltwiseName);
	std::vector<std::string>findUpChannels(const std::vector<std::string>* eltwiseLayers, const std::vector<std::string>* splitLayers);
	std::vector<std::string>findUpFilters(const std::vector<std::string>* eltwiseLayers, const std::vector<std::string>* splitLayers);
	std::string findDown(const std::string layerName, std::vector<std::string>* eltwiseLayers, std::vector<std::string>* splitLayers);
	std::string findUp(const std::string layerName, std::vector<std::string>* eltwiseLayers, std::vector<std::string>* splitLayers);
	bool CheckIsEltwiseFilter(const std::string layerName);
	bool CheckIsEltwiseChannel(const std::string layerName);
	void eltwiseCaculate(const peltwiserecord r, std::vector<int>* channelNeedPrune);
	bool checkIsConv(const std::string layerName);
	std::string hasBottom(const std::string layerName);
	std::string hasTop(const std::string layerName);
	void pruningByratio(void);
	void pruningEltwiseByratio(\
		const peltwiserecord r, \
		std::vector<int>* channelNeedPrune); \
		void pruningConvByratio(\
		const precord r, \
		std::vector<int>* channelNeedPrune); \
		void pruningBottomByratio(\
		const precord r, \
		std::vector<int>* channelNeedPrune); \
		int writePrototxt(\
		const std::string prototxt1, \
		const std::string prototxt2); \

		void filterPruning(\
		::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator iter_, \
		std::vector<int>* channelNeedPrune) const; \
		void channelPruning(\
		::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator iter_, \
		std::vector<int>* channelNeedPrune) const; \
		void pruningBySize();
	void writeModel();
	virtual ~Pruner(){};



private:
	std::string xml_Path;
	std::string pruning_caffemodel_path;
	std::string pruning_proto_path;
	std::string pruned_caffemodel_path;
	std::string pruned_proto_path;
	std::string txt_proto_path;

	

	enum ConvCalculateMode
	{
		Norm = 8, L1 = 11, L2 = 12,Variance = 16
	};
	enum PruningMode
	{
		ratio = 0, size = 1
	};
	int convCalculateMode;
	int pruningMode;
	boost::shared_ptr <Utility> utility_;

	std::vector<convParam> pruning_ratio;
	boost::property_tree::ptree configure;
	caffe::NetParameter proto;
	std::vector<record> conv;
	std::vector<eltwiseRecord> eltwiseConv;
	convParams convNeedRewriteOnPrototxt;
	::google::protobuf::RepeatedPtrField< caffe::LayerParameter >* layer;
	mutable ::google::protobuf::RepeatedPtrField< caffe::LayerParameter >::iterator it;
};



