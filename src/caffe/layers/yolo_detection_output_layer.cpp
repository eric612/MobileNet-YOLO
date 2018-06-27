#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/layers/yolo_detection_output_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/bbox_util.hpp"

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
void setNormalizedBBox(NormalizedBBox& bbox, Dtype x, Dtype y, Dtype w, Dtype h)
{
	Dtype xmin = x - w / 2.0;
	Dtype xmax = x + w / 2.0;
	Dtype ymin = y - h / 2.0;
	Dtype ymax = y + h / 2.0;

	if (xmin < 0.0) {
		xmin = 0.0;
	}
	if (xmax > 1.0) {
		xmax = 1.0;
	}
	if (ymin < 0.0) {
		ymin = 0.0;
	}
	if (ymax > 1.0) {
		ymax = 1.0;
	}
	bbox.set_xmin(xmin);
	bbox.set_ymin(ymin);
	bbox.set_xmax(xmax);
	bbox.set_ymax(ymax);
	float bbox_size = BBoxSize(bbox, true);
	bbox.set_size(bbox_size);
}
template <typename Dtype>
void ApplyNms(vector< PredictionResult<Dtype> >& boxes, vector<int>& idxes, Dtype threshold) {
	map<int, int> idx_map;
	for (int i = 0; i < boxes.size() - 1; ++i) {
		if (idx_map.find(i) != idx_map.end()) {
			continue;
		}
		for (int j = i + 1; j < boxes.size(); ++j) {
			if (idx_map.find(j) != idx_map.end()) {
				continue;
			}
			vector<Dtype> Bbox1, Bbox2;
			Bbox1.push_back(boxes[i].x);
			Bbox1.push_back(boxes[i].y);
			Bbox1.push_back(boxes[i].w);
			Bbox1.push_back(boxes[i].h);

			Bbox2.push_back(boxes[j].x);
			Bbox2.push_back(boxes[j].y);
			Bbox2.push_back(boxes[j].w);
			Bbox2.push_back(boxes[j].h);

			Dtype iou = box_iou(Bbox1, Bbox2);
			if (iou >= threshold) {
			idx_map[j] = 1;
			}
			/*	NormalizedBBox Bbox1, Bbox2;
			setNormalizedBBox(Bbox1, boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h);
			setNormalizedBBox(Bbox2, boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h);

			float overlap = JaccardOverlap(Bbox1, Bbox2, true);

			if (overlap >= threshold) {
				idx_map[j] = 1;
			}*/
		}
	}
	for (int i = 0; i < boxes.size(); ++i) {
		if (idx_map.find(i) == idx_map.end()) {
			idxes.push_back(i);
		}
	}
}
template <typename Dtype>
void class_index_and_score(Dtype* input, int classes, PredictionResult<Dtype>& predict)
{
	Dtype sum = 0;
	Dtype large = input[0];
	int classIndex = 0;
	for (int i = 0; i < classes; ++i) {
		if (input[i] > large)
			large = input[i];
	}
	for (int i = 0; i < classes; ++i) {
		Dtype e = exp(input[i] - large);
		sum += e;
		input[i] = e;
	}

	for (int i = 0; i < classes; ++i) {
		input[i] = input[i] / sum;
	}
	large = input[0];
	classIndex = 0;

	for (int i = 0; i < classes; ++i) {
		if (input[i] > large) {
			large = input[i];
			classIndex = i;
		}
	}
	predict.classType = classIndex;
	predict.classScore = large;
}
template <typename Dtype>
void get_region_box(Dtype* x, PredictionResult<Dtype>& predict, vector<Dtype> biases, int n, int index, int i, int j, int w, int h) {
	predict.x = (i + sigmoid(x[index + 0])) / w;
	predict.y = (j + sigmoid(x[index + 1])) / h;
	predict.w = exp(x[index + 2]) * biases[2 * n] / w;
	predict.h = exp(x[index + 3]) * biases[2 * n + 1] / h;
}
template <typename Dtype>
void YoloDetectionOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const YoloDetectionOutputParameter& yolo_detection_output_param =
      this->layer_param_.yolo_detection_output_param();
  CHECK(yolo_detection_output_param.has_num_classes()) << "Must specify num_classes";
  side_ = yolo_detection_output_param.side();
  side_ = bottom[0]->width();
  num_classes_ = yolo_detection_output_param.num_classes();
  num_box_ = yolo_detection_output_param.num_box();
  coords_ = yolo_detection_output_param.coords();
  confidence_threshold_ = yolo_detection_output_param.confidence_threshold();
  nms_threshold_ = yolo_detection_output_param.nms_threshold();

  for (int c = 0; c < yolo_detection_output_param.biases_size(); ++c) {
     biases_.push_back(yolo_detection_output_param.biases(c));
  } //0.73 0.87;2.42 2.65;4.30 7.04;10.24 4.59;12.68 11.87;

  if (yolo_detection_output_param.has_label_map_file())
  {
    string label_map_file = yolo_detection_output_param.label_map_file();
    if (label_map_file.empty()) 
    {
      // Ignore saving if there is no label_map_file provided.
      LOG(WARNING) << "Provide label_map_file if output results to files.";
    } 
    else 
    {
      LabelMap label_map;
      CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
          << "Failed to read label map file: " << label_map_file;
      CHECK(MapLabelToName(label_map, true, &label_to_name_))
          << "Failed to convert label to name.";
    }
  }
}

template <typename Dtype>
void YoloDetectionOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->num(), 1);
  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  // Since the number of bboxes to be kept is unknown before nms, we manually
  // set it to (fake) 1.
  top_shape.push_back(1);
  // Each row is a 7 dimension vector, which stores
  // [image_id, label, confidence, x, y, w, h]
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);
}
template <typename Dtype>
bool BoxSortDecendScore(const PredictionResult<Dtype>& box1, const PredictionResult<Dtype>& box2) {
	return box1.confidence> box2.confidence;
}
template <typename Dtype>
void YoloDetectionOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->num();
	side_ = bottom[0]->width();
	Blob<Dtype> swap;
	swap.Reshape(bottom[0]->num(), bottom[0]->height()*bottom[0]->width(), num_box_, bottom[0]->channels() / num_box_);
	//std::cout<<"4"<<std::endl;  
	Dtype* swap_data = swap.mutable_cpu_data();
	int index = 0;
	for (int b = 0; b < bottom[0]->num(); ++b) {
		for (int h = 0; h < bottom[0]->height(); ++h) {
			for (int w = 0; w < bottom[0]->width(); ++w) {
				for (int c = 0; c < bottom[0]->channels(); ++c)
				{
					swap_data[index++] = bottom[0]->data_at(b, c, h, w);
				}
			}
		}
	}
    
    //CHECK_EQ(bottom[0]->data_at(0,4,1,2),swap.data_at(0,15,0,4));
    //std::cout<<"5"<<std::endl;
    //*********************************************************Activation********************************************************//
    //disp(swap);
	vector< PredictionResult<Dtype> > predicts;
	PredictionResult<Dtype> predict;
	predicts.clear(); 
	for (int b = 0; b < swap.num(); ++b) {
		for (int j = 0; j < side_; ++j) {
			for (int i = 0; i < side_; ++i) {
				for (int n = 0; n < num_box_; ++n){
					int index = b * swap.channels() * swap.height() * swap.width() + (j * side_ + i) * swap.height() * swap.width() + n * swap.width();
					CHECK_EQ(swap_data[index], swap.data_at(b, j * side_ + i, n, 0));
					get_region_box(swap_data, predict, biases_, n, index, i, j, side_, side_);
					predict.objScore = sigmoid(swap_data[index + 4]);
					class_index_and_score(swap_data + index + 5, num_classes_, predict);
					predict.confidence = predict.objScore *predict.classScore;
					if (predict.confidence >= confidence_threshold_)
					{
						predicts.push_back(predict);
					}
				}
			}
		}
	}
	std::sort(predicts.begin(), predicts.end(), BoxSortDecendScore<Dtype>);
    vector<int> idxes;
    int num_kept = 0;
    if(predicts.size() > 0){
      ApplyNms(predicts, idxes, nms_threshold_);
      num_kept = idxes.size();
    }
    vector<int> top_shape(2, 1);
    top_shape.push_back(num_kept);
    top_shape.push_back(7);

    Dtype* top_data;
  
  if (num_kept == 0) {
    DLOG(INFO) << "Couldn't find any detections";
    top_shape[2] = swap.num();
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
    caffe_set<Dtype>(top[0]->count(), -1, top_data);
    // Generate fake results per image.
    for (int i = 0; i < num; ++i) {
      top_data[0] = i;
      top_data += 7;
    }
  } 
  else {
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
    for (int i = 0; i < num_kept; i++){
      top_data[i*7] = 0;                              //Image_Id
      top_data[i*7+1] = predicts[idxes[i]].classType + 1; //label
      top_data[i*7+2] = predicts[idxes[i]].confidence; //confidence
	  float left = (predicts[idxes[i]].x - predicts[idxes[i]].w / 2.);
	  float right = (predicts[idxes[i]].x + predicts[idxes[i]].w / 2.);
	  float top = (predicts[idxes[i]].y - predicts[idxes[i]].h / 2.);
	  float bot = (predicts[idxes[i]].y + predicts[idxes[i]].h / 2.);

      top_data[i*7+3] = left;
      top_data[i*7+4] = top;
      top_data[i*7+5] = right;
      top_data[i*7+6] = bot;
	  DLOG(INFO) << "Detection box"  << "," << predicts[idxes[i]].classType << "," << predicts[idxes[i]].x << "," << predicts[idxes[i]].y << "," << predicts[idxes[i]].w << "," << predicts[idxes[i]].h;
    }

  }

}

#ifdef CPU_ONLY
//STUB_GPU_FORWARD(DetectionOutputLayer, Forward);
#endif

INSTANTIATE_CLASS(YoloDetectionOutputLayer);
REGISTER_LAYER_CLASS(YoloDetectionOutput);

}  // namespace caffe
