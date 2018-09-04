// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "caffe/util/benchmark.hpp"
//#define custom_class
#ifdef custom_class
//char* CLASSES[6] = { "__background__",
//"bicyle", "car", "motorbike", "person","cones"
//};
char* CLASSES[5] = { "__background__",
"big car","car", "motorbike","person"
};
#else
char* CLASSES[21] = { "__background__",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };
#endif
#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value,
		   const float confidence_threshold,
		   const float normalize_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);
  void Preprocess(const cv::Mat& img,
	  std::vector<cv::Mat>* input_channels,double normalize_value);
 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  float nor_val = 1.0;
};

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value,
				   const float confidence_threshold,
				   const float normalize_value) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif
  //Caffe::set_mode(Caffe::CPU);
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
  nor_val = normalize_value;
}
float sec(clock_t clocks)
{
	return (float)clocks / CLOCKS_PER_SEC;
}
std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  if (nor_val != 1.0) {
	  Preprocess(img, &input_channels, nor_val);
  }
  else {
	  Preprocess(img, &input_channels);
  }
	clock_t time;
	time = clock();
	net_->Forward();
	printf("Predicted in %f seconds.\n",  sec(clock() - time));
  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}
void Detector::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}
void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels, double normalize_value) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3, normalize_value);
  else
    sample_resized.convertTo(sample_float, CV_32FC1, normalize_value);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.90,
    "Only store detections with score higher than the threshold.");
DEFINE_double(normalize_value, 1.0,
	"Normalize image to 0~1");
DEFINE_int32(wait_time, 1000,
	"cv imshow window waiting time ");
int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;
  const string& file_type = FLAGS_file_type;
  const string& out_file = FLAGS_out_file;
  const float& confidence_threshold = FLAGS_confidence_threshold;
  const float& normalize_value = FLAGS_normalize_value;
  const int& wait_time = FLAGS_wait_time;
  // Initialize the network.
  
  Detector detector(model_file, weights_file, mean_file, mean_value, confidence_threshold, normalize_value);

  // Set the output mode.
  std::streambuf* buf = std::cout.rdbuf();
  std::ofstream outfile;
  if (!out_file.empty()) {
    outfile.open(out_file.c_str());
    if (outfile.good()) {
      buf = outfile.rdbuf();
    }
  }
  std::ostream out(buf);

  // Process image one by one.
  //std::ifstream infile(argv[3]);
  const string &indir = "//data";
  std::string file;
  out << file_type <<"demo";
  if (file_type == "image")
  {
	  char buf[1000];
	  sprintf(buf, "%s/*.jpg", "data//");
	  cv::String path(buf); //select only jpg

	  vector<cv::String> fn;
	  vector<cv::Mat> data;
	  cv::glob(path, fn, true); // recurse
	  for (size_t k = 0; k < fn.size(); ++k)
	  {
		  //cv::Mat img = cv::imread("data//000166.jpg");
		  cv::Mat img = cv::imread(fn[k]);
		  if (img.empty()) continue; //only proceed if sucsessful
									// you probably want to do some preprocessing
		  CHECK(!img.empty()) << "Unable to decode image " << file;
		  Timer batch_timer;
		  batch_timer.Start();
		  std::vector<vector<float> > detections = detector.Detect(img);
		  LOG(INFO) << "Computing time: " << batch_timer.MilliSeconds() << " ms.";
		  /* Print the detection results. */
		  for (int i = 0; i < detections.size(); ++i) {
			  const vector<float>& d = detections[i];
			  // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
			  CHECK_EQ(d.size(), 7);
			  const float score = d[2];
			  if (score >= confidence_threshold) {
				  out << file << " ";
				  out << static_cast<int>(d[1]) << " ";
				  out << score << " ";
				  out << static_cast<int>(d[3] * img.cols) << " ";
				  out << static_cast<int>(d[4] * img.rows) << " ";
				  out << static_cast<int>(d[5] * img.cols) << " ";
				  out << static_cast<int>(d[6] * img.rows) << std::endl;

				  cv::Point pt1, pt2;
				  pt1.x = (img.cols*d[3]);
				  pt1.y = (img.rows*d[4]);
				  pt2.x = (img.cols*d[5]);
				  pt2.y = (img.rows*d[6]);
				  cv::rectangle(img, pt1, pt2, cvScalar(0, 255, 0), 1, 8, 0);

				  char label[100];
				  sprintf(label, "%s,%f", CLASSES[static_cast<int>(d[1])], score);
				  int baseline;
				  cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 0, &baseline);
				  cv::Point pt3;
				  pt3.x = pt1.x + size.width;
				  pt3.y = pt1.y - size.height;
				  cv::rectangle(img, pt1, pt3, cvScalar(0, 255, 0), -1);

				  cv::putText(img, label, pt1, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			  }
		  }
		  cv::imshow("show", img);
		  sprintf(buf, "out//%05d.jpg", k);
		  cv::imwrite(buf, img);
		  cv::waitKey(wait_time);
		  data.push_back(img);
	  }
  }
  else if (file_type == "webcam") {

	  for (size_t k = 0; k < 1; ++k)
	  {
		  cv::VideoCapture cap(0);

		  if (!cap.isOpened()) {
			  LOG(FATAL) << "Failed to open camera";
		  }
		  cv::Mat img;
		  int frame_count = 0;
		  while (true) {
			  bool success = cap.read(img);
			  if (!success) {
				  LOG(INFO) << "Process " << frame_count << " frames from " << file;
				  break;
			  }
			  CHECK(!img.empty()) << "Error when read frame";
			  std::vector<vector<float> > detections = detector.Detect(img);

			  /* Print the detection results. */
			  for (int i = 0; i < detections.size(); ++i) {
				  const vector<float>& d = detections[i];
				  // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
				  CHECK_EQ(d.size(), 7);
				  const float score = d[2];
				  if (score >= confidence_threshold) {
					  out << file << "_";
					  out << std::setfill('0') << std::setw(6) << frame_count << " ";
					  out << static_cast<int>(d[1]) << " ";
					  out << score << " ";
					  out << static_cast<int>(d[3] * img.cols) << " ";
					  out << static_cast<int>(d[4] * img.rows) << " ";
					  out << static_cast<int>(d[5] * img.cols) << " ";
					  out << static_cast<int>(d[6] * img.rows) << std::endl;

					  cv::Point pt1, pt2;
					  pt1.x = (img.cols*d[3]);
					  pt1.y = (img.rows*d[4]);
					  pt2.x = (img.cols*d[5]);
					  pt2.y = (img.rows*d[6]);
					  int index = static_cast<int>(d[1]);
					  int green = 255 * ((index + 1) % 3);
					  int blue = 255 * (index % 3);
					  int red = 255 * ((index + 1) % 4);
					  cv::rectangle(img, pt1, pt2, cvScalar(red, green, blue), 1, 8, 0);

					  char label[100];
					  sprintf(label, "%s,%f", CLASSES[static_cast<int>(d[1])], score);
					  int baseline;
					  cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 0, &baseline);
					  cv::Point pt3;
					  pt3.x = pt1.x + size.width;
					  pt3.y = pt1.y - size.height;

					  cv::rectangle(img, pt1, pt3, cvScalar(red, green, blue), -1);

					  cv::putText(img, label, pt1, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
				  }
			  }
			  cv::imshow("show", img);
			  if (cv::waitKey(30) >= 0)
			  {
				  break;
			  }
			  ++frame_count;
		  }
		  if (cap.isOpened()) {
			  cap.release();
		  }
	  }
  }
  else 
  {
	  char buf[1000];
	  sprintf(buf, "%s/*.avi", "data//");
	  cv::String path(buf); //select only jpg
	  int count = 0;
	  vector<cv::String> fn;
	  vector<cv::Mat> data;
	  cv::glob(path, fn, true); // recurse
	  int max = 14500;
	  for (size_t k = 0; k < fn.size(); ++k)
	  {
		  out << fn[k] << std::endl;
		  cv::VideoCapture cap(fn[k]);

		  if (!cap.isOpened()) {
			  LOG(FATAL) << "Failed to open video: " << file;
		  }
		  cv::Mat img;
		  int frame_count = 0;
		  while (true) {
			  bool success = cap.read(img);
			  if (!success) {
				  LOG(INFO) << "Process " << frame_count << " frames from " << file;
				  break;
			  }
			  CHECK(!img.empty()) << "Error when read frame";
			  std::vector<vector<float> > detections = detector.Detect(img);

			  /* Print the detection results. */
			  for (int i = 0; i < detections.size(); ++i) {
				  const vector<float>& d = detections[i];
				  // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
				  CHECK_EQ(d.size(), 7);
				  const float score = d[2];
				  if (score >= confidence_threshold) {
					  out << file << "_";
					  out << std::setfill('0') << std::setw(6) << frame_count << " ";
					  out << static_cast<int>(d[1]) << " ";
					  out << score << " ";
					  out << static_cast<int>(d[3] * img.cols) << " ";
					  out << static_cast<int>(d[4] * img.rows) << " ";
					  out << static_cast<int>(d[5] * img.cols) << " ";
					  out << static_cast<int>(d[6] * img.rows) << std::endl;

					  cv::Point pt1, pt2;
					  pt1.x = (img.cols*d[3]);
					  pt1.y = (img.rows*d[4]);
					  pt2.x = (img.cols*d[5]);
					  pt2.y = (img.rows*d[6]);
					  int index = static_cast<int>(d[1]);
					  int green = 255 * ((index + 1) % 3);
					  int blue = 255 * (index % 3);
					  int red = 255 * ((index + 1) % 2);
					  cv::rectangle(img, pt1, pt2, cvScalar(red, green, blue), 1, 8, 0);

					  char label[100];
					  sprintf(label, "%s,%f", CLASSES[static_cast<int>(d[1])], score);
					  int baseline;
					  cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 0, &baseline);
					  cv::Point pt3;
					  pt3.x = pt1.x + size.width;
					  pt3.y = pt1.y - size.height;

					  cv::rectangle(img, pt1, pt3, cvScalar(red, green, blue), -1);

					  cv::putText(img, label, pt1, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));


				  }
			  }
			  cv::imshow("show", img);
			  
			  if (count <= max)
			  {
				  cv::Size size;
				  size.width = img.cols;
				  size.height = img.rows;
				  static cv::VideoWriter writer;    // cv::VideoWriter output_video;
				  if (count == 0) {
					  writer.open("VideoTest.avi", CV_FOURCC('D', 'I', 'V', 'X'), 30, size);
				  }
				  else if (count == max) {
					  writer << img;
					  writer.release();
				  }
				  else {
					  writer << img;
				  }


				  count++;

			  }
			  cv::waitKey(1);
			  ++frame_count;
		  }
		  if (cap.isOpened()) {
			  cap.release();
		  }
	  }
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
