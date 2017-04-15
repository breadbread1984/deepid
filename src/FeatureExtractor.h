#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include <string>
#include <map>
#include <boost/tuple/tuple.hpp>
#include <boost/filesystem.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include "DeepID.h"

using namespace std;
using namespace boost;
using namespace boost::filesystem;
namespace ublas = boost::numeric::ublas;
using namespace cv;

class FeatureExtractor {
	static const string regions[];
	static const string types[];
	static const boost::tuple<float,float> scales[];
	static dlib::frontal_face_detector frontaldetector;
	static dlib::shape_predictor sp;
	static map<string,DeepID> deepid_extractors;
	//singleton instance
	static FeatureExtractor obj;
protected:
	FeatureExtractor(path model_values_dir="model_values",path deploy_models_dir="deploy_models");
	Mat cropFace(Mat img, Point2f eye_left, Point2f eye_right, double eyesdistance, Point2f new_eye_left, Size dest_sz = Size(31,39));
	Mat cropFace(Mat img, vector<Point2f> landmarks, double eyesdistance, int which_center, Size dest_sz = Size(31,31));
	void save_rgb_gray(Mat img, string label, int scale, map<string,pair<Mat,Mat> > & output);
	void save_orig_flip(Mat img, string label, string clr, int scale, map<string,pair<Mat,Mat> > & output);
	void process(string index,pair<Mat,Mat> imgs,ublas::vector<float> & output,int start_index);
public:
	static FeatureExtractor & get();
	virtual ~FeatureExtractor();
	//img是只包含一个人脸的图片
	boost::tuple<bool,ublas::vector<float> > operator()(const cv::Mat& img);
	boost::tuple<bool,ublas::vector<float> > getFeatureFromFace(const cv::Mat & img);
};

#endif
