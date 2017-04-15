#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/thread/thread.hpp>
#include <boost/shared_ptr.hpp>
#include "FeatureExtractor.h"

//一共就60个模型，所以并行数不要超过60
#define NUM 60

using boost::shared_ptr;

const string FeatureExtractor::regions[] ={
        "1_F","1_E","1_EN","1_N","1_NM",
        "2_LE","2_RE","2_N","2_LM","2_RM"
};

const string FeatureExtractor::types[] = {"GRAY","RGB"};

const boost::tuple<float,float> FeatureExtractor::scales[] = {
	boost::make_tuple(0.4f,0.64f),
	boost::make_tuple(0.3f,0.48f),
	boost::make_tuple(0.24f,0.39f)
};

//需要初始化的静态成员
dlib::frontal_face_detector FeatureExtractor::frontaldetector;
dlib::shape_predictor FeatureExtractor::sp;
map<string,DeepID> FeatureExtractor::deepid_extractors;
FeatureExtractor FeatureExtractor::obj = FeatureExtractor("model_values","deploy_models");

FeatureExtractor::FeatureExtractor(path model_values_dir,path deploy_models_dir)
{
	if(false == exists(model_values_dir) || false == is_directory(model_values_dir))
		throw runtime_error("model_values文件夹不存在！");
	if(false == exists(deploy_models_dir) || false == is_directory(deploy_models_dir))
		throw runtime_error("deploy_models文件夹不存在！");
	
	//初始化静态成员
	frontaldetector = dlib::get_frontal_face_detector();
	dlib::deserialize("model_values/shape_predictor_68_face_landmarks.dat") >> sp;	
	for(int r = 0 ; r < sizeof(regions)/sizeof(string) ; r++)
		for(int t = 0 ; t < sizeof(types) / sizeof(string) ; t++)
			for(int s = 0 ; s < sizeof(scales) / sizeof(boost::tuple<float,float>) ; s++) {
				string caffemodel = regions[r] + "_" + types[t] + "_" + lexical_cast<string>(s);
				path deploypath = deploy_models_dir / (string("deepid_") + (("RGB" == types[t])?"rgb_":"gray_") + (("1_F" == regions[r])?"3139":"3131") + ".prototxt");
				path modelpath = model_values_dir / (caffemodel + ".caffemodel");
				path meanpath = model_values_dir / (caffemodel + ".binaryproto");
#ifndef NDEBUG
				assert(true == exists(deploypath) && false == is_directory(deploypath));
				assert(true == exists(modelpath) && false == is_directory(deploypath));
#endif
				deepid_extractors.insert(make_pair(caffemodel,DeepID(deploypath.string(),modelpath.string(),meanpath.string())));
			}
}

FeatureExtractor::~FeatureExtractor(){}

FeatureExtractor & FeatureExtractor::get() {return obj;}

boost::tuple<bool,ublas::vector<float> > FeatureExtractor::operator()(const cv::Mat& img)
{
	if(img.empty()) return boost::make_tuple(false,ublas::vector<float>());
	//1)检测人脸
	dlib::cv_image<dlib::bgr_pixel> cimg(img);
	vector<dlib::rectangle> frontalfaces = frontaldetector(cimg);
	if(0 == frontalfaces.size()) return boost::make_tuple(false,ublas::vector<float>());
	dlib::rectangle & rect = frontalfaces.front();
	//2)标注人脸并获取五点关键点
	dlib::full_object_detection shape = sp(cimg,rect);
	vector<Point2f> landmarks;
	landmarks.push_back(Point2f(
		(shape.part(36).x() + shape.part(39).x()) / 2,
		(shape.part(36).y() + shape.part(39).y()) / 2
	));
	landmarks.push_back(Point2f(
		(shape.part(42).x() + shape.part(45).x()) / 2,
		(shape.part(42).y() + shape.part(45).y()) / 2
	));
	landmarks.push_back(Point2f(shape.part(33).x(),shape.part(33).y()));
	landmarks.push_back(Point2f(shape.part(48).x(),shape.part(48).y()));
	landmarks.push_back(Point2f(shape.part(54).x(),shape.part(54).y()));
	//3)生成神经网输入
	map<string,pair<Mat,Mat> > inputs;
	//3.1)获取F E EN N NM
	for(int i = 0 ; i < sizeof(scales) / sizeof(boost::tuple<float,float>) ; i++) {
		Mat F,E,EN,N,NM;
		//两眼之间的距离
		double eyesdistance = 31 * ::get<0>(scales[i]);
		Point2f new_left_eye((31 - eyesdistance)*0.5,39*0.5);
		F = cropFace(img,landmarks[0],landmarks[1],eyesdistance,new_left_eye,Size(31,39));
		double span = 39-31;
		Rect earea(0,0,31,31);	F(earea).copyTo(E);
		Rect enarea(0,3,31,31);	F(enarea).copyTo(EN);
		Rect narea(0,5,31,31);	F(narea).copyTo(N);
		Rect nmarea(0,8,31,31);	F(nmarea).copyTo(NM);
		//保存
		save_rgb_gray(F,"1_F",i,inputs);
		save_rgb_gray(E,"1_E",i,inputs);
		save_rgb_gray(EN,"1_EN",i,inputs);
		save_rgb_gray(N,"1_N",i,inputs);
		save_rgb_gray(NM,"1_NM",i,inputs);
	}
	//3.2)获取LE RE N LM RM
	for(int i = 0 ; i < sizeof(scales) / sizeof(boost::tuple<float,float>) ; i++) {
		Mat LE,RE,N,LM,RM;
		//两眼之间的距离
		double eyesdistance = 31 * ::get<1>(scales[i]);
		LE = cropFace(img,landmarks,eyesdistance,0,Size(31,31));
		RE = cropFace(img,landmarks,eyesdistance,1,Size(31,31));
		N = cropFace(img,landmarks,eyesdistance,2,Size(31,31));
		LM = cropFace(img,landmarks,eyesdistance,3,Size(31,31));
		RM = cropFace(img,landmarks,eyesdistance,4,Size(31,31));
		save_rgb_gray(LE,"2_LE",i,inputs);
		save_rgb_gray(RE,"2_RE",i,inputs);
		save_rgb_gray(N,"2_N",i,inputs);
		save_rgb_gray(LM,"2_LM",i,inputs);
		save_rgb_gray(RM,"2_RM",i,inputs);
	}
	//4)提取特征向量
	ublas::vector<float> output = ublas::zero_vector<float>(deepid_extractors.size() * 160 * 2);
	vector<boost::shared_ptr<boost::thread> > handlers;
	int index = 0;
	for(map<string,pair<Mat,Mat> >::iterator it = inputs.begin() ; it != inputs.end() ; it++) {
#ifndef NDEBUG
		assert(false == it->second.first.empty());
		assert(false == it->second.second.empty());
#endif
		handlers.push_back(boost::shared_ptr<boost::thread> (
			new boost::thread(&FeatureExtractor::process,this,it->first,it->second,boost::ref(output),160 * 2 * index++)
		));
		if(NUM <= handlers.size()) {
			for(
				vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ;
				it != handlers.end() ;
				it++
			) (*it)->join();
			handlers.clear();
		}
	}
	if(handlers.size()) {
		for(
			vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ;
			it != handlers.end() ;
			it++
		) (*it)->join();
		handlers.clear();
	}
	return boost::make_tuple(true,output);
}

boost::tuple<bool,ublas::vector<float> > FeatureExtractor::getFeatureFromFace(const cv::Mat & img)
{
	if(img.empty()) return boost::make_tuple(false,ublas::vector<float>());
	dlib::cv_image<dlib::bgr_pixel> cimg(img);
	//2)标注人脸并获取五点关键点
	dlib::full_object_detection shape = sp(cimg,dlib::rectangle(0,0,img.cols - 1,img.rows - 1));
	vector<Point2f> landmarks;
	landmarks.push_back(Point2f(
		(shape.part(36).x() + shape.part(39).x()) / 2,
		(shape.part(36).y() + shape.part(39).y()) / 2
	));
	landmarks.push_back(Point2f(
		(shape.part(42).x() + shape.part(45).x()) / 2,
		(shape.part(42).y() + shape.part(45).y()) / 2
	));
	landmarks.push_back(Point2f(shape.part(33).x(),shape.part(33).y()));
	landmarks.push_back(Point2f(shape.part(48).x(),shape.part(48).y()));
	landmarks.push_back(Point2f(shape.part(54).x(),shape.part(54).y()));
	//3)生成神经网输入
	map<string,pair<Mat,Mat> > inputs;
	//3.1)获取F E EN N NM
	for(int i = 0 ; i < sizeof(scales) / sizeof(boost::tuple<float,float>) ; i++) {
		Mat F,E,EN,N,NM;
		//两眼之间的距离
		double eyesdistance = 31 * ::get<0>(scales[i]);
		Point2f new_left_eye((31 - eyesdistance)*0.5,39*0.5);
		F = cropFace(img,landmarks[0],landmarks[1],eyesdistance,new_left_eye,Size(31,39));
		double span = 39-31;
		Rect earea(0,0,31,31);	F(earea).copyTo(E);
		Rect enarea(0,3,31,31);	F(enarea).copyTo(EN);
		Rect narea(0,5,31,31);	F(narea).copyTo(N);
		Rect nmarea(0,8,31,31);	F(nmarea).copyTo(NM);
		//保存
		save_rgb_gray(F,"1_F",i,inputs);
		save_rgb_gray(E,"1_E",i,inputs);
		save_rgb_gray(EN,"1_EN",i,inputs);
		save_rgb_gray(N,"1_N",i,inputs);
		save_rgb_gray(NM,"1_NM",i,inputs);
	}
	//3.2)获取LE RE N LM RM
	for(int i = 0 ; i < sizeof(scales) / sizeof(boost::tuple<float,float>) ; i++) {
		Mat LE,RE,N,LM,RM;
		//两眼之间的距离
		double eyesdistance = 31 * ::get<1>(scales[i]);
		LE = cropFace(img,landmarks,eyesdistance,0,Size(31,31));
		RE = cropFace(img,landmarks,eyesdistance,1,Size(31,31));
		N = cropFace(img,landmarks,eyesdistance,2,Size(31,31));
		LM = cropFace(img,landmarks,eyesdistance,3,Size(31,31));
		RM = cropFace(img,landmarks,eyesdistance,4,Size(31,31));
		save_rgb_gray(LE,"2_LE",i,inputs);
		save_rgb_gray(RE,"2_RE",i,inputs);
		save_rgb_gray(N,"2_N",i,inputs);
		save_rgb_gray(LM,"2_LM",i,inputs);
		save_rgb_gray(RM,"2_RM",i,inputs);
	}
	//4)提取特征向量
	ublas::vector<float> output = ublas::zero_vector<float>(deepid_extractors.size() * 160 * 2);
	vector<boost::shared_ptr<boost::thread> > handlers;
	int index = 0;
	for(map<string,pair<Mat,Mat> >::iterator it = inputs.begin() ; it != inputs.end() ; it++) {
#ifndef NDEBUG
		assert(false == it->second.first.empty());
		assert(false == it->second.second.empty());
#endif
		handlers.push_back(boost::shared_ptr<boost::thread> (
			new boost::thread(&FeatureExtractor::process,this,it->first,it->second,boost::ref(output),160 * 2 * index++)
		));
		if(NUM <= handlers.size()) {
			for(
				vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ;
				it != handlers.end() ;
				it++
			) (*it)->join();
			handlers.clear();
		}
	}
	if(handlers.size()) {
		for(
			vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ;
			it != handlers.end() ;
			it++
		) (*it)->join();
		handlers.clear();
	}
	return boost::make_tuple(true,output);
}

Mat FeatureExtractor::cropFace(Mat img, Point2f eye_left, Point2f eye_right, double eyesdistance, Point2f new_eye_left, Size dest_sz)
{
	//eyesdistance是两眼之间的距离
	//new_eye_left是左眼的新坐标
	//dest_sz是最后输出图片的大小

	//计算眼睛到顶和侧边的距离
	double offset_w = new_eye_left.x;
	double offset_h = new_eye_left.y;
	pair<double,double> eye_direction(eye_right.x - eye_left.x, eye_right.y - eye_left.y);
	double angle = atan2(eye_direction.second,eye_direction.first);
	double dist = sqrt((eye_left.x - eye_right.x) * (eye_left.x - eye_right.x) + (eye_left.y - eye_right.y) * (eye_left.y - eye_right.y));
	//计算目标两眼之间距离
	double reference = eyesdistance;
	//计算尺度
	double scale = reference / dist;
	Mat affine = Mat::eye(Size(3,2),CV_32FC1);
	affine.at<float>(0,0) = cos(angle) * scale;		affine.at<float>(0,1) = sin(angle) * scale;
	affine.at<float>(1,0) = -sin(angle) * scale;	affine.at<float>(1,1) = cos(angle) * scale;
	affine.at<float>(0,2) = -(eye_left.x * affine.at<float>(0,0) + eye_left.y * affine.at<float>(0,1) - offset_w);
	affine.at<float>(1,2) = -(eye_left.x * affine.at<float>(1,0) + eye_left.y * affine.at<float>(1,1) - offset_h);	
	Mat retVal;
	warpAffine(img, retVal, affine, dest_sz);
	return retVal;
}

Mat FeatureExtractor::cropFace(Mat img, vector<Point2f> landmarks, double eyesdistance, int which_center, Size dest_sz)
{
	//eyesdistance是两眼之间的距离
	//which_center是采用哪个关键点作为新的中心
	//dest_sz是最后输出图片的大小
	
	//计算眼睛到顶和侧边的距离
	double offset_w = 0.5 * dest_sz.width;
	double offset_h = 0.5 * dest_sz.height;
	pair<double,double> eye_direction(landmarks[1].x - landmarks[0].x, landmarks[1].y - landmarks[0].y);
	double angle = atan2(eye_direction.second,eye_direction.first);
	double dist = sqrt((landmarks[0].x - landmarks[1].x) * (landmarks[0].x - landmarks[1].x) + (landmarks[0].y - landmarks[1].y) * (landmarks[0].y - landmarks[1].y));
	//计算目标两眼之间距离
	double reference = eyesdistance;
	//计算尺度
	double scale = reference / dist;
	Mat affine = Mat::eye(Size(3,2),CV_32FC1);
	affine.at<float>(0,0) = cos(angle) * scale;		affine.at<float>(0,1) = sin(angle) * scale;
	affine.at<float>(1,0) = -sin(angle) * scale;	affine.at<float>(1,1) = cos(angle) * scale;
	affine.at<float>(0,2) = -(landmarks[which_center].x * affine.at<float>(0,0) + landmarks[which_center].y * affine.at<float>(0,1) - offset_w);
	affine.at<float>(1,2) = -(landmarks[which_center].x * affine.at<float>(1,0) + landmarks[which_center].y * affine.at<float>(1,1) - offset_h);	
	Mat retVal;
	warpAffine(img, retVal, affine, dest_sz);
	return retVal;
}

void FeatureExtractor::save_rgb_gray(Mat img, string label, int scale, map<string,pair<Mat,Mat> > & output)
{
	Mat gray;
	cvtColor(img,gray,CV_BGR2GRAY);
	save_orig_flip(img,label,"_RGB_",scale,output);
	save_orig_flip(gray,label,"_GRAY_",scale,output);
}

void FeatureExtractor::save_orig_flip(Mat img, string label, string clr, int scale, map<string,pair<Mat,Mat> > & output)
{
	Mat flpimg;
	flip(img,flpimg,1);
	if(label == "2_LE") {
		string index = "2_LE" + clr + lexical_cast<string>(scale);
		img.copyTo(output[index].first);
		index = "2_RE" + clr + lexical_cast<string>(scale);
		flpimg.copyTo(output[index].second);
	} else if(label == "2_RE") {
		string index = "2_RE" + clr + lexical_cast<string>(scale);
		img.copyTo(output[index].first);
		index = "2_LE" + clr + lexical_cast<string>(scale);
		flpimg.copyTo(output[index].second);
	} else if(label == "2_LM") {
		string index = "2_LM" + clr + lexical_cast<string>(scale);
		img.copyTo(output[index].first);
		index = "2_RM" + clr + lexical_cast<string>(scale);
		flpimg.copyTo(output[index].second);
	} else if(label == "2_RM") {
		string index = "2_RM" + clr + lexical_cast<string>(scale);
		img.copyTo(output[index].first);
		index = "2_LM" + clr + lexical_cast<string>(scale);
		flpimg.copyTo(output[index].second);
	} else {
		string index = label + clr + lexical_cast<string>(scale);
		img.copyTo(output[index].first);
		flpimg.copyTo(output[index].second);
	}
}

void FeatureExtractor::process(string index,pair<Mat,Mat> imgs,ublas::vector<float> & output,int start_index)
{
	map<string,DeepID>::iterator which = deepid_extractors.find(index);
#ifndef NDEBUG
	assert(which != deepid_extractors.end());
#endif
	vector<float> fv1 = which->second(imgs.first);
	vector<float> fv2 = which->second(imgs.second);
	copy(fv1.begin(),fv1.end(),output.begin() + start_index);
	copy(fv2.begin(),fv2.end(),output.begin() + start_index + fv1.size());
}
