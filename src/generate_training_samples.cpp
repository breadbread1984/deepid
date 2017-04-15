#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/thread/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/lexical_cast.hpp>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>

#define NUM 32

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace boost::program_options;
using namespace boost::algorithm;
using boost::shared_ptr;
using namespace cv;

//F是39x31图片
//其他都是31x31图片
static const string regions[] ={
	"1_F","1_E","1_EN","1_N","1_NM",
	"2_LE","2_RE","2_N","2_LM","2_RM"
};

static const string types[] = {"GRAY","RGB"};
//三种尺度，patch的高，patch的宽，两眼距离占宽的比例
static const boost::tuple<float,float> scales[] = {boost::make_tuple(0.4f,0.64f),boost::make_tuple(0.3f,0.48f),boost::make_tuple(0.24f,0.39f)};

dlib::frontal_face_detector frontaldetector = dlib::get_frontal_face_detector();
dlib::shape_predictor sp;

void process(path inputdir, path outputroot);
Mat cropFace(Mat img, Point2f eye_left, Point2f eye_right, double eyesdistance, Point2f new_eye_left, Size dest_sz = Size(31,39));
Mat cropFace(Mat img, vector<Point2f> landmarks, double eyesdistance, int which_center, Size dest_sz = Size(31,31));
void save_rgb_gray(Mat img, string label,int scale,string filename,string inputdir,path outputroot);
void save_orig_flip(Mat img, string label, string clr, int scale,string filename, string inputdir, path outputroot);

int main(int argc,char ** argv)
{
	options_description desc;
	string inputdir,outputdir;
	desc.add_options()
		("help,h","打印当前使用方法")
		("input,i",value<string>(&inputdir),"WebFace文件夹路径")
		("output,o",value<string>(&outputdir),"输出训练集的路径");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || vm.count("help") || 1 != vm.count("input") || 1 != vm.count("output")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	if(false == exists(inputdir) || false == is_directory(inputdir)) {
		cout<<"WebFace文件夹不存在！"<<endl;
		return EXIT_FAILURE;
	}
	
	path outputroot(outputdir);
	remove_all(outputroot);
	create_directory(outputroot);
	dlib::deserialize("model_values/shape_predictor_68_face_landmarks.dat") >> sp;
//创建文件夹
	//输出路径下有8x2x3=48个子文件夹
	for(int r = 0 ; r < sizeof(regions)/sizeof(string) ; r++)
		for(int t = 0 ; t < sizeof(types) / sizeof(string) ; t++)
			for(int s = 0 ; s < sizeof(scales) / sizeof(boost::tuple<float,float>) ; s++) {
				string filename = regions[r] + "_" + types[t] + "_" + lexical_cast<string>(s);
				path p = outputroot / filename;
				create_directory(p);
			}
	vector<boost::shared_ptr<boost::thread> > handlers;
	for(directory_iterator it(inputdir) ; it != directory_iterator() ; it++) {
		if(is_directory(it->path())) {
			handlers.push_back(
				boost::shared_ptr<boost::thread> (
					new boost::thread(boost::bind(::process,it->path(),outputroot))
				)
			);
			if(NUM <= handlers.size()) {
				for(
					vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ;
					it != handlers.end();
					it++
				) (*it)->join();
				handlers.clear();
			}
		}//if directory
	}
	if(handlers.size()) {
		for(
			vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ;
			it != handlers.end() ;
			it++
		) (*it)->join();
		handlers.clear();
	}
	
	return EXIT_SUCCESS;
}

void process(path inputdir, path outputroot)
{	
	//创建当前身份文件夹
	for(int r = 0 ; r < sizeof(regions)/sizeof(string) ; r++)
		for(int t = 0 ; t < sizeof(types) / sizeof(string) ; t++)
			for(int s = 0 ; s < sizeof(scales) / sizeof(boost::tuple<float,float>) ; s++) {
				string dirname = regions[r] + "_" + types[t] + "_" + lexical_cast<string>(s);
				path p = outputroot / dirname / inputdir.filename().string();
				create_directory(p);
			}
	
	for(directory_iterator it(inputdir) ; it != directory_iterator() ; it++) {
		if(is_directory(it->path())) continue;
		string ext = it->path().extension().string();
		if(".jpg" != ext && ".JPG" != ext) continue;
		Mat img = imread(it->path().string());
		if(true == img.empty()) {
			cerr << "图片文件错误："<<it->path().string()<<"不存在！"<<endl;
			continue;
		}
		//1)检测人脸
		dlib::cv_image<dlib::bgr_pixel> cimg(img);
		vector<dlib::rectangle> frontalfaces = frontaldetector(cimg);
		if(0 == frontalfaces.size()) continue;
		dlib::rectangle & rect = frontalfaces.front();
		//2)标注人脸并获取左右眼睛坐标
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
		//3)提取F EN NM
		{
			Mat F[sizeof(scales) / sizeof(boost::tuple<float,float>)];
			Mat E[sizeof(scales) / sizeof(boost::tuple<float,float>)];
			Mat EN[sizeof(scales) / sizeof(boost::tuple<float,float>)];
			Mat N[sizeof(scales) / sizeof(boost::tuple<float,float>)];
			Mat NM[sizeof(scales) / sizeof(boost::tuple<float,float>)];
			for(int i = 0 ; i < sizeof(scales) / sizeof(boost::tuple<float,float>) ; i++) {
				//两眼之间的距离
				double eyesdistance = 31 * get<0>(scales[i]);
				Point2f new_left_eye((31 - eyesdistance)*0.5,39*0.5);
				F[i] = cropFace(img,landmarks[0],landmarks[1],eyesdistance,new_left_eye,Size(31,39));
				double span = 39-31;
				Rect earea(0,0,31,31);		F[i](earea).copyTo(E[i]);
				Rect enarea(0,3,31,31);	F[i](enarea).copyTo(EN[i]);
				Rect narea(0,5,31,31);		F[i](narea).copyTo(N[i]);
				Rect nmarea(0,8,31,31);	F[i](nmarea).copyTo(NM[i]);
				//保存到对应文件夹
				save_rgb_gray(F[i],"1_F",i,it->path().filename().string(),inputdir.filename().string(),outputroot);
				save_rgb_gray(E[i],"1_E",i,it->path().filename().string(),inputdir.filename().string(),outputroot);
				save_rgb_gray(EN[i],"1_EN",i,it->path().filename().string(),inputdir.filename().string(),outputroot);
				save_rgb_gray(N[i],"1_N",i,it->path().filename().string(),inputdir.filename().string(),outputroot);
				save_rgb_gray(NM[i],"1_NM",i,it->path().filename().string(),inputdir.filename().string(),outputroot);
			}
		}//提取F EN NM
		//4)提取LE RE N LM RM
		{
			Mat LE[sizeof(scales)/sizeof(boost::tuple<float,float>)];
			Mat RE[sizeof(scales)/sizeof(boost::tuple<float,float>)];
			Mat N[sizeof(scales)/sizeof(boost::tuple<float,float>)];
			Mat LM[sizeof(scales)/sizeof(boost::tuple<float,float>)];
			Mat RM[sizeof(scales)/sizeof(boost::tuple<float,float>)];
			for(int i = 0 ; i< sizeof(scales)/sizeof(boost::tuple<float,float>) ; i++) {
				double eyesdistance = 31 * get<1>(scales[i]);
				LE[i] = cropFace(img,landmarks,eyesdistance,0,Size(31,31));
				RE[i] = cropFace(img,landmarks,eyesdistance,1,Size(31,31));
				N[i] = cropFace(img,landmarks,eyesdistance,2,Size(31,31));
				LM[i] = cropFace(img,landmarks,eyesdistance,3,Size(31,31));
				RM[i] = cropFace(img,landmarks,eyesdistance,4,Size(31,31));
				save_rgb_gray(LE[i],"2_LE",i,it->path().filename().string(),inputdir.filename().string(),outputroot);
				save_rgb_gray(RE[i],"2_RE",i,it->path().filename().string(),inputdir.filename().string(),outputroot);
				save_rgb_gray(N[i],"2_N",i,it->path().filename().string(),inputdir.filename().string(),outputroot);
				save_rgb_gray(LM[i],"2_LM",i,it->path().filename().string(),inputdir.filename().string(),outputroot);
				save_rgb_gray(RM[i],"2_RM",i,it->path().filename().string(),inputdir.filename().string(),outputroot);
			}
		}//提取LE RE N LM RM
	}//for 当前文件夹下面的每个图片
}

Mat cropFace(Mat img, Point2f eye_left, Point2f eye_right, double eyesdistance, Point2f new_eye_left, Size dest_sz)
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

Mat cropFace(Mat img, vector<Point2f> landmarks, double eyesdistance, int which_center, Size dest_sz)
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

void save_rgb_gray(Mat img, string label,int scale,string filename,string inputdir,path outputroot)
{
	//inputdir是正在处理的某个身份对应的文件夹
	//outputroot是输出文件夹
	Mat gray;
	cvtColor(img,gray,CV_BGR2GRAY);
	save_orig_flip(img,label,"_RGB_",scale,filename,inputdir,outputroot);
	save_orig_flip(gray,label,"_GRAY_",scale,filename,inputdir,outputroot);
}

void save_orig_flip(Mat img, string label, string clr, int scale,string filename, string inputdir, path outputroot)
{
	Mat flpimg;
	flip(img,flpimg,1);
	if(label == "2_LE") {
		string dirname = "2_LE" + clr + lexical_cast<string>(scale);
		path dir = outputroot / dirname / inputdir;
		imwrite((dir/("orig_" + filename)).string(),img);
		dirname = "2_RE" + clr + lexical_cast<string>(scale);
		dir = outputroot / dirname / inputdir;
		imwrite((dir/("flip_" + filename)).string(),flpimg);
	} else if(label == "2_RE") {
		string dirname = "2_RE" + clr + lexical_cast<string>(scale);
		path dir = outputroot / dirname / inputdir;
		imwrite((dir/("orig_" + filename)).string(),img);
		dirname = "2_LE" + clr + lexical_cast<string>(scale);
		dir = outputroot / dirname / inputdir;
		imwrite((dir/("flip_" + filename)).string(),flpimg);
	} else if(label == "2_LM") {
		string dirname = "2_LM" + clr + lexical_cast<string>(scale);
		path dir = outputroot / dirname / inputdir;
		imwrite((dir/("orig_" + filename)).string(),img);
		dirname = "2_RM" + clr + lexical_cast<string>(scale);
		dir = outputroot / dirname / inputdir;
		imwrite((dir/("flip_" + filename)).string(),flpimg);
	} else if(label == "2_RM") {
		string dirname = "2_RM" + clr + lexical_cast<string>(scale);
		path dir = outputroot / dirname / inputdir;
		imwrite((dir/("orig_" + filename)).string(),img);
		dirname = "2_LM" + clr + lexical_cast<string>(scale);
		dir = outputroot / dirname / inputdir;
		imwrite((dir/("flip_" + filename)).string(),flpimg);
	} else {
		string dirname = label + clr + lexical_cast<string>(scale);
		path dir = outputroot / dirname / inputdir;
		imwrite((dir/("orig_" + filename)).string(),img);
		imwrite((dir/("flip_" + filename)).string(),flpimg);
	}
}
