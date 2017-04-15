#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <map>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <opencv2/opencv.hpp>
#include "matrix_basic.hpp"
#include "FeatureExtractor.h"

using namespace std;
using namespace boost::program_options;
using namespace boost::filesystem;
using namespace boost::archive;
namespace ublas = boost::numeric::ublas;
using namespace cv;
using namespace cv::ml;

int train(string inputdir);
int test(string paramfile);
void convert(vector<boost::tuple<path,int> > & inputlist,vector<boost::tuple<ublas::vector<float>,int> > & outputlist);
Mat crop(Mat img,Rect bounding,Size size);

int main(int argc, char ** argv)
{
	options_description desc;
	string mode,inputdir,paramfile;
	desc.add_options()
		("help,h","打印当前使用方法")
		("mode,m",value<string>(&mode),"选择运行模式（train或者test）")
		("input,i",value<string>(&inputdir),"训练集文件夹路径")
		("param,p",value<string>(&paramfile),"训练的参数文件");
		variables_map vm;
		store(parse_command_line(argc,argv,desc),vm);
		notify(vm);
		
		if(1 == argc || vm.count("help") || false == ((mode == "train" && 1 == vm.count("input")) || (mode == "test" && 1 == vm.count("param")))) {
			cout<<desc;
			return EXIT_SUCCESS;
		}
		
		return ("train" == mode)?train(inputdir):test(paramfile);
}

int train(string inputdir)
{
	path inputroot(inputdir);
	if(false == exists(inputroot) || false == is_directory(inputroot)) {
		cout<<"文件夹不存在"<<endl;
		return EXIT_FAILURE;
	}
	//1)生成list
	int label = -1;
	map<int,string> label2name;
	vector<boost::tuple<path,int> > trainlist;
	for(directory_iterator it(inputroot) ; it != directory_iterator() ; it++) {
		if(is_directory(it->path())) {
			label2name.insert(make_pair(label,it->path().filename().string()));
			//获得当前文件夹下面的文件列表
			vector<boost::tuple<path,int> > list;
			for(directory_iterator img_itr(it->path()) ; img_itr != directory_iterator() ; img_itr++)
				if(img_itr->path().extension().string() == ".jpg") list.push_back(boost::make_tuple(img_itr->path(),label));
			//对列表打乱顺序
			random_shuffle(list.begin(),list.end());
			//然后分别放到训练和测试集合列表
			trainlist.insert(trainlist.end(),list.begin(),list.end());
			label++;
		} // if 是文件夹
	}
	random_shuffle(trainlist.begin(),trainlist.end());
	//2)生成训练集
#ifndef NDEBUG
	cout<<"生成训练集"<<endl;
#endif
	vector<boost::tuple<ublas::vector<float>,int> > trainsamples;
	convert(trainlist,trainsamples);
	//3)训练knn分类器
#ifndef NDEBUG
	cout<<"训练分类器"<<endl;
#endif
	Mat matTrainFeatures(trainsamples.size(),19200,CV_32F);
	Mat matTrainLabels(trainsamples.size(),1,CV_32F);
	for(int i = 0 ; i < trainsamples.size() ; i++) {
		ublas::vector<float> & fv = get<0>(trainsamples[i]);
		int & label = get<1>(trainsamples[i]);
#ifndef NDEBUG
		assert(19200 == fv.size());
#endif
		copy(fv.begin(),fv.end(),matTrainFeatures.ptr<float>(i));
		matTrainLabels.at<float>(i,0) = label;
	}
	//序列化训练集和监督值
	std::ofstream out("训练参数.dat");
	text_oarchive oa(out);
	oa<<label2name<<matTrainFeatures<<matTrainLabels;

	return EXIT_SUCCESS;
}

int test(string paramfile)
{
	std::ifstream in(paramfile.c_str());
	if(false == in.is_open()) {
		cout<<"错误的参数文件！"<<endl;
		return EXIT_FAILURE;
	}
	
	text_iarchive ia(in);
	map<int,string> label2name;
	Mat matTrainFeatures,matTrainLabels;
	ia >> label2name >> matTrainFeatures >> matTrainLabels;
	Ptr<TrainData> trainingData = TrainData::create(matTrainFeatures,SampleTypes::ROW_SAMPLE,matTrainLabels);
	Ptr<KNearest> kclassifier = KNearest::create();
	kclassifier->setIsClassifier(true);
	kclassifier->setAlgorithmType(KNearest::Types::BRUTE_FORCE);
	kclassifier->setDefaultK(1); //只找到一个可能的身份
	kclassifier->train(trainingData);
	//4)测试训练结果
	namedWindow("test",WINDOW_NORMAL);
	VideoCapture cap(0); //打开摄像头
	Mat img;
	dlib::frontal_face_detector frontaldetector = dlib::get_frontal_face_detector();
	FeatureExtractor & extractor = FeatureExtractor::get();
	while(cap.read(img)) {
		//检测当前图中所有人脸
		dlib::cv_image<dlib::bgr_pixel> cimg(img);
		vector<dlib::rectangle> frontalfaces = frontaldetector(cimg);
		//将所有人脸特征计算出来
		Mat matFeatures(frontalfaces.size(),19200,CV_32F);
		for(int i = 0 ; i < frontalfaces.size() ; i++) {
			dlib::rectangle & rect = frontalfaces[i];
			Rect bounding(Point2i(rect.left(),rect.top()),Point2i(rect.right() + 1,rect.bottom() + 1));
			Mat face = crop(img,bounding,Size(bounding.width,bounding.height));
			boost::tuple<bool,ublas::vector<float> > fv = extractor.getFeatureFromFace(face);
			assert(get<0>(fv));
			copy(get<1>(fv).begin(),get<1>(fv).end(),matFeatures.ptr<float>(i));
		}
		
		//对所有人脸特征寻找最近邻
		Mat matResults(frontalfaces.size(),1,CV_32F);
		kclassifier->findNearest(matFeatures,kclassifier->getDefaultK(),matResults);
		//将识别的身份打印在目标框上
		for(int i = 0 ; i < matResults.rows ; i++) {
			dlib::rectangle & rect = frontalfaces[i];
			Rect bounding(Point2i(rect.left(),rect.top()),Point2i(rect.right() + 1,rect.bottom() + 1));
			rectangle(img,bounding,Scalar(255,0,0),5);
			putText(
				img,
				label2name[static_cast<int>(matResults.at<float>(i,0))],
				Point(bounding.x,bounding.y),
				FONT_HERSHEY_SCRIPT_SIMPLEX,
				2,Scalar(0,255,0),3,8
			);
		}
		imshow("test",img);
		char k = waitKey(1);
		if(k == 'q') break;
	}
	return EXIT_SUCCESS;
}

void convert(vector<boost::tuple<path,int> > & inputlist,vector<boost::tuple<ublas::vector<float>,int> > & outputlist)
{
#ifndef NDEBUG
	int count = 0;
#endif
	FeatureExtractor & extractor = FeatureExtractor::get();
	for(vector<boost::tuple<path,int> >::iterator it = inputlist.begin() ; it != inputlist.end() ; it++) {
		Mat img = imread(get<0>(*it).string());
		if(img.empty()) {
			cout<<"无法打开"<<get<0>(*it).string()<<endl;
			continue;
		}
		boost::tuple<bool,ublas::vector<float> > fv = extractor(img);
		if(false == get<0>(fv)) continue;
		outputlist.push_back(boost::make_tuple(get<1>(fv),get<1>(*it)));
		count++;
#ifndef NDEBUG
		if(++count %1000 == 0) 	cout<<"已经处理了"<<count<<"个样本"<<endl;
#endif
	}
}

Mat crop(Mat img,Rect bounding,Size size)
{
	ublas::matrix<float> A(3,4),B(3,4);
	A(0,0) = bounding.x;	A(0,1) = bounding.x;										A(0,2) = bounding.x + bounding.width;	A(0,3) = bounding.x + bounding.width;
	A(1,0) = bounding.y;	A(1,1) = bounding.y + bounding.height;	A(1,2) = bounding.y + bounding.height;	A(1,3) = bounding.y;
	A(2,0) = 1;					A(2,1) = 1;														A(2,2) = 1;														A(2,3) = 1;
	B(0,0) = 0;	B(0,1) = 0;					B(0,2) = size.width;	B(0,3) = size.width;
	B(1,0) = 0;	B(1,1) = size.height;	B(1,2) = size.height;	B(1,3) = 0;
	B(2,0) = 1;	B(2,1) = 1;					B(2,2) = 1;					B(2,2) = 1;
	ublas::matrix<float> AAt = prod(A,trans(A));
	ublas::matrix<float> ABt = prod(A,trans(B));
	ublas::matrix<float> AAt_inv;
	svd_inv(AAt,AAt_inv);
	ublas::matrix<float> tmp = prod(AAt_inv,ABt);
	tmp = trans(tmp);
	Mat affine(Size(3,2),CV_32FC1,tmp.data().begin());
	Mat patch;
	warpAffine(img, patch, affine, size);
	return patch;
}

namespace boost{
	namespace serialization {
		template<class Archive> void serialize(Archive &ar, cv::Mat& mat, const unsigned int)
		{
			int cols, rows, type;
			bool continuous;

			if (Archive::is_saving::value) {
				cols = mat.cols; rows = mat.rows; type = mat.type();
				continuous = mat.isContinuous();
			}

			ar & cols & rows & type & continuous;

			if (Archive::is_loading::value)
				mat.create(rows, cols, type);

			if (continuous) {
				const unsigned int data_size = rows * cols * mat.elemSize();
				ar & boost::serialization::make_array(mat.ptr(), data_size);
			} else {
				const unsigned int row_size = cols*mat.elemSize();
				for (int i = 0; i < rows; i++) {
					ar & boost::serialization::make_array(mat.ptr(i), row_size);
				}
			}
		}
	}
}
