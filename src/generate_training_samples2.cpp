#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/scoped_ptr.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>
#include <caffe/util/format.hpp>
#include <caffe/util/rng.hpp>
#include <caffe/util/db.hpp>
#include "FeatureExtractor.h"

using namespace std;
using namespace boost;
using namespace boost::program_options;
using namespace boost::filesystem;
using namespace caffe;

void vectorToDatum(const ublas::vector<float> & v,const int label,Datum * datum);
void write2db(vector<boost::tuple<path,int> > & list,scoped_ptr<db::Transaction> & txn,scoped_ptr<db::DB> & db);

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
	
	//1)生成list
	int label = 0;
	vector<boost::tuple<path,int> > trainlist;
	vector<boost::tuple<path,int> > testlist;
	for(directory_iterator it(inputdir) ; it != directory_iterator() ; it++) {
		if(is_directory(it->path())) {
			//获得当前文件夹下面的文件列表
			vector<boost::tuple<path,int> > list;
			for(directory_iterator img_itr(it->path()) ; img_itr != directory_iterator() ; img_itr++)
				if(img_itr->path().extension().string() == ".jpg") list.push_back(boost::make_tuple(img_itr->path(),label));
			//对列表打乱顺序
			random_shuffle(list.begin(),list.end());
			//然后分别放到训练和测试集合列表
			testlist.insert(testlist.end(),list.begin(),list.begin() + 2);
			trainlist.insert(trainlist.end(),list.begin() + 2,list.end());
			label++;
		} // if 是文件夹
	}
	random_shuffle(trainlist.begin(),trainlist.end());
	random_shuffle(testlist.begin(),testlist.end());
	//2)生成lmdb
	scoped_ptr<db::DB> traindb(db::GetDB("lmdb"));
	traindb->Open((outputroot / "training").string(),db::NEW);
	scoped_ptr<db::Transaction> traintxn(traindb->NewTransaction());
	
	scoped_ptr<db::DB> testdb(db::GetDB("lmdb"));
	testdb->Open((outputroot / "testing").string(),db::NEW);
	scoped_ptr<db::Transaction> testtxn(testdb->NewTransaction());
	
	Datum datum;
#ifndef NDEBUG
	cout<<"生成训练集"<<endl;
#endif
	write2db(trainlist,traintxn,traindb);
#ifndef NDEBUG
	cout<<"生成测试集"<<endl;
#endif
	write2db(testlist,testtxn,testdb);
	//3)计算样本期望
	string cmd = "compute_image_mean " + (outputroot / "training").string() + " " + (outputroot / "face_recognizer.binaryproto").string();
	std::system(cmd.c_str());
	
	return EXIT_SUCCESS;
}

void vectorToDatum(const ublas::vector<float> & v,const int label,Datum * datum)
{
	datum->set_channels(v.size());
	datum->set_height(1);
	datum->set_width(1);
	datum->clear_data();
	datum->clear_float_data();
	datum->set_encoded(false);
	datum->set_label(label);
	for(int i = 0 ; i < v.size() ; i++) datum->add_float_data(v[i]);
}

void write2db(vector<boost::tuple<path,int> > & list,scoped_ptr<db::Transaction> & txn,scoped_ptr<db::DB> & db)
{
	FeatureExtractor & extractor = FeatureExtractor::get();
	
	Datum datum;
	int count = 0;
	for(vector<boost::tuple<path,int> >::iterator it = list.begin() ; it != list.end() ; it++) {
		Mat img = imread(get<0>(*it).string());
		if(img.empty()) {
			cout<<get<0>(*it).string()<<"无法打开！"<<endl;
			continue;
		}
		boost::tuple<bool,ublas::vector<float> > retVal = extractor(img);
		if(false == get<0>(retVal)) {
			cout<<get<0>(*it).string()<<"图片特征提取失败！"<<endl;
			continue;
		}
		string buffer;
		vectorToDatum(get<1>(retVal),get<1>(*it),&datum);
		datum.SerializeToString(&buffer);
		txn->Put(lexical_cast<string>(count),buffer);
		++count;
		if(count % 1000 == 0) {
#ifndef NDEBUG
			cout<<"生成"<<count<<"个样本"<<endl;
#endif
			txn->Commit();
			txn.reset(db->NewTransaction());
		}
	}
	if(count % 1000 != 0) {
#ifndef NDEBUG
		cout<<"生成"<<count<<"个样本"<<endl;
#endif
		txn->Commit();
	}
}
