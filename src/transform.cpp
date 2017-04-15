#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/scoped_ptr.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>
#include <caffe/util/format.hpp>
#include <caffe/util/rng.hpp>
#include <caffe/util/db.hpp>
#include <caffe/caffe.hpp>
#include "CCIPCA.h"

using namespace std;
using namespace boost;
using namespace boost::program_options;
using namespace boost::archive;
using namespace boost::filesystem;
using namespace caffe;

ublas::vector<float> loadMean(path file);
void updatePrincipalComponents(path lmdbpath,ublas::vector<float> mean,CCIPCA & ccipca);
void transform(path inputroot,path outputroot,ublas::matrix<float> eigvecs,ublas::vector<float> mean);
void vectorToDatum(const ublas::vector<float> & v,const int label,Datum * datum);

int main(int argc,char ** argv)
{
	options_description desc;
	string inputdir,outputdir,pc_file;
	desc.add_options()
		("help,h","打印当前使用方法")
		("input,i",value<string>(&inputdir),"deepid特征向量训练集")
		("output,o",value<string>(&outputdir),"输出训练集的路径")
		("param,p",value<string>(&pc_file)->default_value(""),"期望与主元参数文件");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || vm.count("help") || 1 != vm.count("input") || 1 != vm.count("output")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	path inputroot(inputdir);
	if(
		false == exists(inputroot / "training") || false == is_directory(inputroot / "training") ||
		false == exists(inputroot / "testing") || false == is_directory(inputroot / "testing") ||
		false == exists(inputroot / "face_recognizer.binaryproto") || true == is_directory(inputroot / "face_recognizer.binaryproto")
	) {
		cout<<"deepid特征向量训练集错误！"<<endl;
		return EXIT_FAILURE;
	}
	
	if(pc_file != "" && (false == exists(pc_file) || true == is_directory(pc_file))) {
		cout<<"期望与主元文件找不到！"<<endl;
		return EXIT_FAILURE;
	}
	
	path outputroot(outputdir);
	remove_all(outputroot);
	create_directory(outputroot);

	ublas::matrix<float> eigvecs;
	ublas::vector<float> mean;
	if(pc_file == "") {
		//1)计算主元
		CCIPCA ccipca(19200,150);
		mean = loadMean(inputroot / "face_recognizer.binaryproto");
#ifndef NDEBUG
		cout<<"用测试集更新主元"<<endl;
#endif
		updatePrincipalComponents(inputroot / "testing",mean,ccipca);
		//2)将所有样本映射到主元空间
		eigvecs = ccipca.getEigVecs();
		std::ofstream out("期望和主元.dat");
		text_oarchive oa(out);
		oa <<mean<< eigvecs;
		out.close();
	} else {
		cout<<"读取期望与主元..."<<endl;
		std::ifstream in(pc_file.c_str());
		text_iarchive ia(in);
		ia >> mean >> eigvecs;
	}
	transform(inputroot / "training",outputroot / "training",eigvecs,mean);
	transform(inputroot / "testing",outputroot / "testing",eigvecs,mean);
	
	return EXIT_SUCCESS;
}

ublas::vector<float> loadMean(path mean_file)
{
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.string().c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	
	ublas::vector<float> retVal(mean_blob.channels());
	float * data = mean_blob.mutable_cpu_data();
	std::copy(data,data + mean_blob.channels(), retVal.begin());
	return retVal;
}

void updatePrincipalComponents(path lmdbpath,ublas::vector<float> mean,CCIPCA & ccipca)
{
	scoped_ptr<db::DB> db(db::GetDB("lmdb"));
	db->Open(lmdbpath.string(),db::READ);
	scoped_ptr<db::Cursor> cursor(db->NewCursor());

	Datum datum;
#ifndef NDEBUG
	int count = 0;
#endif
	do {
		datum.ParseFromString(cursor->value());
		assert(datum.float_data_size());
		ublas::vector<float> v(datum.channels());
		for(int c = 0 ; c < datum.channels() ; c++) v(c) = datum.float_data(c);
		ccipca.update(v - mean);
#ifndef NDEBUG
		count++;
		if(0 == count % 10) {
			cout<<"已经读入"<<count<<"个样本"<<endl;
		}
#endif
		cursor->Next();
	} while(cursor->valid());
}

void transform(path inputroot,path outputroot,ublas::matrix<float> eigvecs,ublas::vector<float> mean)
{
	scoped_ptr<db::DB> inputdb(db::GetDB("lmdb"));
	inputdb->Open(inputroot.string(),db::READ);
	scoped_ptr<db::Cursor> inputcursor(inputdb->NewCursor());
	
	scoped_ptr<db::DB> outputdb(db::GetDB("lmdb"));
	outputdb->Open(outputroot.string(),db::NEW);
	scoped_ptr<db::Transaction> outputtxn(outputdb->NewTransaction());
	
	ublas::matrix<float> buffer(eigvecs.size1(),1000);
	vector<int> labels(1000);
	int count = 0;
	int index = 0;
	Datum datum;
	do {
		if(1000 == count) {
			//映射到主元空间并且写入lmdb
			ublas::matrix<float> transformed_buffer = prod(trans(eigvecs),buffer);
			assert(1000 == count);
			assert(transformed_buffer.size2() == count);
			for(int i = 0 ; i < count ; i++) {
				ublas::matrix_column<ublas::matrix<float> > mc(transformed_buffer,i);
				vectorToDatum(mc,labels[i],&datum);
				string output;
				datum.SerializeToString(&output);
				outputtxn->Put(lexical_cast<string>(index++),output);
			}
			outputtxn->Commit();
			outputtxn.reset(outputdb->NewTransaction());
			count = 0;
		}
		datum.ParseFromString(inputcursor->value());
		assert(datum.float_data_size());
		ublas::matrix_column<ublas::matrix<float> > mc(buffer,count);
		for(int c = 0 ; c < datum.channels() ; c++) mc(c) = datum.float_data(c);
		mc -= mean;
		labels[count] = datum.label();
		count++;
		inputcursor->Next();
	} while(inputcursor->valid());
	if(count) {
		//映射到主元空间并且写入lmdb
		ublas::matrix<float> transformed_buffer = prod(trans(eigvecs),buffer);
		assert(transformed_buffer.size2() == buffer.size2());
		for(int i = 0 ; i < count ; i++) {
			ublas::matrix_column<ublas::matrix<float> > mc(transformed_buffer,i);
			vectorToDatum(mc,labels[i],&datum);
			string output;
			datum.SerializeToString(&output);
			outputtxn->Put(lexical_cast<string>(index++),output);
		}
		outputtxn->Commit();
		count = 0;		
	}
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
