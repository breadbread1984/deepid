#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <set>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "FeatureExtractor.h"

using namespace std;
using namespace boost::program_options;
using namespace boost::filesystem;
namespace ublas = boost::numeric::ublas;

vector<boost::tuple<string,string,bool> > loadList(std::ifstream & list)
{
	vector<boost::tuple<string,string,bool> > retVal;
	string line;
	int n_set,n_num;
	getline(list,line);
	stringstream sstr;
	sstr<<line;
	sstr>>n_set>>n_num;
	for(int i = 0 ; i < n_set ; i++) {
		for(int j = 0 ; j < n_num ; j++) {
			getline(list,line);
			stringstream sstr;
			sstr << line;
			string name; int id1,id2;
			sstr>>name >>id1>>id2;
			ostringstream ss1,ss2;
			ss1<<setw(4)<<setfill('0')<<id1;
			ss2<<setw(4)<<setfill('0')<<id2;
			string file1 = name + "/" + name + "_" + ss1.str() + ".jpg";
			string file2 = name + "/" + name + "_" + ss2.str() + ".jpg";
			retVal.push_back(boost::make_tuple(file1,file2,true));
		}
		for(int j = 0 ; j < n_num ; j++) {
			getline(list,line);
			stringstream sstr;
			sstr << line;
			string name1,name2; int id1,id2;
			sstr >>name1 >> id1>>name2>>id2;
			ostringstream ss1,ss2;
			ss1<<setw(4)<<setfill('0')<<id1;
			ss2<<setw(4)<<setfill('0')<<id2;
			string file1 = name1 + "/" + name1 + "_" + ss1.str() + ".jpg";
			string file2 = name2 + "/" + name2 + "_" + ss2.str() + ".jpg";
			retVal.push_back(boost::make_tuple(file1,file2,false));
		}
	}
	return retVal;
}

void process(path file1,path file2,vector<float> & dists)
{
	Mat img1 = imread(file1.string());
	Mat img2 = imread(file2.string());
	FeatureExtractor & extractor = FeatureExtractor::get();
	boost::tuple<bool,ublas::vector<float> > fv1 = extractor(img1);
	if(false == get<0>(fv1)) {
		cout<<file1.string()<<endl;
		return;
	}
	boost::tuple<bool,ublas::vector<float> > fv2 = extractor(img2);
	if(false == get<0>(fv2)) {
		cout<<file2.string()<<endl;
		return;
	}
	ublas::vector<float> diff = get<1>(fv1) - get<1>(fv2);
	float res = sqrt(inner_prod(diff,diff));
	dists.push_back(res);
}
							
int main(int argc,char ** argv)
{
	options_description desc;
	string inputdir;
	string listfile;
	desc.add_options()
		("help,h","打印当前使用方法")
		("input,i",value<string>(&inputdir),"LFW文件夹路径")
		("pair,p",value<string>(&listfile),"LFW验证列表文件路径");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || vm.count("help") || 1 != vm.count("input") || 1 != vm.count("pair")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	path inputroot(inputdir);
	if(false == exists(inputroot) || false == is_directory(inputroot)) {
		cout<<"LFW文件夹不存在！"<<endl;
		return EXIT_FAILURE;
	}
	std::ifstream verifpair(listfile);
	if(false == verifpair.is_open()) {
		cout<<"LFW验证列表文件无法打开！"<<endl;
		return EXIT_FAILURE;
	}
	vector<boost::tuple<string,string,bool> > list = loadList(verifpair);
	//1)计算样本之间距离
	vector<float> pos_dists,neg_dists;
	cout<<"计算样本之间距离"<<endl;
#ifndef NDEBUG
	int count = 0;
#endif
	for(vector<boost::tuple<string,string,bool> >::iterator it = list.begin() ; it != list.end() ; it++) {
		path file1 = inputroot / get<0>(*it);
		path file2 = inputroot / get<1>(*it);
		process(file1,file2,get<2>(*it)?pos_dists:neg_dists);
		count++;
		if(count % 1000 == 0) cout<<"已经处理了"<<count<<"个图像对"<<endl;
	}
	//2)计算roc曲线
	cout<<"计算roc曲线"<<endl;
	set<float> thresholds;
	thresholds.insert(pos_dists.begin(),pos_dists.end());
	thresholds.insert(neg_dists.begin(),neg_dists.end());
	std::ofstream out("roc.txt");
	map<float,boost::tuple<float,float> > roc;
	float eer = 0;
	float prev_truepos_rate, prev_falsepos_rate;
	for(set<float>::iterator it = thresholds.begin() ; it != thresholds.end() ; it++) {
		float threshold = *it;
		int truepos = 0, falsepos = 0;
		int falseneg = 0, trueneg = 0;
		for(int i = 0 ; i < pos_dists.size() ; i++) if(pos_dists[i] < threshold) truepos++; else falseneg++;
		for(int i = 0 ; i < neg_dists.size() ; i++) if(neg_dists[i] > threshold) trueneg++; else falsepos++;
		float truepos_rate = static_cast<float>(truepos) / (truepos + falseneg);
		float falsepos_rate = static_cast<float>(falsepos) / (falsepos + trueneg);
		roc.insert(make_pair(*it,boost::make_tuple(falsepos_rate,truepos_rate)));
		if(it != thresholds.begin()) eer += 0.5 * (falsepos_rate - prev_falsepos_rate) * (truepos_rate + prev_truepos_rate);
		prev_truepos_rate = truepos_rate; prev_falsepos_rate = falsepos_rate;
		out<<falsepos_rate<<" "<<truepos_rate<<" "<<*it<<endl;
	}
	cout<<"eer = "<<eer<<endl;
		
	return EXIT_SUCCESS;
}
