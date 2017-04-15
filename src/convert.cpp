#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/thread/thread.hpp>

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace boost::program_options;

//F是39x31图片
//其他都是31x31图片
static const string regions[] ={
	"1_F","1_E","1_EN","1_N","1_NM",
	"2_LE","2_RE","2_N","2_LM","2_RM"
};

static const string types[] = {"GRAY","RGB"};
//三种尺度，patch的高，patch的宽，两眼距离占宽的比例
static const boost::tuple<float,float> scales[] = {boost::make_tuple(0.4f,0.64f),boost::make_tuple(0.3f,0.48f),boost::make_tuple(0.24f,0.39f)};

void process(path input,path output,bool rgb = true);
void start_converting(string listfile,string outputpath,bool rgb);

int main(int argc,char ** argv)
{
	options_description desc;
	string inputdir,outputdir;
	desc.add_options()
		("help,h","打印当前使用方法")
		("input,i",value<string>(&inputdir),"训练集文件夹路径")
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
	
	path inputroot(inputdir);
	
	vector<boost::shared_ptr<boost::thread> > handlers;
	for(int r = 0 ; r < sizeof(regions)/sizeof(string) ; r++)
		for(int t = 0 ; t < sizeof(types) / sizeof(string) ; t++)
			for(int s = 0 ; s < sizeof(scales) / sizeof(boost::tuple<float,float>) ; s++) {
				string filename = regions[r] + "_" + types[t] + "_" + lexical_cast<string>(s);
				path curoutputdir = outputroot / filename;
				create_directory(curoutputdir);
				path curinputdir = inputroot / filename;
				if(false == exists(curinputdir) || false == is_directory(curinputdir)) {
					cout<<filename<<"文件夹不存在！"<<endl;
					throw runtime_error(filename + "文件夹不存在！");
				}
#if 0
				handlers.push_back(boost::shared_ptr<boost::thread> (
					new boost::thread(boost::bind(::process,curinputdir,curoutputdir,types[t] == "RGB"))
				));
#else
				process(curinputdir,curoutputdir,types[t] == "RGB");
#endif
			}
	if(handlers.size()) {
		for(
			vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ;
			it != handlers.end() ;
			it++
		) (*it)->join();
		handlers.clear();
	}
}

void process(path input,path output,bool rgb)
{
	//outputdir已经存在的文件夹
	path training = output / "training";
	path testing = output / "testing";
	std::ofstream train_out((output / "trainlist.txt").string().c_str());
	std::ofstream test_out((output / "testlist.txt").string().c_str());
	int label = -1;
	for(directory_iterator it(input) ; it != directory_iterator() ; it++) {
		if(is_directory(it->path())) {
			//文件夹的名字就是label
			vector<string> pics;
			//对文件夹下面的所有图片保存到lmdb
			for(directory_iterator img_itr(it->path()) ; img_itr != directory_iterator() ; img_itr++)
				if(img_itr->path().extension().string() == ".jpg") pics.push_back(absolute(img_itr->path()).string());
			//如果当前文件夹下面的图片少于50张就不处理
			label++;
			//在pics集合中采样50张图片
			random_shuffle(pics.begin(),pics.end());
			int i;
			for(i = 0 ; i < 2 ; i++) { 
				//前两个同时放入训练和测试集
				test_out<<pics[i]<<" "<<label<<endl;
			}
			for( ; i < pics.size() ; i++) 
				train_out << pics[i] <<" " <<label<<endl;
		} //if 是文件夹
	} //end for 每个文件夹
	train_out.close();
	test_out.close();
	
	vector<boost::shared_ptr<boost::thread> > handlers;
	handlers.push_back(boost::shared_ptr<boost::thread> (
		new boost::thread(boost::bind(::start_converting,(output / "trainlist.txt").string(),training.string(),rgb))
	));
	handlers.push_back(boost::shared_ptr<boost::thread> (
		new boost::thread(boost::bind(::start_converting,(output / "testlist.txt").string(),testing.string(),rgb))
	));
	for(
		vector<boost::shared_ptr<boost::thread> >::iterator it = handlers.begin() ;
		it != handlers.end() ;
		it++
	) (*it)->join();
	handlers.clear();
	
	string cmd = "compute_image_mean " + training.string() + " " +  (output / "deepid_mean.binaryproto").string();
	std::system(cmd.c_str());
}

void start_converting(string listfile,string outputpath,bool rgb)
{	
	vector<string> args;
	args.push_back("convert_imageset");
	if(false == rgb)	args.push_back("-gray");		//如果注释这一句就按照三通道图片创建训练集
	args.push_back("-shuffle");
	args.push_back("-backend");
	args.push_back("lmdb");
	args.push_back("/");
	args.push_back(listfile);
	args.push_back(outputpath);
	
	stringstream cmd;
	for(vector<string>::iterator it = args.begin() ; it != args.end() ; it++) {
		cmd<<*it<<" ";
	}
	
	std::system(cmd.str().c_str());
}
