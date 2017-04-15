#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/tuple/tuple.hpp>

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace boost::program_options;

static const string regions[] ={
        "1_F","1_E","1_EN","1_N","1_NM",
        "2_LE","2_RE","2_N","2_LM","2_RM"
};

static const string types[] = {"GRAY","RGB"};
//三种尺度，patch的高，patch的宽，两眼距离占宽的比例
static const boost::tuple<float,float> scales[] = {boost::make_tuple(0.4f,0.64f),boost::make_tuple(0.3f,0.48f),boost::make_tuple(0.24f,0.39f)};

int main(int argc,char ** argv)
{
	options_description desc;
        string inputdir,outputdir;
        desc.add_options()
                ("help,h","打印当前使用方法")
                ("input,i",value<string>(&inputdir),"训练集文件夹路径")
                ("output,o",value<string>(&outputdir),"输出训练结果文件路径");
        variables_map vm;
        store(parse_command_line(argc,argv,desc),vm);
        notify(vm);

	if(1 == argc || vm.count("help") || 1 != vm.count("input") || 1 != vm.count("output")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	if(false == exists(inputdir) || false == is_directory(inputdir)) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	path outputroot(outputdir);
	if(false == exists(outputroot)) create_directory(outputroot);
	path inputroot(inputdir);
	regex expression("deepid_iter_([0-9]+)\\.caffemodel");
        for(int r = 0 ; r < sizeof(regions)/sizeof(string) ; r++)
                for(int t = 0 ; t < sizeof(types) / sizeof(string) ; t++)
                        for(int s = 0 ; s < sizeof(scales) / sizeof(boost::tuple<float,float>) ; s++) {
				string filename = regions[r] + "_" + types[t] + "_" + lexical_cast<string>(s);
				path dir = inputroot / filename;
				if(false == exists(dir) || false == is_directory(dir))
					throw runtime_error("输入文件夹结构不正确！");
				int maxid = 0;
				path caffemodel;
				for(directory_iterator itr(dir) ; itr != directory_iterator() ; itr++) {
					string modelname = itr->path().filename().string();
					cmatch what;
					if(regex_match(modelname.c_str(),what,expression)) {
						string id(what[1].first,what[1].second);
						if(lexical_cast<int>(id) > maxid) {
							maxid = lexical_cast<int>(id);
							caffemodel = itr->path();
						}
					}
				}
				if("" == caffemodel.string()) throw runtime_error(filename + "模型还没有训练！");
				//拷贝模型和期望文件
				copy_file(caffemodel,outputroot / (filename + ".caffemodel"));
				copy_file(inputroot / filename / "deepid_mean.binaryproto",outputroot / (filename + ".binaryproto"));
			}
	return EXIT_SUCCESS;
}
