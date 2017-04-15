#include "CCIPCA.h"

CCIPCA::CCIPCA(int vector_dim,int eigenvector_num,bool c):count(0),centralized(c)

{
	if(eigenvector_num > vector_dim) throw logic_error("主元向量个数不能超过向量维度！");
	V = ublas::zero_matrix<float>(vector_dim, eigenvector_num);
	//如果未来输入的样本没有去掉期望，那么这里就维护一个期望
	if(false == centralized) mean = ublas::zero_vector<float>(vector_dim);
}

CCIPCA::~CCIPCA()
{
}

void CCIPCA::update(ublas::vector<float> x)
{
	if(x.size() != V.size1()) throw logic_error("向量维度与预设维度不相符！");
	//中心化，并更新期望
	if(false == centralized) {
		x -= mean;
		mean = (static_cast<float>(count) / (count + 1)) * mean + (1.0 / (count + 1)) * x;
	}

	//更新主元
	for(int i = 0 ; i < V.size2() ; i++) {
		//1)更新主元
		ublas::matrix_column<ublas::matrix<float> > v(V,i);
		if(count == i) {
			//如果v还是全0的向量，那么就设置去掉之前主元分量的x为当前主元
#ifndef NDEBUG
//			assert(v == ublas::zero_vector<float>(V.size1()));
#endif
			v = x;
			//如果x被设置为主元，那么主元以外分量为0，就不用继续更新了
			break;
		} else {
			//如果v不是全0的向量，那么就更新当前主元
			ublas::vector<float> u = convert2eigvec(v);
			//更新V
			v = (static_cast<float>(count) / (count + 1)) * v 
				+ (1.0 / (count + 1)) * prod(outer_prod(x,x),u);
		}
		//2)从x中去掉当前更新后主元的分量
		ublas::vector<float> u = convert2eigvec(v);
		//从x中去掉当前主元成分
		x -= inner_prod(x,u) * u;
	}
	count++;
}

ublas::vector<float> CCIPCA::getEigVals()
{
	ublas::vector<float> retVal(V.size2());
	for(int i = 0 ; i < V.size2() ; i++) {
		ublas::matrix_column<ublas::matrix<float> > v(V,i);
		retVal(i) = convert2eigval(v);
	}
	return retVal;
}

ublas::matrix<float> CCIPCA::getEigVecs()
{
	ublas::matrix<float> retVal(V.size1(),V.size2());
	for(int i = 0 ; i < V.size2() ; i++) {
		ublas::matrix_column<ublas::matrix<float> > v(V,i);
		ublas::matrix_column<ublas::matrix<float> > o(retVal,i);
		o = convert2eigvec(v);
	}
	return retVal;
}

ublas::vector<float> CCIPCA::getMean()
{
	return mean;
}

ublas::vector<float> CCIPCA::convert2eigvec(ublas::matrix_column<ublas::matrix<float> > v)
{
	float norm = convert2eigval(v);
	norm += (0 == norm)?numeric_limits<float>::min():0;
	ublas::vector<float> u = v / norm;
	return u;
}

float CCIPCA::convert2eigval(ublas::matrix_column<ublas::matrix<float> > v)
{
	return sqrt(inner_prod(v,v));	
}
