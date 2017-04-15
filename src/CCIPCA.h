#ifndef CCIPCA_H
#define CCIPCA_H

#include <cmath>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>

using namespace std;
namespace ublas = boost::numeric::ublas;

class CCIPCA {
	ublas::matrix<float> V;
	bool centralized;
	ublas::vector<float> mean;
	int count; //更新样本的数目
protected:
	ublas::vector<float> convert2eigvec(ublas::matrix_column<ublas::matrix<float> > v);
	float convert2eigval(ublas::matrix_column<ublas::matrix<float> > v);
public:
	CCIPCA(int vector_dim,int eigenvector_num,bool centralized = true);
	virtual ~CCIPCA();
	void update(ublas::vector<float> v);
	ublas::vector<float> getEigVals();
	ublas::matrix<float> getEigVecs();
	ublas::vector<float> getMean();
};

#endif
