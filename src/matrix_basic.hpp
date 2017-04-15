#ifndef MATRIX_BASIC_HPP
#define MATRIX_BASIC_HPP

#include <cstddef>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm> 
#include <vector>
#include <complex>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/traits.hpp>
//numeric bindings headers
#include <boost/numeric/bindings/lapack/gesvd.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/bindings/traits/std_vector.hpp>
#include <boost/numeric/bindings/lapack/posv.hpp>
#include <boost/numeric/bindings/lapack/geqrf.hpp>
#include <boost/numeric/bindings/lapack/ormqr.hpp>
#include <boost/numeric/bindings/lapack/orgqr.hpp>
#include <boost/numeric/bindings/lapack/geev.hpp>

using namespace std;
namespace ublas = boost::numeric::ublas;
namespace lapack = boost::numeric::bindings::lapack;

/** General matrix inversion routine.
* It uses lu_factorize and lu_substitute in uBLAS to invert a matrix
*/
template<class M1, class M2>
void lu_inv(M1 const& m, M2& inv)
{
	assert(m.size1() == m.size2());
	if(inv.size1() != m.size1() || inv.size2() != m.size2())
		inv = ublas::matrix<double>(m.size1(),m.size2());

	// create a working copy of the input
	ublas::matrix<double> mLu(m);

	ublas::permutation_matrix<size_t> pm(mLu.size1());

	// perform LU-factorization
	int res = ublas::lu_factorize(mLu,pm);
	if(res)
		throw runtime_error("determinant matrix equals zero!");

	// create identity matrix of "inverse"
	inv.assign(ublas::identity_matrix<double>(m.size1()));

	// backsubstitute to get the inverse
	ublas::lu_substitute(mLu, pm, inv);
}

/** General matrix determinant.
 * It uses lu_factorize in uBLAS.
 */
template<class M> 
double lu_det(M const& m)
{
	assert(m.size1() == m.size2());

	// create a working copy of the input
	ublas::matrix<double> mLu(m);
	ublas::permutation_matrix<std::size_t> pivots(m.size1());

	ublas::lu_factorize(mLu, pivots);

	double det = 1.0;

	for (std::size_t i=0; i < pivots.size(); ++i) {
		if (pivots(i) != i)
			det *= -1.0;
		det *= mLu(i,i);
	}
	return det;
}
/** Calculate matrix trace.
 */
template<class M>
double trace(M const & m)
{
	assert(m.size1() == m.size2());
	double tr = 0;
	for(int i = 0 ; i < m.size1() ; i++)
		tr += m(i,i);
	return tr;
}

//下面是通过boost numeric bindings实现的功能函数
/** 通过SVD分解计算矩阵的逆
 */
template<class M1, class M2>
void svd_inv(M1 const & A, M2 & inv)
{
	ublas::matrix<double,ublas::column_major> M = A;

	size_t m = M.size1();
	size_t n = M.size2();
	size_t minmn = m < n ? m : n;

	ublas::vector<double> s(minmn);
	ublas::matrix<double,ublas::column_major> u(m,m);
	ublas::matrix<double,ublas::column_major> vt(n,n);
	vector<double> w(lapack::gesvd_work('O','A','A',M));

	lapack::gesvd('A','A',M,s,u,vt,w);

	ublas::matrix<double,ublas::column_major> _1divs = ublas::zero_matrix<double> (n,m);
	for(int i = 0 ; i < s.size() ; i++)
		if(s(i)) _1divs(i,i) = 1.0 / s(i);

	inv = prod(trans(vt),_1divs);
	inv = prod(inv,trans(u));
}

/** 矩阵SVD分解
 */
template<class M1,class M2,class M3,class M4>
void svd_decomp(M1 const & A,M2 & U,M3 & S,M4 & V)
{
	ublas::matrix<double,ublas::column_major> M = A;

	size_t m = M.size1();
	size_t n = M.size2();
	size_t minmn = m < n ? m : n;

	ublas::vector<double> s(minmn);
	ublas::matrix<double,ublas::column_major> u(m,m);
	ublas::matrix<double,ublas::column_major> vt(n,n);
	vector<double> w(lapack::gesvd_work('O','A','A',M));

	lapack::gesvd('A','A',M,s,u,vt,w);

	U = u;
	S = s;
	V = trans(vt);
}

/** 矩阵SVD分解(经济型)
 */
template<class M1,class M2,class M3,class M4>
void svd_decomp2(M1 const & A,M2 & U,M3 & S,M4 & V)
{
	svd_decomp(A,U,S,V);
	if(A.size1() <= A.size2())
		return;
	else
		U = ublas::matrix_range<ublas::matrix<double> >(
			U,
			ublas::range(0,A.size1()),
			ublas::range(0,A.size2())
		);
}

/** 矩阵Cholesky分解
 */
template<class M1,class M2>
void chol_decomp(M1 const & A,M2 & L)
{
#ifndef NDEBUG
	assert(A.size1() == A.size2());
#endif
	ublas::matrix<double,ublas::column_major> M = A;
	ublas::matrix<double,ublas::column_major> res(A.size1(),A.size2());
	lapack::potrf('U',M);
	for(int h = 0 ; h < A.size1() ; h++) {
		for(int w = 0 ; w < h ; w++)
			res(h,w) = 0;
		for(int w = h ; w < A.size2() ; w++)
			res(h,w) = M(h,w);
	}
	L = res;
}

/** 矩阵QR分解
 */
template<class M1,class M2,class M3>
void qr_decomp(M1 const & A,M2 & Q,M3 & R)
{
	Q = ublas::zero_matrix<double>(A.size1(),A.size1());
	R = ublas::zero_matrix<double>(A.size1(),A.size2());

	if(A.size1() >= A.size2()) {
		ublas::matrix_range<ublas::matrix<double> > subQ(
			Q,
			ublas::range(0,A.size1()),
			ublas::range(0,A.size2())
		);
		ublas::matrix_range<ublas::matrix<double> > subR(
			R,
			ublas::range(0,A.size2()),
			ublas::range(0,A.size2())
		);
		ublas::matrix<double> q,r;
		qr_decomp2(A,q,r);
		subQ = q; subR = r;
	} else
		qr_decomp2(A,Q,R);
}

/** 矩阵QR分解(经济型)
 */
template<class M1,class M2,class M3>
void qr_decomp2(M1 const & A,M2 & Q,M3 & R)
{
	ublas::matrix<double,ublas::column_major> M = A;
	ublas::vector<double> tau(min(M.size1(),M.size2()));

	lapack::geqrf(M,tau);
	ublas::matrix_range<ublas::matrix<double,ublas::column_major> > subM(
		M,
		ublas::range(0,(A.size1() < A.size2())?A.size1():A.size2()),
		ublas::range(0,A.size2())
	);
	R = ublas::triangular_adaptor<
		ublas::matrix_range<ublas::matrix<double,ublas::column_major> >,
		ublas::upper
	> (subM);
	if(A.size1() < A.size2())
		M = ublas::matrix_range<ublas::matrix<double,ublas::column_major> > (
			M,
			ublas::range(0,A.size1()),ublas::range(0,A.size1())
		);
	lapack::orgqr(M,tau,lapack::optimal_workspace());
	Q = M;
	R = ublas::matrix_range<ublas::matrix<double> > (
		R,
		ublas::range(0,Q.size2()),ublas::range(0,R.size2())
	);
}

/** 计算特征值和特征向量
 */
template<class M1,class M2,class M3>
void eig(M1 const & A,M2 & v,M3 & e)
{
	//NOTICE:v和e一定不是实数而是complex<>类型的。因为特征值很有可能是虚数
#ifndef NDEBUG
	assert(A.size1() == A.size2());
#endif
	ublas::matrix<complex<double>,ublas::column_major> B(A.size1(),A.size2());
	B = A;
	ublas::vector<complex<double> > w(A.size1());
	ublas::matrix<complex<double>,ublas::column_major> Vl(A.size1(),A.size2()),Vr(A.size1(),A.size2());
	lapack::geev(B,w,&Vl,&Vr,lapack::optimal_workspace());
	e = w;
	v = Vr;
}

/** 矩阵的exp运算
 */
template<typename MATRIX> 
MATRIX expm(
	const MATRIX &H, 
	typename ublas::type_traits<typename MATRIX::value_type>::real_type t = 1.0, 
	const int p = 6
){
	typedef typename MATRIX::value_type value_type;
	typedef typename MATRIX::size_type size_type;
	typedef typename ublas::type_traits<value_type>::real_type real_value_type;
	assert(H.size1() == H.size2());
	assert(p >= 1);
	const size_type n = H.size1();
	const ublas::identity_matrix<value_type> I(n);
	ublas::matrix<value_type> U(n,n),H2(n,n),P(n,n),Q(n,n);
	real_value_type norm = 0.0;
	// Calcuate Pade coefficients
	ublas::vector<real_value_type> c(p+1);
	c(0)=1;  
	for(size_type i = 0; i < p; ++i) 
		c(i+1) = c(i) * ((p - i)/((i + 1.0) * (2.0 * p - i)));
	// Calcuate the infinty norm of H, which is defined as the largest row sum of a matrix
	for(size_type i=0; i<n; ++i) {
		real_value_type temp = 0.0;
		for(size_type j = 0; j < n; j++)
			temp += std::abs(H(i, j)); 
		norm = t * std::max<real_value_type>(norm, temp);
	}
	// If norm = 0, and all H elements are not NaN or infinity but zero, 
	// then U should be identity.
	if (norm == 0.0) {
		bool all_H_are_zero = true;
		for(size_type i = 0; i < n; i++)
			for(size_type j = 0; j < n; j++)
				if( H(i,j) != value_type(0.0) ) 
					all_H_are_zero = false; 
		if( all_H_are_zero == true ) return I;
		// Some error happens, H has elements which are NaN or infinity. 
		std::cerr<<"Null input error in the template expm_pad.\n";
		std::cout << "Null INPUT : " << H <<"\n";
		exit(0);
	}
	// Scaling, seek s such that || H*2^(-s) || < 1/2, and set scale = 2^(-s)
	int s = 0;
	real_value_type scale = 1.0;
	if(norm > 0.5) {
		s = std::max<int>(0, static_cast<int>((log(norm) / log(2.0) + 2.0)));
		scale /= real_value_type(std::pow(2.0, s));
		U.assign((scale * t) * H); // Here U is used as temp value due to that H is const
	}
	else
		U.assign(H);

	// Horner evaluation of the irreducible fraction, see the following ref above.
	// Initialise P (numerator) and Q (denominator) 
	H2.assign( prod(U, U) );
	Q.assign( c(p)*I );
	P.assign( c(p-1)*I );
	size_type odd = 1;
	for( size_type k = p - 1; k > 0; --k) {
		( odd == 1 ) ?
			( Q = ( prod(Q, H2) + c(k-1) * I ) ) :
			( P = ( prod(P, H2) + c(k-1) * I ) ) ;
		odd = 1 - odd;
	}
	( odd == 1 ) ? ( Q = prod(Q, U) ) : ( P = prod(P, U) );
	Q -= P;
	// In origine expokit package, they use lapack ZGESV to obtain inverse matrix,
	// and in that ZGESV routine, it uses LU decomposition for obtaing inverse matrix.
	// Since in ublas, there is no matrix inversion template, I simply use the build-in
	// LU decompostion package in ublas, and back substitute by myself.

	// Implement Matrix Inversion
	ublas::permutation_matrix<size_type> pm(n); 
	int res = ublas::lu_factorize(Q, pm);
	if( res != 0) {
		std::cerr << "Matrix inversion error in the template expm.\n";
		exit(0);
	}
	// H2 is not needed anymore, so it is temporary used as identity matrix for substituting.
	H2.assign(I); 
	ublas::lu_substitute(Q, pm, H2); 
	(odd == 1) ? 
		( U.assign( -(I + real_value_type(2.0) * prod(H2, P))) ):
		( U.assign(   I + real_value_type(2.0) * prod(H2, P) ) );
	// Squaring 
	for(size_type i = 0; i < s; ++i)
		U = (prod(U,U));
	return U;
}

#endif
