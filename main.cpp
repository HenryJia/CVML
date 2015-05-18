#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <lapackpp.h>

using namespace Eigen;
using namespace std;

double testEigen()
{
	auto start = chrono::steady_clock::now();
	MatrixXd a = MatrixXd::Random(300000, 80);
	MatrixXd b = MatrixXd::Random(80, 30);
	MatrixXd result(300000, 30);
	result = a * b;
	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}

double testLapack()
{
	auto start = chrono::steady_clock::now();
	LaGenMatDouble a = LaGenMatDouble::rand(300000, 80);
	LaGenMatDouble b = LaGenMatDouble::rand(80, 30);
	LaGenMatDouble result(300000, 30);
	Blas_Mat_Mat_Mult(a, b, result, 1.0, 0.0);
	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}

int main()
{
	cout << "Eigen Start" << endl;
	cout << testEigen() << " s" << endl;
	cout << "Lapack Start" << endl;
	cout << testLapack() << " s" << endl;
	getchar();
}
