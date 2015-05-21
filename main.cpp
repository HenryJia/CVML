#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <lapackpp.h>
#include <armadillo>

using namespace Eigen;
using namespace std;
using namespace arma;

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

double testArmadillo()
{
	auto start = chrono::steady_clock::now();
	mat a = randu<mat>(300000, 80);
	mat b = randu<mat>(80, 30);
	mat c = a * b;
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
	cout << "Armadillo Start" << endl;
	cout << testArmadillo() << " s" << endl;
	cout << "Finished, press enter to end" << endl;
	getchar();
}
