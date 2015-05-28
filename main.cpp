#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <lapackpp.h>
#include <armadillo>
#include <fstream>
#include <iomanip>

#include "cvnn.h"

using namespace Eigen;
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

double testOpenCV()
{
	auto start = chrono::steady_clock::now();
	cv::Mat a(300000, 80, CV_64F);
	cv::Mat b(80, 30, CV_64F);
	cv::randu(a, 0.0, 1.0);
	cv::randu(b, 0.0, 1.0);
	cv::Mat c = a * b;
	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}

int main()
{
	cout << "Matrix Multiplication Tests" << endl;
	cout << "Eigen Start" << endl;
	cout << testEigen() << " s" << endl;
	cout << "Lapack Start" << endl;
	cout << testLapack() << " s" << endl;
	cout << "Armadillo Start" << endl;
	cout << testArmadillo() << " s" << endl;
	cout << "OpenCV Start" << endl;
	cout << testOpenCV() << " s" << endl;

	cout << "Neural Network Gradient Descent Tests" << endl;
	cvnn nn;
	cout << "Read CSV" << endl;
	double csvTime;
	vector<vector<double>> xVec = nn.readCSV("trainP7_1.csv", false, csvTime);
	vector<vector<double>> yVec = nn.readCSV("trainYP2_1.csv", false, csvTime);
	cout << csvTime << " s" << endl;

	for (size_t i = 0; i < xVec.size(); i++)
	{
		xVec[i].erase(xVec[i].begin() + 1);
	}

	nn.setData(xVec, yVec);
	nn.setAlpha(0.015);
	nn.setIters(100);
	nn.setClassify(false);
	nn.setThreads(4);
	vector<int> layers = {10, 40, 160, 1};
	nn.setLayers(layers);
	double singleTime = nn.train();
	nn.setLayers(layers);
	double concurrentTime = nn.trainConcurrent();
	cout << "Single thread " << singleTime << " s" << endl;
	cout << "Concurrent " << concurrentTime << " s" << endl;
	cout << "Concurrent forward propagation is " << singleTime - concurrentTime << " s faster (" 
		<< concurrentTime / singleTime * 100 << "%)" << endl;
	cout << "Finished, press enter to end" << endl;
	getchar();
}
