#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <lapackpp.h>
#include <armadillo>
#include <fstream>
#include <iomanip>

#include "datatools.h"
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

double testOpenCVNN(cv::Mat x, cv::Mat y, vector<double> &J, double alpha = 0.015, int iters = 100)
{
	auto start = chrono::steady_clock::now();

	int m = x.size().height;

	cv::hconcat(cv::Mat(m, 1, CV_64F, 1), x, x);

	cv::Mat theta1 = randInitialiseWeights(10, 40);
	cv::Mat theta2 = randInitialiseWeights(40, 160);
	cv::Mat theta3 = randInitialiseWeights(160, 1);

	cv::Mat trans;
	cv::Mat product;

	cv::Mat z2;
	cv::Mat a2;
	cv::Mat z3;
	cv::Mat a3;
	cv::Mat z4;
	cv::Mat a4;

	cv::Mat delta4;
	cv::Mat delta3;
	cv::Mat delta2;

	cv::Mat Delta3;
	cv::Mat Delta2;
	cv::Mat Delta1;

	cv::Mat theta3Grad;
	cv::Mat theta2Grad;
	cv::Mat theta1Grad;

	for(int i = 0; i < iters; i++)
	{
		//Forward propagation
		z2 = x * theta1;
		a2 = sigmoid(z2);

		cv::hconcat(cv::Mat(a2.size().height, 1, CV_64F, 1), a2, a2);
		z3 = a2 * theta2;
		a3 = sigmoid(z3);

		cv::hconcat(cv::Mat(a3.size().height, 1, CV_64F, 1), a3, a3);
		z4 = a3 * theta3;
		a4 = z4;

		//Calculate small delta
		delta4 = a4 - y;

		cv::transpose(theta3, trans);
		product = delta4 * trans;
		delta3 = product.colRange(0, product.size().width - 1).mul(sigmoidGradient(z3));

		cv::transpose(theta2, trans);
		product = delta3 * trans;
		delta2 = product.colRange(0, product.size().width - 1).mul(sigmoidGradient(z2));

		//Calculate cost
		J.push_back(sum(delta4.mul(delta4))[0] / (2 * m));
		cout << "Iteration " << i << ": " << "Cost: " << J[i] << endl;

		//Accumulate small delta and calculate big delta which is the partial derivatives
		cv::transpose(delta4, trans);
		Delta3 = trans * a3;

		cv::transpose(delta3, trans);
		Delta2 = trans * a2;

		cv::transpose(delta2, trans);
		Delta1 = trans * x;

		//Finish off the calculation
		cv::transpose(Delta3, trans);
		theta3Grad = trans / m;

		cv::transpose(Delta2, trans);
		theta2Grad = trans / m;

		cv::transpose(Delta1, trans);
		theta1Grad = trans / m;

		//Gradient descent
		theta3 = theta3 - alpha * theta3Grad;
		theta2 = theta2 - alpha * theta2Grad;
		theta1 = theta1 - alpha * theta1Grad;
	}
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
	cout << "Read CSV" << endl;
	double csvTime;
	vector<vector<double>> xVec = readCSV("trainP7_1.csv", false, csvTime);
	vector<vector<double>> yVec = readCSV("trainYP2_1.csv", false, csvTime);
	cout << csvTime << " s" << endl;

	for (size_t i = 0; i < xVec.size(); i++)
	{
		xVec[i].erase(xVec[i].begin() + 1);
	}

	cout << "Train" << endl;
	cv::Mat x = vector2dToMat(xVec);
	cv::Mat y = vector2dToMat(yVec);

	vector<double> J;
	cout << testOpenCVNN(x, y, J) << " s" << endl;

	cout << "Finished, press enter to end" << endl;
	getchar();
}
