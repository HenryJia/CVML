#include "cvnn.h"

#include <math.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <thread>

cvnn::cvnn()
{
	iters = 100;
	alpha = 1;
	lambda = 0;
}

void cvnn::train()
{
	if(classification == false)
		cout << "Training" << trainFuncApprox() << " s" << endl;
}

void cvnn::trainConcurrent()
{
	if(classification == false)
		cout << "Training" << trainConcurrentFuncApprox() << " s" << endl;
}

void cvnn::setLayers(vector<int> l)
{
	layers = l;
	layerNum = l.size();

	vector<Mat> m;
	for(size_t i = 0; i < (l.size() - 1); i++)
		m.push_back(randInitialiseWeights(l[i], l[i + 1]));
	theta = m;
}

void cvnn::setData(vector<vector<double>> xVec, vector<vector<double>> yVec)
{
	x = vector2dToMat(xVec);
	y = vector2dToMat(yVec);
}

vector<vector<double>> cvnn::readCSV(string fileName, bool header, double &time)
{
	auto start = chrono::steady_clock::now();

	vector<vector<double>> result;
	ifstream in(fileName);
	string lineStr;
	string delimiter = ",";

	if (!in.is_open())
		cerr << "failed to open file\n";
	if (header == true)
		std::getline(in, lineStr);

	while (std::getline(in, lineStr))
	{
		vector<double> lineVec;
		size_t pos = 0;
		while ((pos = lineStr.find(delimiter)) != std::string::npos)
		{
			lineVec.push_back(stold(lineStr.substr(0, pos)));
			lineStr.erase(0, pos + delimiter.length());
		}
		lineVec.push_back(stold(lineStr));
		result.push_back(lineVec);
	}

	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	time = chrono::duration <double> (elapsed).count();

	return result;
}

double cvnn::trainConcurrentFuncApprox()
{
	auto start = chrono::steady_clock::now();

	int m = x.size().height;

	J.clear();

	cv::hconcat(cv::Mat(m, 1, CV_64F, 1), x, x);

	cv::Mat trans;
	cv::Mat product;

	cv::Mat z[layerNum - 1];
	cv::Mat a[layerNum - 1];

	cv::Mat delta[layerNum - 1];

	cv::Mat Delta[layerNum - 1];

	cv::Mat thetaGrad[layerNum - 1];
	for(int i = 0; i < iters; i++)
	{
		z[0] = x * theta[0];
		a[0] = sigmoid(z[0]);

		for(size_t i = 1; i < layerNum - 2; i++)
		{
			cv::hconcat(cv::Mat(a[i - 1].size().height, 1, CV_64F, 1), a[i - 1], a[i - 1]);
			z[i] = a[i - 1] * theta[i];
			a[i] = sigmoid(z[i]);
		}

		cv::hconcat(cv::Mat(a[layerNum - 3].size().height, 1, CV_64F, 1), a[layerNum - 3], a[layerNum - 3]);
		z[layerNum - 2] = a[layerNum - 3] * theta[layerNum - 2];
		a[layerNum - 2] = z[layerNum - 2];

		//Calculate small delta
		delta[layerNum - 2] = a[layerNum - 2] - y;

		for(int i = layerNum - 3; i >= 0; i--)
		{
			cv::transpose(theta[i + 1], trans);
			product = delta[i + 1] * trans;
			delta[i] = product.colRange(0, product.size().width - 1).mul(sigmoidGradient(z[i]));
		}

		//Calculate cost
		J.push_back(sum(delta[layerNum - 2].mul(delta[layerNum - 2]))[0] / (2 * m));
		cout << "Iteration " << i << ": " << "Cost: " << J[i] << endl;

		cv::transpose(delta[0], trans);
		Delta[0] = trans * x;

		for(size_t i = 1; i < layerNum - 1; i++)
		{
			cv::transpose(delta[i], trans);
			Delta[i] = trans * a[i - 1];
		}

		for(size_t i = 0; i < layerNum - 1; i++)
		{
			cv::transpose(Delta[i], trans);
			thetaGrad[i] = trans / m;
		}

		for(size_t i = 1; i < layerNum - 1; i++)
			theta[i] = theta[i] - alpha * thetaGrad[i];
	}
	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}

double cvnn::trainFuncApprox()
{
	auto start = chrono::steady_clock::now();

	int m = x.size().height;

	J.clear();

	cv::hconcat(cv::Mat(m, 1, CV_64F, 1), x, x);

	cv::Mat trans;
	cv::Mat product;

	/*cv::Mat z2;
	cv::Mat a2;
	cv::Mat z3;
	cv::Mat a3;
	cv::Mat z4;
	cv::Mat a4;*/

	cv::Mat z[layerNum - 1];
	cv::Mat a[layerNum - 1];

	/*cv::Mat delta4;
	cv::Mat delta3;
	cv::Mat delta2;*/

	cv::Mat delta[layerNum - 1];

	/*cv::Mat Delta3;
	cv::Mat Delta2;
	cv::Mat Delta1;*/

	cv::Mat Delta[layerNum - 1];

	/*cv::Mat theta3Grad;
	cv::Mat theta2Grad;
	cv::Mat theta1Grad;*/

	cv::Mat thetaGrad[layerNum - 1];
	for(int i = 0; i < iters; i++)
	{
		//Forward propagation
		/*z2 = x * theta1;
		a2 = sigmoid(z2);

		cv::hconcat(cv::Mat(a2.size().height, 1, CV_64F, 1), a2, a2);
		z3 = a2 * theta2;
		a3 = sigmoid(z3);

		cv::hconcat(cv::Mat(a3.size().height, 1, CV_64F, 1), a3, a3);
		z4 = a3 * theta3;
		a4 = z4;*/

		z[0] = x * theta[0];
		a[0] = sigmoid(z[0]);

		for(size_t i = 1; i < layerNum - 2; i++)
		{
			cv::hconcat(cv::Mat(a[i - 1].size().height, 1, CV_64F, 1), a[i - 1], a[i - 1]);
			z[i] = a[i - 1] * theta[i];
			a[i] = sigmoid(z[i]);
		}

		cv::hconcat(cv::Mat(a[layerNum - 3].size().height, 1, CV_64F, 1), a[layerNum - 3], a[layerNum - 3]);
		z[layerNum - 2] = a[layerNum - 3] * theta[layerNum - 2];
		a[layerNum - 2] = z[layerNum - 2];

		//Calculate small delta
		delta[layerNum - 2] = a[layerNum - 2] - y;

		/*cv::transpose(theta3, trans);
		product = delta4 * trans;
		delta3 = product.colRange(0, product.size().width - 1).mul(sigmoidGradient(z3));

		cv::transpose(theta2, trans);
		product = delta3 * trans;
		delta2 = product.colRange(0, product.size().width - 1).mul(sigmoidGradient(z2));*/

		for(int i = layerNum - 3; i >= 0; i--)
		{
			cv::transpose(theta[i + 1], trans);
			product = delta[i + 1] * trans;
			delta[i] = product.colRange(0, product.size().width - 1).mul(sigmoidGradient(z[i]));
		}

		//Calculate cost
		J.push_back(sum(delta[layerNum - 2].mul(delta[layerNum - 2]))[0] / (2 * m));
		cout << "Iteration " << i << ": " << "Cost: " << J[i] << endl;

		//Accumulate small delta and calculate big delta which is the partial derivatives
		/*cv::transpose(delta4, trans);
		Delta3 = trans * a3;

		cv::transpose(delta3, trans);
		Delta2 = trans * a2;

		cv::transpose(delta2, trans);
		Delta1 = trans * x;*/

		cv::transpose(delta[0], trans);
		Delta[0] = trans * x;

		for(size_t i = 1; i < layerNum - 1; i++)
		{
			cv::transpose(delta[i], trans);
			Delta[i] = trans * a[i - 1];
		}

		//Finish off the calculation
		/*cv::transpose(Delta3, trans);
		theta3Grad = trans / m;

		cv::transpose(Delta2, trans);
		theta2Grad = trans / m;

		cv::transpose(Delta1, trans);
		theta1Grad = trans / m;*/

		for(size_t i = 0; i < layerNum - 1; i++)
		{
			cv::transpose(Delta[i], trans);
			thetaGrad[i] = trans / m;
		}

		//Gradient descent
		/*theta3 = theta3 - alpha * theta3Grad;
		theta2 = theta2 - alpha * theta2Grad;
		theta1 = theta1 - alpha * theta1Grad;*/

		for(size_t i = 1; i < layerNum - 1; i++)
			theta[i] = theta[i] - alpha * thetaGrad[i];
	}
	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}

Mat cvnn::randInitialiseWeights(int in, int out)
{
	double epsilon = sqrt(6) / sqrt(1 + in + out);

	Mat weights(in + 1, out, CV_64F);
	randu(weights, 0.0, 1.0);

	weights = weights * epsilon - epsilon;

	return weights;
}

Mat cvnn::vector2dToMat(vector<vector<double>> data)
{
	size_t m = data.size();

	Mat mat;
	for(size_t i = 0; i < m; i++)
		mat.push_back(Mat(data[i]).reshape(1,1));
	mat.convertTo(mat, CV_64F);
	return mat;
}

Mat cvnn::sigmoid(Mat data)
{
	data = -data;
	exp(data, data);
	return 1 / (1 + data);
}

Mat cvnn::sigmoidGradient(Mat data)
{
	Mat sig = sigmoid(data);
	return sig.mul(1 - sig);
}