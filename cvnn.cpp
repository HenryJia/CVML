#include "cvnn.h"

#include <math.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>

cvnn::cvnn()
{
	iters = 100;
	alpha = 1;
	lambda = 0;
	threads = thread::hardware_concurrency();
}

double cvnn::train()
{
	double time = trainFuncApprox();
	if(classification == false)
		cout << "Training" << time << " s" << endl;
	return time;
}

double cvnn::trainConcurrent()
{
	double time = trainConcurrentFuncApprox();
	if(classification == false)
		cout << "Training" << time << " s" << endl;
	return time;
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
	int m = x.size().height;
	if(sum(x.col(0) == Mat(m, 1, CV_64F, 1))[0] != x.rows * 255)
		hconcat(Mat(m, 1, CV_64F, 1), x, x);
}

vector<vector<double>> cvnn::readCSV(string fileName, bool header, double &time)
{
	auto start = chrono::steady_clock::now();

	vector<vector<double>> result;
	ifstream in(fileName);
	string lineStr;
	string delimiter = ",";

	if(!in.is_open())
		cerr << "failed to open file\n";
	if(header == true)
		std::getline(in, lineStr);

	while(std::getline(in, lineStr))
	{
		vector<double> lineVec;
		size_t pos = 0;
		while((pos = lineStr.find(delimiter)) != std::string::npos)
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

void cvnn::forwardPropagate(size_t threadNum, int rangeLower, int rangeUpper)
{
	Mat data = x.rowRange(rangeLower, rangeUpper);
	size_t base = threadNum * (layerNum - 1);
	z[base] = data * theta[0];
	a[base] = sigmoid(z[base]);
	
	for(size_t i = 1; i < layerNum - 2; i++)
	{
		hconcat(Mat(a[base + i - 1].size().height, 1, CV_64F, 1), a[base + i - 1], a[base + i - 1]);
		z[base + i] = a[base + i - 1] * theta[i];
		a[base + i] = sigmoid(z[base + i]);
	}
	
	hconcat(Mat(a[base + layerNum - 3].size().height, 1, CV_64F, 1), a[base + layerNum - 3], a[base + layerNum - 3]);
	z[base + layerNum - 2] = a[base + layerNum - 3] * theta[layerNum - 2];
	a[base + layerNum - 2] = z[base + layerNum - 2];
}

void cvnn::smallDelta(size_t threadNum, int rangeLower, int rangeUpper)
{
	Mat data = x.rowRange(rangeLower, rangeUpper);
	Mat datay = y.rowRange(rangeLower, rangeUpper);
	size_t base = threadNum * (layerNum - 1);

	Mat trans;
	Mat product;

	z[base] = data * theta[0];
	a[base] = sigmoid(z[base]);
	
	for(size_t i = 1; i < layerNum - 2; i++)
	{
		hconcat(Mat(a[base + i - 1].size().height, 1, CV_64F, 1), a[base + i - 1], a[base + i - 1]);
		z[base + i] = a[base + i - 1] * theta[i];
		a[base + i] = sigmoid(z[base + i]);
	}
	
	hconcat(Mat(a[base + layerNum - 3].size().height, 1, CV_64F, 1), a[base + layerNum - 3], a[base + layerNum - 3]);
	z[base + layerNum - 2] = a[base + layerNum - 3] * theta[layerNum - 2];
	a[base + layerNum - 2] = z[base + layerNum - 2];

	delta[base + layerNum - 2] = a[base + layerNum - 2] - datay;

	for(int i = layerNum - 3; i >= 0; i--)
	{
		transpose(theta[i + 1], trans);
		product = delta[base + i + 1] * trans;
		delta[base + i] = product.colRange(1, product.cols).mul(sigmoidGradient(z[base + i]));
	}
}

double cvnn::trainConcurrentFuncApprox()
{
	auto start = chrono::steady_clock::now();

	Mat concatA;
	Mat concatZ;
	Mat concatdelta;
	Mat trans;

	vector<Mat> zFinal;
	vector<Mat> aFinal;

	vector<Mat> deltaFinal;
	vector<Mat> DeltaFinal;
	vector<Mat> thetaGradFinal;

	int m = x.size().height;

	J.clear();

	z.resize(threads * (layerNum - 1));
	a.resize(threads * (layerNum - 1));

	delta.resize(threads * (layerNum - 1));
	Delta.resize(threads * (layerNum - 1));
	thetaGrad.resize(threads * (layerNum - 1));

	zFinal.resize(layerNum - 1);
	aFinal.resize(layerNum - 1);

	deltaFinal.resize(layerNum - 1);
	DeltaFinal.resize(layerNum - 1);
	thetaGradFinal.resize(layerNum - 1);

	t.resize(threads);

	int batch = m / threads;
	int lowerRanges[threads];
	int upperRanges[threads];
	for(size_t i = 0; i < threads; i++)
	{
		lowerRanges[i] = i * batch;
		upperRanges[i] = (i + 1) * batch;
	}
	upperRanges[threads - 1] = m;

	for(int i = 0; i < iters; i++)
	{
		for(size_t j = 0; j < threads; j++)
			t[j] = thread(&cvnn::smallDelta, this, j, lowerRanges[j], upperRanges[j]);
		for(size_t j = 0; j < threads; j++)
			if(t[j].joinable())
				t[j].join();
		//for(size_t j = 0; j < threads; j++)
		//	smallDelta(j, lowerRanges[j], upperRanges[j]);
		for(size_t j = 0; j < layerNum - 1; j++)
		{
			concatA = a[j];
			concatZ = z[j];
			concatdelta = delta[j];
			for(size_t k = 1; k < threads; k++)
			{
				vconcat(concatA, a[k * (layerNum - 1) + j], concatA);
				vconcat(concatZ, z[k * (layerNum - 1) + j], concatZ);
				vconcat(concatdelta, delta[k * (layerNum - 1) + j], concatdelta);
			}
			aFinal[j] = concatA;
			zFinal[j] = concatZ;
			deltaFinal[j] = concatdelta;
		}
		/*//Calculate small delta
		deltaFinal[layerNum - 2] = aFinal[layerNum - 2] - y;

		for(int i = layerNum - 3; i >= 0; i--)
		{
			transpose(theta[i + 1], trans);
			product = deltaFinal[i + 1] * trans;
			deltaFinal[i] = product.colRange(0, product.size().width - 1).mul(sigmoidGradient(zFinal[i]));
		}*/

		//Calculate cost
		J.push_back(sum(deltaFinal[layerNum - 2].mul(deltaFinal[layerNum - 2]))[0] / (2 * m));
		cout << "Iteration " << i << ": " << "Cost: " << J[i] << endl;

		transpose(deltaFinal[0], trans);
		DeltaFinal[0] = trans * x;

		for(size_t i = 1; i < layerNum - 1; i++)
		{
			transpose(deltaFinal[i], trans);
			DeltaFinal[i] = trans * aFinal[i - 1];
		}

		for(size_t i = 0; i < layerNum - 1; i++)
		{
			transpose(DeltaFinal[i], trans);
			thetaGradFinal[i] = trans / m;
		}

		for(size_t i = 1; i < layerNum - 1; i++)
			theta[i] = theta[i] - alpha * thetaGradFinal[i];
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

	Mat trans;
	Mat product;

	/*Mat z2;
	Mat a2;
	Mat z3;
	Mat a3;
	Mat z4;
	Mat a4;*/

	Mat z[layerNum - 1];
	Mat a[layerNum - 1];

	/*Mat delta4;
	Mat delta3;
	Mat delta2;*/

	Mat delta[layerNum - 1];

	/*Mat Delta3;
	Mat Delta2;
	Mat Delta1;*/

	Mat Delta[layerNum - 1];

	/*Mat theta3Grad;
	Mat theta2Grad;
	Mat theta1Grad;*/

	Mat thetaGrad[layerNum - 1];
	for(int i = 0; i < iters; i++)
	{
		//Forward propagation
		/*z2 = x * theta1;
		a2 = sigmoid(z2);

		hconcat(Mat(a2.size().height, 1, CV_64F, 1), a2, a2);
		z3 = a2 * theta2;
		a3 = sigmoid(z3);

		hconcat(Mat(a3.size().height, 1, CV_64F, 1), a3, a3);
		z4 = a3 * theta3;
		a4 = z4;*/

		z[0] = x * theta[0];
		a[0] = sigmoid(z[0]);

		for(size_t i = 1; i < layerNum - 2; i++)
		{
			hconcat(Mat(a[i - 1].size().height, 1, CV_64F, 1), a[i - 1], a[i - 1]);
			z[i] = a[i - 1] * theta[i];
			a[i] = sigmoid(z[i]);
		}

		hconcat(Mat(a[layerNum - 3].size().height, 1, CV_64F, 1), a[layerNum - 3], a[layerNum - 3]);
		z[layerNum - 2] = a[layerNum - 3] * theta[layerNum - 2];
		a[layerNum - 2] = z[layerNum - 2];

		//Calculate small delta
		delta[layerNum - 2] = a[layerNum - 2] - y;

		/*transpose(theta3, trans);
		product = delta4 * trans;
		delta3 = product.colRange(0, product.size().width - 1).mul(sigmoidGradient(z3));

		transpose(theta2, trans);
		product = delta3 * trans;
		delta2 = product.colRange(0, product.size().width - 1).mul(sigmoidGradient(z2));*/

		for(int i = layerNum - 3; i >= 0; i--)
		{
			transpose(theta[i + 1], trans);
			product = delta[i + 1] * trans;
			delta[i] = product.colRange(1, product.cols).mul(sigmoidGradient(z[i]));
		}

		//Calculate cost
		J.push_back(sum(delta[layerNum - 2].mul(delta[layerNum - 2]))[0] / (2 * m));
		cout << "Iteration " << i << ": " << "Cost: " << J[i] << endl;

		//Accumulate small delta and calculate big delta which is the partial derivatives
		/*transpose(delta4, trans);
		Delta3 = trans * a3;

		transpose(delta3, trans);
		Delta2 = trans * a2;

		transpose(delta2, trans);
		Delta1 = trans * x;*/

		transpose(delta[0], trans);
		Delta[0] = trans * x;

		for(size_t i = 1; i < layerNum - 1; i++)
		{
			transpose(delta[i], trans);
			Delta[i] = trans * a[i - 1];
		}

		//Finish off the calculation
		/*transpose(Delta3, trans);
		theta3Grad = trans / m;

		transpose(Delta2, trans);
		theta2Grad = trans / m;

		transpose(Delta1, trans);
		theta1Grad = trans / m;*/

		for(size_t i = 0; i < layerNum - 1; i++)
		{
			transpose(Delta[i], trans);
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