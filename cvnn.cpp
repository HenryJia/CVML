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

double cvnn::validate()
{
	Mat z[layerNum - 1];
	Mat a[layerNum - 1];

	if(sum(xValidate.col(0) == Mat(mValidate, 1, CV_64F, 1))[0] != mValidate * 255)
		hconcat(Mat(mValidate, 1, CV_64F, 1), xValidate, xValidate);
	
	z[0] = xValidate * theta[0];
	a[0] = sigmoid(z[0]);
	
	for(size_t i = 1; i < layerNum - 2; i++)
	{
		hconcat(Mat(a[i - 1].rows, 1, CV_64F, 1), a[i - 1], a[i - 1]);
		z[i] = a[i - 1] * theta[i];
		a[i] = sigmoid(z[i]);
	}

	hconcat(Mat(a[layerNum - 3].rows, 1, CV_64F, 1), a[layerNum - 3], a[layerNum - 3]);
		z[layerNum - 2] = a[layerNum - 3] * theta[layerNum - 2];
		a[layerNum - 2] = z[layerNum - 2];

	Mat delta = a[layerNum - 2] - yValidate;

	//Calculate cost
	JValidate = sum(delta.mul(delta))[0] / (2 * mValidate);
	cout << "Validation Cost: " << JValidate << endl;
	return JValidate;
}

double cvnn::predict(string fileName)
{
	Mat z[layerNum - 1];
	Mat a[layerNum - 1];

	if(sum(xPredict.col(0) == Mat(mPredict, 1, CV_64F, 1))[0] != mPredict * 255)
		hconcat(Mat(mPredict, 1, CV_64F, 1), xPredict, xPredict);

	z[0] = xPredict * theta[0];
	a[0] = sigmoid(z[0]);
	
	for(size_t i = 1; i < layerNum - 2; i++)
	{
		hconcat(Mat(a[i - 1].rows, 1, CV_64F, 1), a[i - 1], a[i - 1]);
		z[i] = a[i - 1] * theta[i];
		a[i] = sigmoid(z[i]);
	}

	hconcat(Mat(a[layerNum - 3].rows, 1, CV_64F, 1), a[layerNum - 3], a[layerNum - 3]);
		z[layerNum - 2] = a[layerNum - 3] * theta[layerNum - 2];
		a[layerNum - 2] = z[layerNum - 2];

	return writeCSV(fileName, a[layerNum - 2]);
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
	m = x.rows;
}

void cvnn::setValidateData(vector<vector<double>> xVec, vector<vector<double>> yVec)
{
	xValidate = vector2dToMat(xVec);
	yValidate = vector2dToMat(yVec);
	mValidate = xValidate.rows;
}

void cvnn::setPredictData(vector<vector<double>> xVec)
{
	xPredict = vector2dToMat(xVec);
	mPredict = xPredict.rows;
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

double cvnn::writeCSV(string fileName, Mat data)
{
	auto start = chrono::steady_clock::now();

	vector<vector<double>> result;
	ofstream out(fileName);

	for(int i = 0; i < mPredict; i++)
		out << data.row(i) << endl;

	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;

	return chrono::duration <double> (elapsed).count();;
}

void cvnn::grad(size_t threadNum, int rangeLower, int rangeUpper)
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
		hconcat(Mat(a[base + i - 1].rows, 1, CV_64F, 1), a[base + i - 1], a[base + i - 1]);
		z[base + i] = a[base + i - 1] * theta[i];
		a[base + i] = sigmoid(z[base + i]);
	}
	
	hconcat(Mat(a[base + layerNum - 3].rows, 1, CV_64F, 1), a[base + layerNum - 3], a[base + layerNum - 3]);
	z[base + layerNum - 2] = a[base + layerNum - 3] * theta[layerNum - 2];
	a[base + layerNum - 2] = z[base + layerNum - 2];

	delta[base + layerNum - 2] = a[base + layerNum - 2] - datay;

	for(int i = layerNum - 3; i >= 0; i--)
	{
		transpose(theta[i + 1], trans);
		product = delta[base + i + 1] * trans;
		delta[base + i] = product.colRange(1, product.cols).mul(sigmoidGradient(z[base + i]));
	}

	transpose(delta[base], trans);
	Delta[base] = trans * data;

	for(size_t i = 1; i < layerNum - 1; i++)
	{
		transpose(delta[base + i], trans);
		Delta[base + i] = trans * a[base + i - 1];
	}

	for(size_t i = 0; i < layerNum - 1; i++)
	{
		transpose(Delta[base + i], trans);
		thetaGrad[base + i] = trans / m;
	}

	JBatch[threadNum] = sum(delta[base + layerNum - 2].mul(delta[base + layerNum - 2]))[0] / (2 * m);
}

void cvnn::sumthetaGrad()
{
	Mat sum;
	for(size_t j = 0; j < layerNum - 1; j++)
	{
		sum = thetaGrad[j];
		for(size_t k = 1; k < threads; k++)
		sum += thetaGrad[k * (layerNum - 1) + j];
		thetaGradFinal[j] = sum;
	}
}

double cvnn::trainConcurrentFuncApprox()
{
	auto start = chrono::steady_clock::now();

	if(sum(x.col(0) == Mat(m, 1, CV_64F, 1))[0] != m * 255)
		hconcat(Mat(m, 1, CV_64F, 1), x, x);

	J.clear();
	JBatch.resize(threads);

	z.resize(threads * (layerNum - 1));
	a.resize(threads * (layerNum - 1));

	delta.resize(threads * (layerNum - 1));
	Delta.resize(threads * (layerNum - 1));
	thetaGrad.resize(threads * (layerNum - 1));

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
			t[j] = thread(&cvnn::grad, this, j, lowerRanges[j], upperRanges[j]);
		for(size_t j = 0; j < threads; j++)
			if(t[j].joinable())
				t[j].join();

		thread sumthetaGradThread(&cvnn::sumthetaGrad, this);

		//Calculate cost
		J.push_back(JBatch[0]);
		for(size_t j = 1; j < threads; j++)
			J.back() += JBatch[j];
		cout << "Iteration " << i << ": " << "Cost: " << J[i] << endl;

		sumthetaGradThread.join();
		for(size_t j = 0; j < layerNum - 1; j++)
			theta[j] = theta[j] - alpha * thetaGradFinal[j];
	}

	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}

double cvnn::trainFuncApprox()
{
	auto start = chrono::steady_clock::now();

	int m = x.rows;

	if(sum(x.col(0) == Mat(m, 1, CV_64F, 1))[0] != m * 255)
		hconcat(Mat(m, 1, CV_64F, 1), x, x);

	J.clear();

	Mat trans;
	Mat product;

	Mat z[layerNum - 1];
	Mat a[layerNum - 1];

	Mat delta[layerNum - 1];

	Mat Delta[layerNum - 1];

	Mat thetaGrad[layerNum - 1];

	for(int i = 0; i < iters; i++)
	{

		z[0] = x * theta[0];
		a[0] = sigmoid(z[0]);
		for(size_t j = 1; j < layerNum - 2; j++)
		{
			hconcat(Mat(a[j - 1].rows, 1, CV_64F, 1), a[j - 1], a[j - 1]);
			z[j] = a[j - 1] * theta[j];
			a[j] = sigmoid(z[j]);
		}

		hconcat(Mat(a[layerNum - 3].rows, 1, CV_64F, 1), a[layerNum - 3], a[layerNum - 3]);
		z[layerNum - 2] = a[layerNum - 3] * theta[layerNum - 2];
		a[layerNum - 2] = z[layerNum - 2];

		//Calculate small delta
		delta[layerNum - 2] = a[layerNum - 2] - y;

		for(int j = layerNum - 3; j >= 0; j--)
		{
			transpose(theta[j + 1], trans);
			product = delta[j + 1] * trans;
			delta[j] = product.colRange(1, product.cols).mul(sigmoidGradient(z[j]));
			//cout << "product " << j << ": " << product.row(0).col(1) << endl;
			//cout << "delta " << j << ": " << delta[j].row(0).col(0) << endl;
		}

		//Calculate cost
		J.push_back(sum(delta[layerNum - 2].mul(delta[layerNum - 2]))[0] / (2 * m));
		cout << "Iteration " << i << ": " << "Cost: " << J[i] << endl;

		transpose(delta[0], trans);
		Delta[0] = trans * x;

		for(size_t j = 1; j < layerNum - 1; j++)
		{
			transpose(delta[j], trans);
			Delta[j] = trans * a[j - 1];
		}

		for(size_t j = 0; j < layerNum - 1; j++)
		{
			transpose(Delta[j], trans);
			thetaGrad[j] = trans / m;
		}

		for(size_t j = 0; j < layerNum - 1; j++)
			theta[j] = theta[j] - alpha * thetaGrad[j];
	}
	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	return chrono::duration <double> (elapsed).count();
}

Mat cvnn::randInitialiseWeights(int in, int out)
{
	double epsilon = sqrt(6) / sqrt(1 + in + out);

	Mat weights(in + 1, out, CV_64F);
	cv::theRNG().state = getTickCount();
	randu(weights, 0.0, 1.0);

	weights = abs(weights * 2 * epsilon - epsilon);

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
	Mat result;
	exp(-data, result);
	return 1 / (1 + result);
}

Mat cvnn::sigmoidGradient(Mat data)
{
	Mat sig = sigmoid(data);
	return sig.mul(1 - sig);
}

Mat cvnn::normalise(Mat data)
{
	Mat result(data.rows, data.cols, CV_64F);
	Mat mean(1, data.cols, CV_64F);
	Mat stddev(1, data.cols, CV_64F);
	for (int i = 0; i < data.cols; i++)
	{
		cv::Mat meanValue, stdValue;
		meanStdDev(data.col(i), meanValue, stdValue);
		stddev.at<double>(i) = stdValue.at<double>(0);
		mean.at<double>(i) = meanValue.at<double>(0);
    }
	for(int i = 0; i < data.rows; ++i)
	{
		result.row(i) = data.row(i) - mean;
		result.row(i) = result.row(i) / stddev;
	}
	return result;
}