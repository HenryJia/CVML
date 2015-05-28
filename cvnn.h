#ifndef _CVNN_H
#define _CVNN_H
#endif

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <thread>

using namespace std;
using namespace cv;

class cvnn
{
public:
	cvnn();

	double train();
	double trainConcurrent();
	void setLayers(vector<int> layers);
	void setClassify(bool c) { classification = c; }
	void setAlpha(double a) { alpha = a; }
	void setThreads(size_t t) { threads = t; }
	void setIters(int i) { iters = i; }
	void setLambda(int l) { lambda = l; }
	void setData(vector<vector<double>> xVec, vector<vector<double>> yVec);

	vector<Mat> getTheta() { return theta; }

	vector<vector<double>> readCSV(string fileName, bool header, double &time);

private:
	double trainFuncApprox();
	double trainConcurrentFuncApprox();

	Mat randInitialiseWeights(int in, int out);
	Mat vector2dToMat(vector<vector<double>> data);
	Mat sigmoid(Mat data);
	Mat sigmoidGradient(Mat data);

	double alpha;
	double lambda;
	int iters;
	size_t threads;
	size_t layerNum;
	Mat x;
	Mat y;
	vector<Mat> theta;
	vector<int> layers;
	vector<double> J;
	vector<thread> t;
	bool classification;

	// Variables for concurrency:
	vector<Mat> z;
	vector<Mat> a;

	vector<Mat> delta;
	vector<Mat> Delta;
	vector<Mat> thetaGrad;

	vector<Mat> aFinal;
	vector<Mat> deltaFinal;
	vector<Mat> DeltaFinal;
	vector<Mat> thetaGradFinal;

	// Functions for concurrency
	void smallDelta(size_t threadNum, int rangeLower, int rangeUpper);
	void concata();
	void concatdelta();
};