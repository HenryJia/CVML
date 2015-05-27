#ifndef _CVNN_H
#define _CVNN_H
#endif

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

class cvnn
{
public:
	cvnn();

	void train();
	void trainConcurrent();
	void setLayers(vector<int> layers);
	void setClassify(bool c) { classification = c; }
	void setAlpha(double a) { alpha = a; }
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
	size_t layerNum;
	Mat x;
	Mat y;
	vector<Mat> theta;
	vector<int> layers;
	vector<double> J;
	bool classification;
};