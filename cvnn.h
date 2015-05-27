#ifndef _CVNN_H
#define _CVNN_H
#endif

#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>

using namespace std;
using namespace cv;

Mat randInitialiseWeights(int in, int out);
Mat vector2dToMat(vector<vector<double>>);
Mat vector2dToMat(vector<double> data);
Mat sigmoid(Mat data);
Mat sigmoidGradient(Mat data);