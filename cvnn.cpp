#include "cvnn.h"

Mat randInitialiseWeights(int in, int out)
{
	double epsilon = sqrt(6) / sqrt(1 + in + out);

	Mat weights(in + 1, out, CV_64F);
	randu(weights, 0.0, 1.0);

	weights = weights * epsilon - epsilon;

	return weights;
}

Mat vector2dToMat(vector<vector<double>> data)
{
	size_t m = data.size();

	Mat mat;
	for(size_t i = 0; i < m; i++)
		mat.push_back(Mat(data[i]).reshape(1,1));
	mat.convertTo(mat, CV_64F);
	return mat;
}

Mat sigmoid(Mat data)
{
	data = -data;
	exp(data, data);
	return 1 / (1 + data);
}

Mat sigmoidGradient(Mat data)
{
	Mat sig = sigmoid(data);
	return sig.mul(1 - sig);
}