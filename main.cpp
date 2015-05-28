#include <iostream>

#include "cvnn.h"

int main()
{
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
