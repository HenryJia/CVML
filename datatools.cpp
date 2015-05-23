#include "datatools.h"

vector<vector<double>> readCSV(string fileName, bool header, double &time)
{
	auto start = chrono::steady_clock::now();
	
	vector<vector<double>> result;
	ifstream in(fileName);
	string lineStr;
	string delimiter = ",";
	
	if (!in.is_open())
		cerr << "failed to open file\n";
	if(header == true)
		std::getline(in, lineStr);
	
	while(std::getline(in, lineStr))
	{
		vector<double> lineVec;
		size_t pos = 0;
		while ((pos = lineStr.find(delimiter)) != std::string::npos)
		{
			lineVec.push_back(stod(lineStr.substr(0, pos)));
			lineStr.erase(0, pos + delimiter.length());
		}
		result.push_back(lineVec);
	}
	
	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	time = chrono::duration <double> (elapsed).count();
	
	return result;
}