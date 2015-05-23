#ifndef _DATATOOLS_H
#define _DATATOOLS_H
#endif

#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>

using namespace std;

vector<vector<double>> readCSV(string fileName, bool header, double &time);