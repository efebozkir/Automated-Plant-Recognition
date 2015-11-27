#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include "opencv\cv.h"

using namespace std;
using namespace cv;

#define numberOfElements 9



class NormaliseGeoFeatures
{
	protected:
	vector<string> fileNamesVect;
	double maxArray[numberOfElements];
	double minArray[numberOfElements];

	public:
		void produceFileNamesVect();
		void calcMaxMin();
		void initializeMaxMin();
		int normalizeGeoFeatures();
		void writeMaxMinToFile();
		double* getMaxArray();
		double* getMinArray();


};