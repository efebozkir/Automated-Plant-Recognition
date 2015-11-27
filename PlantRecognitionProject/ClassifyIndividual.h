/*
#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include "opencv\cv.h"

using namespace std;
using namespace cv;

#define numberOfElements 9

struct DiffvName
{
	double difference;
	string name;
};

class ClassifyIndividual
{
	protected:
	double compareTrainSetArray[59];
	double compareTestSetArray[59];
	vector<DiffvName> myVect;
	double maxFeatures[numberOfElements];
	double minFeatures[numberOfElements];

	public:
	int makeClassification();
	void sortDataVect();
	void printVector();
	void getMinMaxFromFile();
	

};
*/