#ifndef CLASSIFY_H
#define CLASSIFY_H
#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include "opencv\cv.h"

using namespace std;
using namespace cv;

#define numberOfElements 9
#define featureVectSize 159

struct DiffvName
{
	double difference;
	string name;
};

class Classify
{
	protected:
	double compareTrainSetArray[featureVectSize];
	double compareTestSetArray[featureVectSize];
	vector<DiffvName> myVect;
	double maxFeatures[numberOfElements];
	double minFeatures[numberOfElements];

	public:
	int makeClassification();
	void sortDataVect();
	void printVector();
	void getMinMaxFromFile();
	

};
#endif 