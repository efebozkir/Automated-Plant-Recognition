#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include "opencv\cv.h"
#include "ExtractDescriptorHelper.h"

using namespace std;
using namespace cv;
#define arraySize 159

class ConvertToCSV
{
	public:
	void TrainFeaturesAsCSV();
	void TestFeaturesAsCSV();
};