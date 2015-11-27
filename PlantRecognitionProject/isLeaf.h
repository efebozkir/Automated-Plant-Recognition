#ifndef ISLEAF_H
#define ISLEAF_H

#define _CRT_SECURE_NO_DEPRECATE

#include <cv.h>
#include <cxcore.h>
#include <ml.h>
#include "opencv\cv.h"
#include <opencv\cv.h>
#include <ml.h> 
#include <opencv\highgui.h>
#include <opencv2/core/core.hpp> 
#include <opencv2/features2d/features2d.hpp> 
#include <opencv2/highgui/highgui.hpp> 


#include "dirent.h"
using namespace std;
using namespace cv;

class isLeaf
{
public:
//1
	vector<KeyPoint> keypoints;
	Mat descriptor;
	SiftDescriptorExtractor detector;

//2
	Ptr<BOWImgDescriptorExtractor> myBowT;
	Ptr<DescriptorMatcher> matcherT;
	Ptr<FeatureDetector> detectorT;
	Ptr<DescriptorExtractor> extractorT;
	Mat trainingData;
	Mat trainingDataLabels;
	Mat testingData;
	Mat testingDataLabels;


//
	Mat myTrainData;
	Mat myTrainDataLabel;
	Mat myTestData;
	

	public:
	void featureExtractSamples();  //
	void csvOfSamples();  //
	int readProduceFromCsv();  //
	
	int produceDictionary();
	int produceTrainData();
	int produceTestData();
	void isLeafOrNot(); 



};
#endif 