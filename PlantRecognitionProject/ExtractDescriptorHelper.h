#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include "opencv\cv.h"



using namespace std;
using namespace cv;

class ExtractDescriptorHelper
{
	protected:
		int numberOfToothPoints;
		double perimeter;
		double isoperimetricQuotient;
		double compactness;
		double convexHullArea;
		double convexRatio;
		double aspectRatio;
		double rectangularity;
		double eccentricity;
		double formFactor;
		double elongatedness;
		double nonConvexRatio;
		vector<double> myFourierDescriptors;
		vector<double> myFeatureVector;

	public:
		CvSeq* FindLargestContour(IplImage* g_image,int *largestArea);
		IplImage* RemoveStem(IplImage* inputImage);
		int chainHull_2D(Point2f* P, int n, Point2f* H);
		bool isToothPoint(CvSeq* contour,int index, int threshold);
		void ExtractDescriptors(IplImage* inputImageForExtraction);
		float isLeft(Point2f P0, Point2f P1, Point2f P2);
		void extractFourierDescriptors(CvSeq* inputSeq);
		//void sortComplexVector(vector<ComplexNumber> input);
		void sortVector();
		void createFeatureVector();
		vector<double> getMyFeatureVector();
		int getNumberOfToothPoints();
		double getPerimeter();
		double getIsoperimetricQuotient();
		double getCompactness();
		double getConvexHullArea();
		double getConvexRatio();
		double getAspectRatio();
		double getRectangularity();
		double getEccentricity();
		double getFormFactor();
		double getElongatedness();
		double getNonConvexRatio();

};