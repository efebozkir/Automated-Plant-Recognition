#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include <cmath>
#include "opencv\cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv\highgui.h"
#include "opencv2\core\core.hpp"
#include "ExtractDescriptorHelper.h"
#include "ComplexNumber.h"

#define PI 3.14159265
#define fourierSize 151

using namespace std;
using namespace cv;

//Sorting according to the angles of the contour points
void sortComplexVector(vector<ComplexNumber> input)
{
	ComplexNumber temp1;
	
	for(int pass=0; pass<input.size()-1; pass++)
	{
		for(int j=0; j<input.size()-pass-1; j++)
		{
			if(atan2(input.at(j).getImag(), input.at(j).getReal())>atan2(input.at(j+1).getImag(), input.at(j+1).getReal()))
			{
				temp1=input.at(j);
				input.at(j)=input.at(j+1);
				input.at(j+1)=temp1;
			}
		}
	}
};

//Finding largest surrounding contour on binary image
CvSeq* ExtractDescriptorHelper::FindLargestContour(IplImage* g_image,int *largestArea)
{
	IplImage*	g_gray = NULL;
	IplImage* gg_gray=NULL;
	int		g_thresh = 100;
    CvMemStorage* 	g_storage = NULL;	
	CvMemStorage* gg_storage=NULL;
	
	if( g_storage == NULL )
	{
		g_gray = cvCreateImage( cvGetSize( g_image ), 8, 1 );
		gg_gray=cvCreateImage(cvGetSize(g_image), 8, 1);
		g_storage = cvCreateMemStorage(0);
		gg_storage= cvCreateMemStorage(0);

	} 
	else 
	{
		cvClearMemStorage( g_storage );
		cvClearMemStorage(gg_storage);
	}


	// contour çýkarýlmasý
	CvSeq* contours = 0;
	CvSeq* ptr;
	cvThreshold( g_image, g_gray, g_thresh, 255, CV_THRESH_BINARY);
	cvFindContours( g_gray, g_storage, &contours );
	CvChain* chain=0;

	CvSeq* current_contour = contours;
	CvSeq* largestContour = contours;
	int largest = 0;

	while (current_contour != NULL)
	{
		double area = fabs(cvContourArea(current_contour,CV_WHOLE_SEQ, false));
		if(area>largest)
		{
			largest = area;
			largestContour = current_contour;
		}
		
		current_contour = current_contour->h_next;
	}

	*largestArea = largest;

	return largestContour;

};

//Removing the stem of the leaf in binary image
IplImage* ExtractDescriptorHelper::RemoveStem(IplImage* inputImage)
{
	// resize the image to 256 x 256 and convert binary 
	int workingRes = 256;
	int		g_thresh = 100;
	
	IplImage* readImage = inputImage; // resmin ilk okunan nhali 
	//cout<<"Depth:"<<readImage->depth<<"Channels:"<<readImage->nChannels<<endl;
	IplImage* originalImage = cvCreateImage (cvSize (workingRes, workingRes), readImage->depth, readImage->nChannels); // 256*256 versiyonu
	
	cvResize (readImage,originalImage , CV_INTER_LINEAR);

	IplImage* g_gray = cvCreateImage( cvGetSize( originalImage ), 8, 1 ); // grayscale hali 


	//g_gray grayscale haline sift uygulayabiliriz.


	cvCvtColor( originalImage, g_gray, CV_BGR2GRAY );
	cvThreshold( g_gray, g_gray, g_thresh, 255, CV_THRESH_BINARY );	// orjinal binary hali 
	
	//cvShowImage("zaaaa", g_gray);
	IplImage* subImage = cvCreateImage( cvGetSize( g_gray ),g_gray->depth , g_gray->nChannels ); 
	cvCopyImage(g_gray,subImage);
	

	// erotion ve delation ile yaprak sapýnýn ayrýlmasý
	int pos = 5;	
	IplConvKernel* element = cvCreateStructuringElementEx(pos*2+1, pos*2+1, pos, pos, CV_SHAPE_ELLIPSE);
	cvErode(subImage,subImage,element,1);
	cvDilate(subImage,subImage,element,1);

	// resimlerin farkýný al
	cvAbsDiff(g_gray,subImage,subImage);

	// sapý çýkarma
	int *g_grayArea = new int;
	int *subImageArea = new int;
	int *tempArea = new int;
	
	FindLargestContour(g_gray,g_grayArea); // resimdeki toplam yaprak alaný
	
	CvSeq* largestContour = FindLargestContour(subImage,subImageArea); // sap	
	
	IplImage* temp = cvCreateImage(Size(g_gray->width,g_gray->height),g_gray->depth,g_gray->nChannels);
	IplImage* old =  cvCreateImage(Size(g_gray->width,g_gray->height),g_gray->depth,g_gray->nChannels);
	cvCopyImage(g_gray,temp);
	cvCopyImage(g_gray,old);
	largestContour->h_next = NULL;
	largestContour->h_prev = NULL;

	cvDrawContours(temp,largestContour,cvScalarAll(0),cvScalarAll(0),100,-1);	
	FindLargestContour(temp,tempArea);
	
	cvReleaseImage(&readImage);
	cvReleaseImage(&originalImage);
	cvReleaseImage(&g_gray);
	
	if(fabs((double)(*g_grayArea - *subImageArea - *tempArea) / *g_grayArea) < 0.2)
	{
		//cvShowImage("temp", temp);
		return temp;
	}
	else
	{
		//cvShowImage("old", old);
		return old;	
	}
};

// Finding convex hull area of the leaf
int ExtractDescriptorHelper::chainHull_2D(Point2f* P, int n, Point2f* H)
{
	// the output array H[] will be used as the stack
    int    bot=0, top=(-1);  // indices for bottom and top of the stack
    int    i;                // array scan index

    // Get the indices of points with min x-coord and min|max y-coord
    int minmin = 0, minmax;
    float xmin = P[0].x;
    for (i=1; i<n; i++)
        if (P[i].x != xmin) break;
    minmax = i-1;
    if (minmax == n-1) {       // degenerate case: all x-coords == xmin
        H[++top] = P[minmin];
        if (P[minmax].y != P[minmin].y) // a nontrivial segment
            H[++top] = P[minmax];
        H[++top] = P[minmin];           // add polygon endpoint
        return top+1;
    }

    // Get the indices of points with max x-coord and min|max y-coord
    int maxmin, maxmax = n-1;
    float xmax = P[n-1].x;
    for (i=n-2; i>=0; i--)
        if (P[i].x != xmax) break;
    maxmin = i+1;

    // Compute the lower hull on the stack H
    H[++top] = P[minmin];      // push minmin point onto stack
    i = minmax;
    while (++i <= maxmin)
    {
        // the lower line joins P[minmin] with P[maxmin]
        if (isLeft( P[minmin], P[maxmin], P[i]) >= 0 && i < maxmin)
            continue;          // ignore P[i] above or on the lower line

        while (top > 0)        // there are at least 2 points on the stack
        {
            // test if P[i] is left of the line at the stack top
            if (isLeft( H[top-1], H[top], P[i]) > 0)
                break;         // P[i] is a new hull vertex
            else
                top--;         // pop top point off stack
        }
        H[++top] = P[i];       // push P[i] onto stack
    }

    // Next, compute the upper hull on the stack H above the bottom hull
    if (maxmax != maxmin)      // if distinct xmax points
        H[++top] = P[maxmax];  // push maxmax point onto stack
    bot = top;                 // the bottom point of the upper hull stack
    i = maxmin;
    while (--i >= minmax)
    {
        // the upper line joins P[maxmax] with P[minmax]
        if (isLeft( P[maxmax], P[minmax], P[i]) >= 0 && i > minmax)
            continue;          // ignore P[i] below or on the upper line

        while (top > bot)    // at least 2 points on the upper stack
        {
            // test if P[i] is left of the line at the stack top
            if (isLeft( H[top-1], H[top], P[i]) > 0)
                break;         // P[i] is a new hull vertex
            else
                top--;         // pop top point off stack
        }
        H[++top] = P[i];       // push P[i] onto stack
    }
    if (minmax != minmin)
        H[++top] = P[minmin];  // push joining endpoint onto stack

    return top+1;
};

//Looking if it is tooth point or not
bool ExtractDescriptorHelper::isToothPoint(CvSeq* contour,int index, int threshold)
{
	int nextIndex = (index + contour->total + threshold) % contour->total;
	int previousIndex = (index + contour->total - threshold) & contour->total;

	CvPoint* p1 = CV_GET_SEQ_ELEM(CvPoint,contour,index); // center point
	CvPoint* p2 = CV_GET_SEQ_ELEM(CvPoint,contour,nextIndex);
	CvPoint* p3 = CV_GET_SEQ_ELEM(CvPoint,contour,previousIndex);

	int x1 = p1->x - p2->x;
	int y1 = p1->y - p2->y;

	int x2 =  p1->x - p3->x;
	int y2 = p1->y -p3->y;

	double cost = (x1 * x2 + y1*y2) / (sqrt((double)x1*x1 + y1*y1) * sqrt((double)x2*x2 + y2*y2));

	double degree = acos(cos(cost)) * ( 180/ CV_PI );

	if( 0.8 <= sin(degree) && sin(degree)<=1 )
		return true;
	return false;
};

//Extracting the descriptors of the leaf
void ExtractDescriptorHelper::ExtractDescriptors(IplImage* inputImageForExtraction)
{
	IplImage* leafImage = RemoveStem(inputImageForExtraction); // sapý ayýr

	vector<Point2f> originalPoints;   // Your original points
    vector<Point2f> ch;  // Convex hull points
	vector<Point2f> contour; // convex contour

	// extraction of isoperimetric quotient

	int *ptrlargestArea = new(int);	
	CvSeq* largest_contour = FindLargestContour(leafImage,ptrlargestArea);
	int largestArea = *ptrlargestArea;
	extractFourierDescriptors(largest_contour);//Extracting the fourier descriptors and storing them in vector

	//largest_contour

	// toothlarý hesapla
	
	for(int k=3;k<=30;k++)
	{
		int count = 0;
		for(int i=0;i<largest_contour->total;i++)
			if(isToothPoint(largest_contour,i,k))
				count++;
		//myfile<<count<<"   "; 

		numberOfToothPoints=count;
	}

	perimeter = fabs(cvContourPerimeter(largest_contour));
	isoperimetricQuotient = largestArea / perimeter;
	compactness= pow(perimeter, 2)/(4*PI*largestArea);
	//compactness=1;

	

	for(int i=0;i<largest_contour->total;i++)
	{
		CvPoint* newPoint = CV_GET_SEQ_ELEM(CvPoint,largest_contour,i);
		Point2f point = Point2f(newPoint->x,newPoint->y);
		originalPoints.push_back(point);		
	}		
	
	ch.resize(2*originalPoints.size());
	chainHull_2D(&originalPoints[0],originalPoints.size(),&ch[0]);

	convexHullArea = 0;
	for (int i = 0; i < ch.size(); i++)
	{
		int next_i = (i+1)%(ch.size());
        double dX   = ch[next_i].x - ch[i].x;
        double avgY = (ch[next_i].y + ch[i].y)/2;
	    convexHullArea += dX*avgY;  // This is the integration step.
    }
	convexHullArea = abs(convexHullArea); 

	if(convexHullArea == 0)
		convexHullArea = 0;

	convexRatio = largestArea/convexHullArea;	
	nonConvexRatio= (convexHullArea-largestArea)/largestArea;

	RotatedRect rect = fitEllipse(Mat(originalPoints));

	double major,minor;
	if(rect.size.height > rect.size.width)
	{
		major = (double) rect.size.height/2;
		minor = (double) rect.size.width/2;

	}
	else
	{
		minor = (double) rect.size.height/2;
		major = (double) rect.size.width/2;
	}

	//elongatedness=largestArea/2*pow(major,2); /// RF ÝÇÝN DAHA ÝYÝ
	//elongatedness=(double) major/minor;//// RF ÝÇÝN
	elongatedness=1; // k-NN için

	aspectRatio = (double) minor/major;

	rectangularity = largestArea / (minor*major); 
	
	double c =(double) sqrt(pow(major,2)-pow(minor,2));

	eccentricity = (double)c/major;
	
	formFactor = (4*(22/7)*largestArea) / ( perimeter * perimeter);
	/*
	//int typeNumber = atoi(&type[0]);	
	cout<<"ConvexRatio:"<<convexRatio<<" isoperimetricQuotient:"<<isoperimetricQuotient<<" Eccentricity:"<<eccentricity<<endl;
	cout<<" AspectRatio:"<<aspectRatio<<" Rectangularity:"<<rectangularity<<" FormFactor:"<<formFactor<<" Compactness: "<<compactness<<endl;
	cout<<"Number of Tooth Points: "<<numberOfToothPoints<<endl;
	cout<<"Perimeter: "<<perimeter<<endl;
	cout<<"Elongatedness: "<<elongatedness<<endl;
	cout<<"NonConvexRatio: "<<nonConvexRatio<<endl;

	//cout<<"Perimeter Ratio: "<<perimeterRatio<<endl;
	cout<<"Fourier Descriptorlar vector icerisinde tutuldu"<<endl;
	*/
	cvReleaseImage(&leafImage);
};

void ExtractDescriptorHelper::extractFourierDescriptors(CvSeq* contour)
{

	int x=contour->total;

	CvPoint *pt=new CvPoint[contour->total];
	for(int i=0; i<contour->total; i++)
	{
		pt[i] = *(CvPoint*)cvGetSeqElem(contour, i);
		//cout<<"x:"<< pt[i].x<<" y:"<<pt[i].y<<endl;
	}
	//cout<<"Koordinatlar yazildi"<<endl;
	
	vector<ComplexNumber> contoursComplex; //Complex Number vectoru olusturduk
	int r1, i1;
	
	for(int j=0; j<contour->total; j++)  // contour kordinatlarýný vectorde karmaþýk olarak tuttuk  1+2j gibi
	{  
		ComplexNumber temp(pt[j].x, pt[j].y);
		//temp.print();
		contoursComplex.push_back(temp);	
		// copy constructor ve assignment operator implemente et çünkü push_back bunlarý kullanýyor. Ve destructor.
	}
	//contoursComplex ile chainCode u hesapla...




	//<Shifting the contour coordinates to center> 
	ComplexNumber shiftC;
	ComplexNumber total(0,0);
	for(int x=0; x<contour->total; x++)
	{
		total=total.add(contoursComplex[x]);
	}
	
	ComplexNumber divisor(contour->total, 0);
	shiftC=total.div(divisor);
	
	vector<ComplexNumber> T;
	
	for(int x=0; x<contour->total; x++)
	{
		ComplexNumber tmp;
		tmp=contoursComplex[x].sub(shiftC);
		T.push_back(tmp);
		//T.at(x)=tmp;
	}
	//</Shifting the contour coordinates to center> 

	//BURAYA KADAR TRANSITION OLAYI HALLOLDU, BÜTÜN NOKTALAR SANAL BÝR ORJÝNE GÖRE ALINDI.

	sortComplexVector(T);// açýlara gore sýralamayý yapýyor. Ufak tefek sapmalar var, elle yazmak yerine quicksort falan yap.

	//FFT KISMINI BURADA YAPIYORUM, MATEMATIKSEL ISLEMLERE GORE. DFT nin MATEMATIKSEL TANIMINA BAKILARAK BURADAKI ISLEM ANLASILABILIR
	ComplexNumber tempSum;
	vector<ComplexNumber> fourierDesc;
	ComplexNumber div(T.size(), 0);
	for(int u=0; u<T.size(); u++)
	{
		tempSum.setReal(0);
		tempSum.setImag(0);

		for(int k=0; k<T.size(); k++)
		{
			ComplexNumber tempComplex(cos((2*360*u*k*PI)/(180*T.size())), (-1)*(sin((2*360*u*k*PI)/(T.size()*180))));
			tempSum=tempSum.add((T.at(k)).mult(tempComplex));
		}
		
		fourierDesc.push_back((tempSum.div(div)));
	}
	//FFT KISMI BURADA BITIYOR
	

	// NORMALIZATION KISMI YAPILIYOR BURADA, SCALE ,TRANSITION VE ROTATION INVARIANT OLUYOR BOYLECE
	vector<ComplexNumber> tmpFourierDesc;
	for(int i=0; i<fourierDesc.size(); i++)
		tmpFourierDesc.push_back(fourierDesc.at(i));
	
	ComplexNumber zero(0,0);
	//Translation Invariance:
	tmpFourierDesc.at(0)=zero;
	//si=abs(T(1)) sayisini tutuyorum
	//Scale Invariance:
	ComplexNumber myVarSiDiv(sqrt(pow(tmpFourierDesc.at(1).getReal(),2) + pow(tmpFourierDesc.at(1).getImag(),2)), 0);

	//T(i)=T(i)/si iþlemi yapýlýyor.
	for(int m=0; m<tmpFourierDesc.size(); m++)
		tmpFourierDesc.at(m).div(myVarSiDiv);

	//Son olarak tum elemanlarý double vectorunde tutuyorum. Boylece fourier desc lar cýktý.
	//vector<double> finalFourierDesc;
	//Burada da T=abs(T) yi fourier descriptor olarak tuttum, böylece ROTATION AND CHANGES IN STARTING POINT oldu.
	//myFourierDescriptors vectoru, class içinde tuttuðum en sonki fourier descriptorlarý içeren vectordur. 
	for(int n=0; n<tmpFourierDesc.size(); n++)
	{
		myFourierDescriptors.push_back(sqrt(pow(tmpFourierDesc.at(n).getReal(),2) + pow(tmpFourierDesc.at(n).getImag(),2)));
		//cout<<n<<". element: "<<myFourierDescriptors.at(n)<<endl;
	}
	/*
	if(tmpFourierDesc.size()<fourierSize)
	{
		int substruct=fourierSize-tmpFourierDesc.size();
		for(int j=0; j<substruct; j++)
			myFourierDescriptors.push_back(0);
	
	
	}*/

};

//Bubble sort
void ExtractDescriptorHelper::sortVector()
{
	double temp1;
	for(int pass=0; pass<myFourierDescriptors.size()-1; pass++)
	{
		for(int j=0; j<myFourierDescriptors.size()-pass-1; j++)
		{
			if(myFourierDescriptors.at(j)>myFourierDescriptors.at(j+1))
			{
				temp1=myFourierDescriptors.at(j);
				myFourierDescriptors.at(j)=myFourierDescriptors.at(j+1);
				myFourierDescriptors.at(j+1)=temp1;
			}
		}
	}

};

//CREATION OF FINAL FEATURE VECTOR
void ExtractDescriptorHelper::createFeatureVector()
{
	
	myFeatureVector.push_back(convexRatio);
	myFeatureVector.push_back(isoperimetricQuotient);
	myFeatureVector.push_back(eccentricity);
	myFeatureVector.push_back(aspectRatio);
	myFeatureVector.push_back(rectangularity);
	myFeatureVector.push_back(formFactor);
	myFeatureVector.push_back(compactness);
	//myFeatureVector.push_back(numberOfToothPoints);
	//myFeatureVector.push_back(perimeter);
	myFeatureVector.push_back(elongatedness);
	myFeatureVector.push_back(nonConvexRatio);
	
	/*
	myFeatureVector.push_back(0);
	myFeatureVector.push_back(0);
	myFeatureVector.push_back(0);
	myFeatureVector.push_back(0);
	myFeatureVector.push_back(0);
	myFeatureVector.push_back(0);
	myFeatureVector.push_back(0);
	//myFeatureVector.push_back(numberOfToothPoints);
	//myFeatureVector.push_back(perimeter);
	myFeatureVector.push_back(0);
	myFeatureVector.push_back(0);
	*/

	for(int i=0; i<fourierSize; i++)
	{
		if(i==1)
			continue;
		myFeatureVector.push_back(myFourierDescriptors.at(i));
		//myFeatureVector.push_back(0);
	}
	/*
	for(int j=0; j<myFeatureVector.size(); j++)
		cout<<j<<". feature: "<<myFeatureVector.at(j)<<endl;*/
	
};

vector<double> ExtractDescriptorHelper::getMyFeatureVector()
{
	return myFeatureVector;
};


float ExtractDescriptorHelper::isLeft(Point2f P0, Point2f P1, Point2f P2)
{
	return (P1.x - P0.x)*(P2.y - P0.y) - (P2.x - P0.x)*(P1.y - P0.y);
};


int ExtractDescriptorHelper::getNumberOfToothPoints()
{
	return numberOfToothPoints;
};

double ExtractDescriptorHelper::getPerimeter()
{
	return perimeter;
};

double ExtractDescriptorHelper::getIsoperimetricQuotient()
{
	return isoperimetricQuotient;
};

double ExtractDescriptorHelper::getCompactness()
{
	return compactness;
};

double ExtractDescriptorHelper::getConvexHullArea()
{
	return convexHullArea;
};

double ExtractDescriptorHelper::getConvexRatio()
{
	return convexRatio;
};

double ExtractDescriptorHelper::getAspectRatio()
{
	return aspectRatio;
};

double ExtractDescriptorHelper::getRectangularity()
{
	return rectangularity;
};

double ExtractDescriptorHelper::getEccentricity()
{
	return eccentricity;
};
		
double ExtractDescriptorHelper::getFormFactor()
{
	return formFactor;
};

double ExtractDescriptorHelper::getElongatedness()
{
	return elongatedness;
};

double ExtractDescriptorHelper::getNonConvexRatio()
{
	return nonConvexRatio;
};


