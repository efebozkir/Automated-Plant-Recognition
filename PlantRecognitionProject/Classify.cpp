#define _CRT_SECURE_NO_DEPRECATE
#include<iostream>
#include<cstdlib>
#include <fstream>
#include <windows.h>
#include "dirent.h"
#include <string>
#include <cmath>

#include "opencv\cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv\highgui.h"

#include "Classify.h"
#include "ExtractDescriptorHelper.h"
#include "NormaliseGeoFeatures.h"

using namespace std;
using namespace cv;




void Classify::getMinMaxFromFile()
{
	DIR *dir;
	//struct dirent *ent;
	char readMaxFile[100];
	char readMinFile[100];
	

	if ((dir = opendir ("C:/Features/"))) 
	{
				sprintf (readMaxFile, "D:/maxOfFeatures.txt");
				sprintf (readMinFile, "D:/minOfFeatures.txt");
			
				ifstream myfileMax;
					myfileMax.open (readMaxFile);
					for(int k=0; k<numberOfElements; k++)
						myfileMax>> maxFeatures[k];
					myfileMax.close();

				ifstream myfileMin;
					myfileMin.open (readMinFile);
					for(int k=0; k<numberOfElements; k++)
						myfileMin>>minFeatures[k];
					myfileMin.close();

		closedir (dir);
	} 
		
	else 
	{
		cout<<"Error exists"<<endl;
		perror ("");
	}
	
	for(int y=0; y<numberOfElements; y++)
	{
		cout<<"Max of "<<y<<". element: "<<maxFeatures[y]<<endl;
		cout<<"Min of "<<y<<". element: "<<minFeatures[y]<<endl;
		cout<<endl;
	}

};


void Classify::sortDataVect()
{
	DiffvName temp;
    bool finished = false;
    while (!finished)    {
       finished = true;
       for (int i = 0; i < myVect.size()-1; i++) {
		   if (myVect[i].difference > myVect[i+1].difference) {
             temp = myVect[i];
             myVect[i] = myVect[i+1];
             myVect[i+1] = temp;
             finished=false;
          }
        }
     }

};

void Classify::printVector()
{
	for(int a=0; a<10; a++)
	{
		
		cout<<a<<". name: "<<myVect.at(a).name<<endl;
		cout<<a<<". difference"<<myVect.at(a).difference<<endl;
	}
}

int Classify::makeClassification()
{
	DIR *dir, *dirD;
	struct dirent *ent;
	char folder[100];
	char readMaxFile[100];
	char readMinFile[100];
	string topTenNames[10];
	double topTenDifference[10];
	double maximumFeatures[numberOfElements];
	double minimumFeatures[numberOfElements];


	for(int m=0; m<9; m++)
		topTenDifference[m]=1000;

	string str1="C:/Features/.";
	string str2="C:/Features/..";
	string str3="C:/TestSet/.";
	string str4="C:/TestSet/..";
	char testFileName[100];
	DiffvName dataStruct;
	double testImgCounter=0;
	double testImgFoundCounter=0;

	DIR *dirTest;
	struct dirent *entTest;

	if ((dirTest = opendir ("C:/TestSet/"))) 
	{
		while ((entTest = readdir (dirTest)) != NULL) 
		{
					sprintf (testFileName, "C:/TestSet/%s", entTest->d_name);
		
					if(str3.compare(testFileName)==0 || str4.compare(testFileName)==0)
						continue;
					

					cout<<"File name: "<<testFileName<<endl;
					
					testImgCounter=testImgCounter+1;
	IplImage* src;
	src=cvLoadImage(testFileName);
	ExtractDescriptorHelper* tmpDescriptorFinder=new ExtractDescriptorHelper();
	tmpDescriptorFinder->ExtractDescriptors(src);
					
	tmpDescriptorFinder->sortVector();
	tmpDescriptorFinder->createFeatureVector();

	for(int p=0; p<featureVectSize; p++)
		compareTestSetArray[p]=tmpDescriptorFinder->getMyFeatureVector().at(p);

	/////
	/*NORMALÝZE EDÝLMÝÞ VEKTÖRLER ÜZERÝNDEN KONTROL ÝÇÝN BU BÖLÜMÜ KULLANINIZ.
	*////////////
	//!!!!!!!!!!!!!ÞU AN NORMALIZE SAYILAR SONSUZ OLARAK GELÝYOR.
	/*
	for(int t=0; t<numberOfElements; t++)
	{
		if(t==7)
			continue;

		compareTestSetArray[t]=(compareTestSetArray[t]-minimumFeatures[t])/(maximumFeatures[t]-minimumFeatures[t]);
	}


	for(int r=0; r<numberOfElements; r++)
	{
		cout<<"Normalized Test Image Array: " <<r<<" :"<<compareTestSetArray[r]<<endl;
	
	}
	*/
	////////////////

	
	if ((dir = opendir ("C:/Features/"))) 
		{
			
			while ((ent = readdir (dir)) != NULL) 
			{
					dataStruct.difference=0;
					dataStruct.name="x";

					sprintf (folder, "C:/Features/%s", ent->d_name);
					//cout<<"File name:"<<folder<<endl;
					
					if(str1.compare(folder)==0 || str2.compare(folder)==0)
						continue;
					
					ifstream myfile;
					
					myfile.open(folder);
						for(int i=0; i<featureVectSize; i++)
						{
							myfile>>compareTrainSetArray[i];
							//cout << compareTrainSetArray[i] << endl;
							
						}

				
					double tmpSum=0;
					for(int x=0; x<featureVectSize;x++)
					{
						tmpSum=tmpSum+pow((compareTrainSetArray[x]-compareTestSetArray[x]),2);
					}
					tmpSum=sqrt(tmpSum);

					//cout<<"Sayi:"<<tmpSum<<endl;
					
					dataStruct.difference=tmpSum;
					dataStruct.name=ent->d_name;
					
					myVect.push_back(dataStruct);
					
									
			}
		closedir (dir);
		//return 0;
		} 
		
		else {
			cout<<"Error exists"<<endl;
			perror ("");
			//return EXIT_FAILURE;
			}//////TRAINSETTEN OKUNAN HER RESIM ICIN KIYASLAMALAR BURADA BITER; HER RESIM ÝÇÝN VECTORE YAZIM YAPILR.
		
		this->sortDataVect(); //myVect i sýralýyor.  
		////vectorün içinden ilk 10 tipi alacaðýz. bunlar tespit edilmiþ bitki türleri olacak.
		string cutTheName=entTest->d_name;
		string splitter="-";
		size_t found=cutTheName.find(splitter);
		string testTypeName=cutTheName.substr(0,found); //test resminin tam tip adý.
		cout<<"Test name: "<<testTypeName<<endl;
		for(int a=0; a<myVect.size(); a++)
		{
			string cutTheTrainName=myVect.at(a).name;
			size_t trainImgFound=cutTheTrainName.find(splitter);
			string trainTypeName=cutTheTrainName.substr(0, trainImgFound);
			myVect.at(a).name=trainTypeName;
		}

		this->printVector();
		
		
		for(int l=0; l<10; l++)    ////ilk 10 resimde bulursa testImgFoundCounter ý artýr
		{
			if(testTypeName.compare(myVect.at(l).name)==0)
			{
				testImgFoundCounter++;
				break;
			}

		}
		
		myVect.clear(); //vectoru bosaltýyor.
		//cvReleaseImage(&src);
		} ///testSet in okunmasýnýn while ýnýn bitmesi
		closedir (dirTest);
	} 	
	else 
	{
		cout<<"Error exists"<<endl;
		perror ("");
	}

	double successRate= testImgFoundCounter/testImgCounter*100;
	cout<<"Number of test-set images: "<<testImgCounter<<endl;
	cout<<"Number of found images: "<<testImgFoundCounter<<endl;
	cout<<"Success Rate of Test: %"<<successRate<<endl;

		return 0;
};




	/*
					ofstream myfile;
					myfile.open (writeFile);
					for(int k=0; k<59; k++)
						myfile << featureArray[k]<<"\n";
					myfile.close();*/