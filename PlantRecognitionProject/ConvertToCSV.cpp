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
#include "ConvertToCSV.h"

using namespace std;
using namespace cv;

void ConvertToCSV::TrainFeaturesAsCSV()
{
	DIR *dir;
	struct dirent *ent;
	char folder[100];
	string str1="C:/Features/.";
	string str2="C:/Features/..";
	double readArray[arraySize];

	if ((dir = opendir ("C:/Features/"))) 
		{
			while ((ent = readdir (dir)) != NULL) 
			{

					sprintf (folder, "C:/Features/%s", ent->d_name);
					//cout<<"File name:"<<folder<<endl;
					
					if(str1.compare(folder)==0 || str2.compare(folder)==0)
						continue;
					
					ifstream myfile;
					
					myfile.open(folder);
						for(int i=0; i<arraySize; i++)
						{
							myfile>>readArray[i];
						}

					string cutTheName=ent->d_name;
					string splitter="_";
					size_t found=cutTheName.find(splitter);
					string trainTypeName=cutTheName.substr(0,found); //Train resminin tam tip adý
						
					ofstream myCsvFile;
					myCsvFile.open("C:/CSV-Features/trainSet.csv", ios::out | ios::app);
					

					for(int j=0; j<arraySize; j++)
					{
						myCsvFile<<readArray[j]<<",";
					}
					myCsvFile<<trainTypeName;
					myCsvFile<<"\n";
					myCsvFile.close();


			}
		closedir (dir);
		
		} 
		
	else {
			cout<<"Error exists"<<endl;
			perror ("");
		 }

};

void ConvertToCSV::TestFeaturesAsCSV()
{
	
	DIR *dirTest;
	struct dirent *entTest;
	string str3="C:/TestSet/.";
	string str4="C:/TestSet/..";
	char testFileName[100];
	
	if ((dirTest = opendir ("C:/TestSet/"))) 
	{
		while ((entTest = readdir (dirTest)) != NULL) 
		{
					sprintf (testFileName, "C:/TestSet/%s", entTest->d_name);
		
					if(str3.compare(testFileName)==0 || str4.compare(testFileName)==0)
						continue;
							
					IplImage* src;
					src=cvLoadImage(testFileName);
					ExtractDescriptorHelper* tmpDescriptorFinder=new ExtractDescriptorHelper();
					tmpDescriptorFinder->ExtractDescriptors(src);
					
					tmpDescriptorFinder->sortVector();
					tmpDescriptorFinder->createFeatureVector();


					string cutTheName=entTest->d_name;
					string splitter="_";
					size_t found=cutTheName.find(splitter);
					string testTypeName=cutTheName.substr(0,found); //Test resminin tam tip adý

					ofstream myCsvFile;
					myCsvFile.open("C:/CSV-Features/testSet.csv", ios::out | ios::app);
					

					for(int j=0; j<arraySize; j++)
					{
						myCsvFile<<tmpDescriptorFinder->getMyFeatureVector().at(j)<<",";
					}
					myCsvFile<<testTypeName;
					myCsvFile<<"\n";
					myCsvFile.close();


		} ///testSet in okunmasýnýn while ýnýn bitmesi
			closedir (dirTest);
	} 	
	else 
	{
		cout<<"Error exists"<<endl;
		perror ("");
	}


};