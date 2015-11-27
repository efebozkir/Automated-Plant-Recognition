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

#include "NormaliseGeoFeatures.h"

#define first_N 2


using namespace std;
using namespace cv;

void NormaliseGeoFeatures::initializeMaxMin()
{
	for(int a=0; a<numberOfElements; a++)
	{
		maxArray[a]=0;
		minArray[a]=1000;
	}
};

double* NormaliseGeoFeatures::getMaxArray()
{
	return maxArray;
};

double* NormaliseGeoFeatures::getMinArray()
{
	return minArray;
};

void NormaliseGeoFeatures::writeMaxMinToFile()
{
	//DIR *dir;
	//struct dirent *ent;
	char writeMaxFile[100];
	char writeMinFile[100];
	
				sprintf (writeMaxFile, "D:/maxOfFeatures.txt");
				sprintf (writeMinFile, "D:/minOfFeatures.txt");
			
				ofstream myfileMax;
					myfileMax.open (writeMaxFile);
					for(int k=0; k<numberOfElements; k++)
						myfileMax << maxArray[k]<<"\n";
					myfileMax.close();

				ofstream myfileMin;
					myfileMin.open (writeMinFile);
					for(int k=0; k<numberOfElements; k++)
						myfileMin << minArray[k]<<"\n";
					myfileMin.close();
};

int NormaliseGeoFeatures::normalizeGeoFeatures()
{
	DIR *dir;
	struct dirent *ent;
	string str1="C:/Features/.";
	string str2="C:/Features/..";
	char folder[100];
	char toBeWritten[100];
	double updatedFeatures[numberOfElements];
	
	if ((dir = opendir ("C:/Features/"))) 
		{
			while ((ent = readdir (dir)) != NULL) 
			{
				sprintf (folder, "C:/Features/%s", ent->d_name);
				sprintf (toBeWritten, "C:/NormalizedFeatures/%s", ent->d_name);
				//cout<<"File name:"<<folder<<endl;
					
				if(str1.compare(folder)==0 || str2.compare(folder)==0)
					continue;
					
				ifstream myfile;
					
					myfile.open(folder);
						for(int i=0; i<numberOfElements; i++)
						{
							myfile>>updatedFeatures[i];
							//cout << nonNormalizedFeatures[i] << endl;
						}

					////ÇIKARILMIÞ FEATURELARIN NORMALÝZASYON ÝÞLEMÝ BURADA BAÞLIYOR.

						for(int k=0; k<numberOfElements; k++)
						{
							if(k==7)   /////elongatedness düzgünleþtirilince silinmeli
								continue;

							updatedFeatures[k]=(updatedFeatures[k]-minArray[k])/(maxArray[k]-minArray[k]);
						
						}
					////HER RESÝM ÝÇÝN FEATURE VECTORLERÝ ARTIK NORMALIZED OLDU. AYRI BÝR FOLDER A YAZILMALI.


					ofstream writtenFile;
					writtenFile.open (toBeWritten);
					for(int k=0; k<numberOfElements; k++)
						writtenFile << updatedFeatures[k]<<"\n";
					writtenFile.close();

			}
		closedir (dir);
		return 0;
		} 
		
		else {
			
			cout<<"Error exists"<<endl;
			perror ("");
			return EXIT_FAILURE;
			}

};

void NormaliseGeoFeatures::produceFileNamesVect()
{
	string str1=".";
	string str2="..";
	DIR *dir;
	struct dirent *ent;
	string a_name, a_name1;
	int i=0;
	bool flag=false;
	if ((dir = opendir ("C:/Features/"))) 
		{
			while ((ent = readdir (dir)) != NULL) 
			{
					
					if(str1.compare(ent->d_name)==0 || str2.compare(ent->d_name)==0)
						continue;
			
					a_name1=ent->d_name;
					a_name=a_name1.substr(0, first_N);
	
					if(flag==false)
					{
						fileNamesVect.push_back(a_name);
						i++;
						flag=true;
					}
					else 
					{
						if(fileNamesVect.at(i-1)!=a_name)
						{
							fileNamesVect.push_back(a_name);
							i++;
						}
					}
			}
		closedir (dir);
		} 
		
	else {
			cout<<"Error exists"<<endl;
			perror ("");
		}
	/*
	for(int j=0; j<fileNamesVect.size(); j++)
		cout<<"Name of type: "<<fileNamesVect.at(j)<<endl;
	*/
};

void NormaliseGeoFeatures::calcMaxMin()   //Duzeltilmeli
{
	DIR *dir;
	struct dirent *ent;
	char folder[100];
	double tmpReadArray[numberOfElements];
	string str1="C:/Features/.";
	string str2="C:/Features/..";
	string maxNameArray[numberOfElements];
	string minNameArray[numberOfElements];


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
						for(int i=0; i<numberOfElements; i++)
						{
							myfile>>tmpReadArray[i];

							if(maxArray[i]<tmpReadArray[i])
							{
								maxArray[i]=tmpReadArray[i];
								maxNameArray[i]=ent->d_name;
							}
							if(minArray[i]>tmpReadArray[i])
							{
								minArray[i]=tmpReadArray[i];
								minNameArray[i]=ent->d_name;
							}
							//cout << tmpReadArray[i] << endl;
						}
			}
		closedir (dir);
		
		} 
		
	else {
		
		cout<<"Error exists"<<endl;
		perror ("");
		
		}
	/*
	for(int l=0; l<numberOfElements; l++)
	{
		cout<<"Max "<<l<<". eleman= "<<maxArray[l]<<" Name: "<<maxNameArray[l]<<endl;
		cout<<"Min: "<<l<<". eleman= "<<minArray[l]<<endl<<" Name: "<<minNameArray[l]<<endl;
		cout<<endl;
	}
	*/
};
