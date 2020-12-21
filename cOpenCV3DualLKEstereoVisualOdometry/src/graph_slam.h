/*
 * graph_slam.h
 *
 *  Created on: 21 Dec 2020
 *      Author: Francisco Dominguez
 */

#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef Mat Matrix;

float &operator()(Matrix m,int i,int j){m.at<float>(i,j);}

class GraphSLAM{
	int N;
	int L;
	int subDim;
	int dim;
	Matrix Omega;
	Matrix Xi;
public:
	int getDim(){
		int d=(N+L)*subDim;
		return d;
	}
	int getN(){return N;}
	int getL(){return L;}
	void setSubmatrix(int pn,int pm,float dataOmega,Matrix dataXi){
		int n=subDim*pn;
		int m=subDim*pm;
		for(int b=0;b<subDim;b++){
			Omega(n+b,n+b)+= dataOmega;
			Omega(m+b,m+b)+= dataOmega;
			Omega(n+b,m+b)+=-dataOmega;
			Omega(m+b,n+b)+=-dataOmega;
			Xi(n+b,0)+=-dataXi(b,0);
			Xi(m+b,0)+= dataXi(b,0);
		}
	}
	void allocPose(){

	}
	void allocMeasurement(){

	}
	void setPose2PoseEdge(int ps,int pe,Matrix values,float noise=1){
		if(ps>=N) throw runtime_error("Pose source must be a existing node");
		if(pe==N) allocPose();
		if(pe> N) throw runtime_error("Pose end must exists or be N+1 in order to insert new node");
		setSubmatrix(ps,pe,1.0/noise,values/noise);
	}
	void setPose2MeasurementEdge(int ps,int pe,Matrix values,float noise=1){
		if(ps>=N) throw runtime_error("Pose source must be an existing node");
		if(pe==N) allocMeasurement();
		if(pe> N) throw runtime_error("Meaurement end must exists or be L+1 in order to insert new node");
		setSubmatrix(ps,N+pe,1.0/noise,values/noise);
	}
	Matrix solve(){
		Mat mu=Omega.inv()*Xi;
		return mu;
	}
};



