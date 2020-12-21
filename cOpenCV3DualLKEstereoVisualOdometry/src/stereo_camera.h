/*
 * stereo_camera.h
 *
 *  Created on: 19 Dec 2020
 *      Author: Francisco Dominguez
 */

#pragma once
class StereoCamera{
public:
	// Left = 1 and Right = 2
	// Intrinsic matrix and distortion parameter for each camera
	Mat KL,distL;
	Mat KR,distR;
	Mat R,T;
	Mat E,F;
	Mat R1,R2,P1,P2,Q;
	//for a horizontal camera should be T.at<double>(0)
	double baseLine;
	Size imageSize;
	// Common rectified intrinsic matrix
	Mat K;
	//Create transformation and rectify maps
	Mat cam1map1, cam1map2;
	Mat cam2map1, cam2map2;

	StereoCamera(string s){
		load(s);
	}
	StereoCamera(){
		//default Kitty parameters
		double f = (double)(7.188560000000e+02 + 7.188560000000e+02)/2;
		Point2f pp=Point2f((float)6.071928000000e+02, (float)1.852157000000e+02);
		double baseLine=3.861448000000e+02/f*16;//base = -P2_roi(1,4)/P2_roi(1,1)
		setParams(f,pp,baseLine);
	}
	StereoCamera(double f,Point2f pp,double baseline){
		setParams(f,pp,baseline);
	}
	StereoCamera(Mat KLp,Mat distLp,
			     Mat KRp,Mat distRp,
				 Mat Rp,Mat Tp,Mat Ep,Mat Fp){
		setParams(KLp,distLp,KRp,distRp,Rp,Tp,Ep,Fp);
	}
	void save(string out_file){
		  FileStorage fs(out_file, FileStorage::WRITE);
		  if(!fs.isOpened())
			  throw runtime_error("File not opened in StereoCamera::save "+out_file);
		  fs << "KL" << KL;
		  fs << "KR" << KR;
		  fs << "DL" << distL;
		  fs << "DR" << distR;
		  fs << "R" << R;
		  fs << "T" << T;
		  fs << "E" << E;
		  fs << "F" << F;
	}
	void load(string in_file){
		  FileStorage fs(in_file, FileStorage::READ);
		  if(!fs.isOpened())
			  throw runtime_error("File not opened in StereoCamera::load "+in_file);
		  fs["KL"]>>KL;
		  fs["DL"]>>distL;
		  fs["KR"]>>KR;
		  fs["DR"]>>distR;
		  fs["R"] >>R;
		  fs["T"] >>T;
		  fs["E"] >>E;
		  fs["F"] >>F;
		  setParams(KL,distL,KR,distR,R,T,E,F);
	}
	void setParams(double f,Point2f pp,double baseline){
	    baseLine=baseline;
        //Intrinsic kitti values 00
	    double cx=pp.x;
	    double cy=pp.y;
	    double Tx=baseLine;
	    //Intrinsic Matrix
		K = (Mat_<double>(3, 3) << f   ,  0.00, cx,
				                   0.00,  f   , cy,
								   0.00,  0.00, 1.00);
		//reprojection Matrix
		Q = (Mat_<double>(4, 4) << 1.00,  0.00, 0.00, -cx,
				                   0.00,  1.00, 0.00, -cy,  // turn points 180 deg around x-axis,
								   0.00,  0.00, 0.00,  f,     // so that y-axis looks up
								   0.00,  0.00, 1./Tx,  0);
	}
	void setParams(Mat KLp,Mat distLp,
		     	   Mat KRp,Mat distRp,
			       Mat Rp,Mat Tp,Mat Ep,Mat Fp){
	KL=KLp; distL=distLp;
	KR=KRp; distR=distRp;
	R=Rp; T=Tp;
	E=Ep, F=Fp;
	// but could be P1.at<double>(0,3)??
	baseLine=Tp.at<double>(0,0);
	int rectivy_scale=0; // 0=full_crop, 1=no_crop
	stereoRectify(KL, distL, KR, distR, imageSize, R, T, R1, R2, P1, P2, Q,CALIB_ZERO_DISPARITY,rectivy_scale);
	K=(Mat_<double>(3, 3) <<
		P1.at<double>(0,0),P1.at<double>(0,1),P1.at<double>(0,2),
		P1.at<double>(1,0),P1.at<double>(1,1),P1.at<double>(1,2),
		P1.at<double>(2,0),P1.at<double>(2,2),P1.at<double>(2,2));
    // Compute undistortion and rectification mapping
	initUndistortRectifyMap(KL, distL, R1, P1, imageSize , CV_16SC2, cam1map1, cam1map2);
	initUndistortRectifyMap(KR, distR, R2, P2, imageSize , CV_16SC2, cam2map1, cam2map2);
	}
	Mat rectifyLeft(Mat imgL){
		Mat leftStereoUndistorted; //Create matrices for storing rectified images
		//Rectify and undistort images
		remap(imgL, leftStereoUndistorted, cam1map1, cam1map2, INTER_LINEAR);
		return leftStereoUndistorted;
	}
	Mat rectifyRight(Mat imgR){
		Mat rightStereoUndistorted; //Create matrices for storing rectified images
		//Rectify and undistort images
		remap(imgR, rightStereoUndistorted, cam2map1, cam2map2, INTER_LINEAR);
		return rightStereoUndistorted;
	}
	Mat getLeftFromStereo(Mat imgS){
		int w=imgS.cols>>1;
		Rect r(0,0,w,imgS.rows);
		Mat imgL(imgS(r));
		return imgL;
	}
	Mat getRightFromStereo(Mat imgS){
		int w=imgS.cols>>1;
		Rect r(w,0,w,imgS.rows);
		Mat imgR(imgS(r));
		return imgR;
	}
};




