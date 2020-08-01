/*
 * DualLKEstereoVisualOdometry.h
 * Improve DualEstereoVisualodometry with Lucas-Kanade optical flow 2D point traking
 *
 *  Created on: Jun 19, 2016
 *      Author: Francisco Dominguez
 */

#ifndef DUALLKESTEREOVISUALODOMETRY_H_
#define DUALLKESTEREOVISUALODOMETRY_H_
#include <sstream>
#include <fstream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/viz.hpp>
#include "rigid_transformation.h"
#include "binned_detector.h"
#include "opencv3_util.h"

using namespace std;
using namespace cv;

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
class DualLKEstereoVisualOdometry {
public :
	Mat curFrameL,curFrameL_c, prevFrameL,prevFrameL_c, curFrameL_kp, prevFrameL_kp;
	Mat curFrameR,curFrameR_c, prevFrameR,prevFrameR_c, curFrameR_kp, prevFrameR_kp;
	vector<KeyPoint> curKeypointsL, prevKeypointsL, curGoodKeypointsL, prevGoodKeypointsL;
	vector<KeyPoint> curKeypointsR, prevKeypointsR, curGoodKeypointsR, prevGoodKeypointsR;
	Mat curDescriptorsL, prevDescriptorsL;
	Mat curDescriptorsR, prevDescriptorsR;
	vector<Point2f> curPointsL,prevPointsL;
	vector<Point2f> curPointsR,prevPointsR;
    vector<Point2f> cur2Dpts,prev2Dpts;//temporal variables
	vector<DMatch> goodMatchesL;
	vector<DMatch> goodMatchesR;

	//BinnedDetector bd;
	BinnedGoodFeaturesToTrack bd;
	Mat descriptors_1, descriptors_2,  img_matches;
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	Ptr<flann::IndexParams> indexParams;
	Ptr<flann::SearchParams> searchParams;
	Ptr<DescriptorMatcher> matcher;
	// relative scale
	double scale;
	// Rectify CAMERA PARAMS
	// after rectifying both cameras have same parameters
	//double f = (double)(8.941981e+02 + 8.927151e+02)/2;
	//Point2f pp((float)6.601406e+02, (float)2.611004e+02);
	double f ; // focal length in pixels as in K intrinsic matrix
	Point2f pp; //principal point in pixel
	Mat K; //intrinsic matrix
	//global rotation and translation
	Mat Rgl, tgl,Rglprev,tglprev;
	Mat Rgr, tgr,Rgrprev,tgrprev;
	Mat Rg, tg,Rgprev,tgprev;
	//local rotation and transalation from prev to cur
	Mat Rll, tll;
	Mat Rlr, tlr;
	Mat Rl, tr;
	//STEREO DATA
	Mat curDisp,prevDisp;
	Mat curPointCloud,prevPointCloud;
	//Stereo Matches
	Ptr<StereoMatcher> sm;
	//Reprojection Matrix from 2D (u,v) and disp to 3D (X,Y,Z)
	Mat Q;
	double baseLine;
	//temp attributes
	vector<vector<DMatch> > matches;
	Mat E,mask;
	vector<Point3f> prev3Dpts,cur3Dpts;
	//Historic data
	vector< vector<Point2f> > pts2D;
	vector< vector<Point3f> > pts3D;

	//Visualization objects
	Mat cpImg;
    viz::Viz3d myWindow;
//public:
	DualLKEstereoVisualOdometry():bd(3,10,10),myWindow("Coordinate Frame"){
        // relative scale
        scale = 1.0;
        //f = (double)(8.941981e+02 + 8.927151e+02)/2;
		//pp((float)6.601406e+02, (float)2.611004e+02);
        //Intrinsic kitti values 00
		f = (double)(7.188560000000e+02 + 7.188560000000e+02)/2;
		pp=Point2f((float)6.071928000000e+02, (float)1.852157000000e+02);
		baseLine=3.861448000000e+02/f*16;//base = -P2_roi(1,4)/P2_roi(1,1)
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
	DualLKEstereoVisualOdometry(double f,Point2f pp,double baseLine):f(f),pp(pp),baseLine(baseLine),bd(3,10,10),myWindow("Coordinate Frame"){
        // relative scale
        scale = 1.0;
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
/*	DualLKEstereoVisualOdometry(Mat &pcurFrameL_c,Mat &pcurFrameR_c):bd(3,10,10),myWindow("Coordinate Frame"){
        // relative scale
        scale = 1.0;
        //f = (double)(8.941981e+02 + 8.927151e+02)/2;
		//pp((float)6.601406e+02, (float)2.611004e+02);
        //Intrinsic kitti values 00
		f = (double)(7.188560000000e+02 + 7.188560000000e+02)/2;
		pp=Point2f((float)6.071928000000e+02, (float)1.852157000000e+02);
		baseLine=3.861448000000e+02/f*16;//base = -P2_roi(1,4)/P2_roi(1,1)
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
		init(pcurFrameL_c,pcurFrameR_c);
	}*/
	void init(Mat &pcurFrameL_c,Mat &pcurFrameR_c){
			cout << pcurFrameL_c.rows << ","<<pcurFrameL_c.cols<<endl;
			pcurFrameL_c.copyTo(curFrameL_c);
			pcurFrameR_c.copyTo(curFrameR_c);
			cvtColor(curFrameL_c, curFrameL, CV_BGR2GRAY);
			cvtColor(curFrameR_c, curFrameR, CV_BGR2GRAY);

			//detector = ORB::create(1000);//number of features to detect
			//extractor = ORB::create();
			//indexParams = makePtr<flann::LshIndexParams> (6, 12, 1);
			//searchParams = makePtr<flann::SearchParams>(50);
			//matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);

			//detector->detect(curFrameL, curKeypointsL);
			bd.binnedDetection(curFrameL, curKeypointsL);
			curPointsL.clear();
			KeyPoint::convert(curKeypointsL, curPointsL);
	        //cornerSubPix(curFrameL, curPointsL, subPixWinSize, Size(-1,-1), termcrit);
			//detector->detect(curFrameR, curKeypointsR);
			bd.binnedDetection(curFrameR, curKeypointsR);
			//extractor->compute(curFrameL, curKeypointsL, curDescriptorsL);
			//extractor->compute(curFrameR, curKeypointsR, curDescriptorsR);
			//Global transformations
			Rg  = (Mat_<double>(3, 3) << 1., 0., 0.,
					                     0., 1., 0.,
										 0., 0., 1.);
			tg  = (Mat_<double>(3, 1) << 0., 0., 0.);
			Rgl = (Mat_<double>(3, 3) << 1., 0., 0.,
					                     0., 1., 0.,
										 0., 0., 1.);
			tgl = (Mat_<double>(3, 1) << 0., 0., 0.);
			Rgr = (Mat_<double>(3, 3) << 1., 0., 0.,
					                     0., 1., 0.,
										 0., 0., 1.);
			tgr = (Mat_<double>(3, 1) << 0., 0., 0.);
			//Local transformations
			Rll = (Mat_<double>(3, 3) << 1., 0., 0.,
					                     0., 1., 0.,
										 0., 0., 1.);
			tll = (Mat_<double>(3, 1) << 0., 0., 0.);
			Rlr = (Mat_<double>(3, 3) << 1., 0., 0.,
					                     0., 1., 0.,
										 0., 0., 1.);
			tlr = (Mat_<double>(3, 1) << 0., 0., 0.);

			// Stereo 3D data extraction
			sm=StereoBM::create(16*6,9);//9);
			sm->compute(curFrameL,curFrameR,curDisp);
	        reprojectImageTo3D(curDisp, curPointCloud, Q, true);
	        //bilateralFilterPointCloud(curPointCloud);
			//Mix disparity and current left frame
	        Mat cdisp8;
			Mat cdispC8;
			curDisp.convertTo(cdisp8,CV_8U);
			cvtColor(cdisp8,cdispC8,CV_GRAY2BGR);
			addWeighted(cdispC8,0.2,curFrameL_c,0.8,0.0,cdispC8);
			imshow("test",cdispC8);
			//waitKey(-1);

	        //Visualization
	    	//Display variables
	        myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
	        viz::WLine axis(Point3f(-1.0f,-1.0f,-1.0f), Point3f(1.0f,1.0f,1.0f));
	        axis.setRenderingProperty(viz::LINE_WIDTH, 1.0);
	        myWindow.showWidget("Line Widget", axis);
	        viz::WCube cube_widget(Point3f(0.5,0.5,0.0), Point3f(0.0,0.0,-0.5), true, viz::Color::blue());
	        cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);
	        myWindow.showWidget("Cube Widget", cube_widget);
	}
	virtual ~DualLKEstereoVisualOdometry(){}
	// function performs ratiotest
	// to determine the best keypoint matches
	// between consecutive poses
	void ratioTest(vector<vector<DMatch> > &matches, vector<DMatch> &good_matches) {
		for (vector<vector<DMatch> >::iterator it = matches.begin(); it!=matches.end(); it++) {
			if (it->size()>1 ) {
				if ((*it)[0].distance/(*it)[1].distance > 0.6f) {
					it->clear();
				}
			} else {
				it->clear();
			}
			if (!it->empty()) good_matches.push_back((*it)[0]);
		}
	}
	inline void findGoodMatches(vector<KeyPoint> &keypoints_1,Mat &descriptors_1,
			                    vector<KeyPoint> &keypoints_2,Mat &descriptors_2,
						        vector<DMatch>   &good_matches,
						        vector<KeyPoint> &good_keypoints_1, vector<KeyPoint> &good_keypoints_2){
		matches.clear();
		good_matches.clear();

		try {
			matcher->knnMatch(descriptors_1, descriptors_2, matches, 2);
			ratioTest(matches, good_matches);
		} catch(Exception &e) {
			//cerr << "knnMatch error"<<endl;;
		}

		good_keypoints_1.clear();
		good_keypoints_2.clear();
		for ( size_t m = 0; m < good_matches.size(); m++) {
			int i1 = good_matches[m].queryIdx;
			int i2 = good_matches[m].trainIdx;
			CV_Assert(i1 >= 0 && i1 < static_cast<int>(keypoints_1.size()));
            CV_Assert(i2 >= 0 && i2 < static_cast<int>(keypoints_2.size()));
            good_keypoints_1.push_back(keypoints_1[i1]);
            good_keypoints_2.push_back(keypoints_2[i2]);
		}
	}
	inline void getPoseFromEssentialMat(vector<Point2f> &point1, vector<Point2f> &point2,Mat &R,Mat &t){
		E = findEssentialMat(point2, point1, f, pp, RANSAC, 0.999, 1.0,mask);
		recoverPose(E, point2, point1, R, t, f, pp,mask);
	}
	void getUVdispFromKeyPoints(vector<KeyPoint> &kpts,Mat &disp,Mat &uvd){
		vector<Point3f> vp;
		Point3f p;
		for(unsigned int i=0;i<kpts.size();i++){
			KeyPoint &kp=kpts[i];
			float u=kp.pt.x;
			float v=kp.pt.y;
		    float d=disp.at<float>(u,v);
			p.x=u;
			p.y=v;
			p.z=d;
			vp.push_back(p);
		}
		uvd= Mat(vp);
	}
	inline bool is2DpointInFrame(Point2f& p){
		return p.x>=0 && p.x<curFrameL.cols-1 && p.y>=0 && p.y<curFrameL.rows-1;
	}
	void opticalFlowPyrLKTrack(){
	    //Lucas-Kanade parameters
	    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
	    //Size subPixWinSize(10,10), winSize(31,31);
	    Size subPixWinSize(10,10), winSize(21,21);
        vector<uchar> status;
        vector<float> err;
        vector<Point2f> tmpPrevPointsL,trackedPointsL;
		calcOpticalFlowPyrLK(prevFrameL, curFrameL, prevPointsL, trackedPointsL, status, err, winSize,
                             3, termcrit, 0, 0.001);
		tmpPrevPointsL=prevPointsL;
		prevPointsL.clear();
		curPointsL.clear();
		for(size_t i=0;i<status.size();i++){
			if(status[i]){//good tracking
				if(is2DpointInFrame(tmpPrevPointsL[i]) &&
				   is2DpointInFrame(trackedPointsL[i])){//LK can give points not in frame
				prevPointsL.push_back(tmpPrevPointsL[i]);
				 curPointsL.push_back(trackedPointsL[i]);
				}
			}
		}
		//Photometry subpixel improvement
        //cornerSubPix(prevFrameL, prevPointsL, subPixWinSize, Size(-1,-1), termcrit);
        //cornerSubPix( curFrameL,  curPointsL, subPixWinSize, Size(-1,-1), termcrit);
	}

	void selectGoodPoints(){
		int cbad=0,nbad=0,dbad=0;
        prev3Dpts.clear();
        cur3Dpts.clear();
        prev2Dpts.clear();
        cur2Dpts.clear();
        Point3f p3d,c3d;
        for(unsigned int i=0;i<prevPointsL.size();i++){
    		Point2f p2df1=prevPointsL[i];
    		Point2f c2df1= curPointsL[i];
    		Point2f ppx(p2df1.x,p2df1.y+cpImg.rows/2);//pixel of prev in cpImg
    		//cout << "i"<< i <<endl;
    		//cout << "p2df1="<< p2df1 <<endl;
    		//cout << "c2df1="<< c2df1 <<endl;

    		//theses two points would be the same or be very close to each other
        	p3d=prevPointCloud.at<Point3f>(p2df1);
        	c3d= curPointCloud.at<Point3f>(c2df1);
    		//cout << "p3d="<< p3d <<endl;
    		//cout << "c3d="<< c3d <<endl;
        	float pz=p3d.z;
        	float cz=c3d.z;
        	//not disparity points have a z value of 1000
        	if(pz<10000 && cz<10000){
				Point3f dif3D=p3d-c3d;
				float d=sqrt(dif3D.dot(dif3D));
				//d should be velocity/time
				if (true /*d<50e3/36000*/){
					//cout << "dif3D="<<dif3D<<":"<<d<<endl;
					int disp=prevDisp.at<unsigned short>(p2df1)>>4;//16-bit fixed-point disparity map (where each disparity value has 4 fractional bits)
					Point2f p2f2=prevPointsL[i];
					p2f2.x+=disp;
					p2f2.y+=cpImg.rows/2;
					line(cpImg,ppx,p2f2,Scalar(0, 255, 0));
					if(pz<5000.5 && pz>0.0){
						putText(cpImg,toString(pz)+":"+toString(d),ppx,1,1,Scalar(0, 255, 255));
						putText(cpImg,toString(cz),c2df1,1,1,Scalar(255, 255, 0));
						prev3Dpts.push_back(p3d);
						 cur3Dpts.push_back(c3d);
						prev2Dpts.push_back(p2df1);
						 cur2Dpts.push_back(c2df1);
						circle(cpImg,ppx ,8,Scalar(255, 0, 0));
						circle(cpImg,c2df1,8,Scalar(255, 0, 0));
						line(cpImg,ppx,c2df1,Scalar(255,255,255));
					}
					else{
						putText(cpImg,toString(cz),c2df1,1,1,Scalar(0, 64, 255));
						//circle(cpImg,ppx,3,Scalar(0, 128, 255));
						line(cpImg,ppx,c2df1,Scalar(0,   0, 255));
						nbad++;
					}
				}
				else{
					//putText(cpImg,to_string(pz),ppx,1,1,Scalar(0, 0, 255));
					//circle(cpImg,ppx,3,Scalar(0, 0, 255));
					cbad++;
				}
        	}
        	else{
				circle(cpImg,ppx,3,Scalar(255, 0, 255));
        		dbad++;
        	}
        }
        cout <<"#prevPointsL   ="<<prevPointsL.size() << endl;
        cout <<"#cbad distance ="<<cbad<<endl;
        cout <<"#nbad too far  ="<<nbad<<endl;
        cout <<"#dbad disparity="<<dbad<<endl;
        cout <<"#PointsL left  ="<<prevPointsL.size()-cbad-nbad-dbad<<endl;
        cout <<"prev3Dpts="<< prev3Dpts.size() << endl;
        cout <<"cur2DforPnP="<< cur2Dpts.size() << endl;
	}
	void projectionError(Mat &rvec,Mat &tl){
		vector<Point2f> proj2DafterPnP;
		cv::projectPoints(cur3Dpts,rvec,tll,K,Mat(),proj2DafterPnP);
		//cv::transform(cur3Dpts,prevAfterFitting,Rll);
		float tdp=0;
		vector<Point3f> cur3DptsClean,prev3DptsClean;
		vector<Point2f> cur2DptsClean,prev2DptsClean;
		Point2f dif2D;
		//Point3f dif3D;
		for(unsigned int i=0;i<proj2DafterPnP.size();i++){
			//Visualization variables
    		Point2f p2daf1=proj2DafterPnP[i];
    		Point2f p2df1=prev2Dpts[i];
    		Point2f ppxa(p2daf1.x,p2df1.y+cpImg.rows/2);//pixel of prev in cpImg
    		Point2f ppx(p2df1.x,p2df1.y+cpImg.rows/2);//pixel of prev in cpImg
    		//work out varibles
        	//dif3D=prev3Dpts[i]-prevAfterFitting[i];
        	float dp=distPoint2f(prev2Dpts[i],proj2DafterPnP[i]);
        	//float dp3D=sqrt(dif3D.dot(dif3D));
        	tdp+=dp;
        	if(dp>1.0){
        		//cout <<prev2Dpts[i]<<":"<<dp<<"~" <<proj2DafterPnP[i] << endl;
    			line(cpImg,ppx,ppxa,Scalar(0, 0, 255),2);
        	}
        	else{
        		 cur3DptsClean.push_back( cur3Dpts[i]);
        		prev3DptsClean.push_back(prev3Dpts[i]);
        		 cur2DptsClean.push_back( cur2Dpts[i]);
        		prev2DptsClean.push_back(prev2Dpts[i]);
    			line(cpImg,ppx,ppxa,Scalar(0, 255, 255),5);
        	}
		}

 		cur3Dpts= cur3DptsClean;
 		prev3Dpts= prev3DptsClean;
 		curPointsL=cur2DptsClean;
 		prevPointsL=prev2DptsClean;

		//reprojection mean error
		tdp/=proj2DafterPnP.size();
		cout << "tdp0="<< tdp << endl;
		if(cur3DptsClean.size()>0){
			Mat cur3DMat1=Mat( cur3DptsClean).reshape(1);
			//cout << " cur3DMat1=" << cur3DptsClean.size()<< endl;
			Mat prev3DMat1=Mat(prev3DptsClean).reshape(1);
			//cout << "prev3DMat1=" << prev3DptsClean.size() <<endl;
			rigidTransformation(cur3DMat1,prev3DMat1,Rll,tll);
			//cout << "rigidTransformation=" << endl;
			Rodrigues(Rll,rvec);
			proj2DafterPnP.clear();
			cv::projectPoints(cur3DptsClean,rvec,tll,K,Mat(),proj2DafterPnP);
			//cout << "projectPoints=" << endl;
			tdp=0;
			for(unsigned int i=0;i<proj2DafterPnP.size();i++){
				dif2D=prev2DptsClean[i]-proj2DafterPnP[i];
				float dp=sqrt(dif2D.dot(dif2D));
				tdp+=dp;
				//if(dp>2.0){
				//	cout <<prev2Dpts[i]<<":"<<dp<<"~" <<proj2DafterPnP[i] << endl;
				//}
			}
			//reprojection mean error
			tdp/=proj2DafterPnP.size();
			cout << "tdp1="<< tdp << endl;
		}

	}
	//bilateral filter to a pointcloud from a stereo disparity
	//invalid values are 10000
	// it doesn't improve the accuracy or is bad implemented
	void bilateralFilterPointCloud(Mat &pointCloud){
        Mat splitImgs[3];
        split(pointCloud,splitImgs);
        Mat depth_flt=splitImgs[2];
        //Change invalid depth with quiet_NaN
        cout <<"depth_flt.type"<<type2str(depth_flt.type())<<endl;
        Mat invalidDepthMask=depth_flt>=10000.0;

        Mat depth = Mat(depth_flt.size(), CV_32FC1, Scalar(0));
        const double depth_sigma = 0.03*4;
        const double space_sigma = 4.5;  // in pixels
        depth_flt.setTo(-5*depth_sigma, invalidDepthMask);
        bilateralFilter(depth_flt, depth, -1, depth_sigma, space_sigma);
        //depth.setTo(std::numeric_limits<float>::quiet_NaN(), invalidDepthMask);
        depth.setTo(10000.0, invalidDepthMask);
        splitImgs[2]=depth;
        merge(splitImgs,3,pointCloud);
	}
	void showCloud(string widgetName,Mat &Rg,Mat tg){
        Affine3d *camPose=new Affine3d(Rg,tg);
        string s0("CPW");
        string s1("CPW_FRUSTUM");
        viz::WCameraPosition *cpw=new viz::WCameraPosition(0.05); // Coordinate axes
        viz::WCameraPosition *cpw_frustum=new viz::WCameraPosition(Matx33d(K),curFrameL_c,0.91); // Camera frustum
        //myWindow.showWidget(s0+widgetName,*cpw,*camPose);
        //myWindow.showWidget(s1+widgetName,*cpw_frustum,*camPose);
        myWindow.setViewerPose(*camPose);
		Mat pc;
		curPointCloud.copyTo(pc);
		int s=pc.rows*pc.cols;
	    Point3f* data = pc.ptr<cv::Point3f>();
	    for(int i = 0; i < s; ++i){
	    	if(data[i].z<1.0 || data[i].z>=40.0){
	    		//cout <<"NaN"<<endl;
	    		//data[i].z=std::numeric_limits<float>::quiet_NaN();
	    		data[i].z=std::numeric_limits<float>::quiet_NaN();
	    	}
	    }
        //Mat invalidDepthMask=depth_flt>=10000.0;
        //depth.setTo(std::numeric_limits<float>::quiet_NaN(), invalidDepthMask);
      	viz::WCloud *wpc=new viz::WCloud(pc,curFrameL_c);
      	//viz::WCloud *wpc=new viz::WCloud(pc);
        myWindow.showWidget(widgetName+"cloud",*wpc,*camPose);
	}
	void current2Previous(Mat& pcurFrameL_c,Mat& pcurFrameR_c){
		// prev<=cur
		curDisp.copyTo(prevDisp);
		curPointCloud.copyTo(prevPointCloud);
		curFrameL.copyTo(prevFrameL);		      curFrameR.copyTo(prevFrameR);
		curFrameL_c.copyTo(prevFrameL_c);	      curFrameR_c.copyTo(prevFrameR_c);
		prevKeypointsL = curKeypointsL;		      prevKeypointsR = curKeypointsR;
		curDescriptorsL.copyTo(prevDescriptorsL); curDescriptorsR.copyTo(prevDescriptorsR);
		prevPointsL = curPointsL;    		      prevPointsR = curPointsR;
        // New values of current frame
		pcurFrameL_c.copyTo(curFrameL_c);	      pcurFrameR_c.copyTo(curFrameR_c);
		cvtColor(curFrameL_c, curFrameL, CV_BGR2GRAY);		cvtColor(curFrameR_c, curFrameR, CV_BGR2GRAY);
	}
	void buildRelativePose(Mat &rr,Mat &tr){
		// Estimate relative pose from previous to current frame Rll and tll

		//vector<Point3f> prevAfterFitting;
    	//Mat  cur3DMat=Mat( cur3Dpts).reshape(1);
    	//Mat prev3DMat=Mat(prev3Dpts).reshape(1);
        //rigidTransformation(cur3DMat,prev3DMat,Rll,tll);
        //Rodrigues(Rll,rvec);

        //Mat r,t,rr,tr,rri,tri;
        //cv::solvePnP(cur3Dpts,prev2Dpts,K,noArray(),r,t,0,SOLVEPNP_ITERATIVE);

        Mat inliers;
        cv::solvePnPRansac(cur3Dpts,prev2Dpts,K,noArray(),rr,tr,false,100,1.5,0.99,inliers);
        vector<Point3f> prev3DptsIn,cur3DptsIn;
        vector<Point2f> prev2DptsIn, cur2DptsIn;
        for(int i=0;i<inliers.rows;i++){
        	int idx=inliers.at<int>(i,0);
        	 cur3DptsIn.push_back( cur3Dpts[idx]);
        	prev3DptsIn.push_back(prev3Dpts[idx]);
        	prev2DptsIn.push_back(prev2Dpts[idx]);
        	 cur2DptsIn.push_back( cur2Dpts[idx]);
        }
         cur3Dpts=cur3DptsIn;
        prev3Dpts=prev3DptsIn;
         cur2Dpts=cur2DptsIn;
        prev2Dpts=prev2DptsIn;
        cv::solvePnPRansac(cur3Dpts,prev2Dpts,K,noArray(),rr,tr,true,100,1.0,0.99,inliers);
        prev3DptsIn.clear();
         cur3DptsIn.clear();
        prev2DptsIn.clear();
         cur2DptsIn.clear();
        for(int i=0;i<inliers.rows;i++){
        	int idx=inliers.at<int>(i,0);
        	 cur3DptsIn.push_back( cur3Dpts[idx]);
        	prev3DptsIn.push_back(prev3Dpts[idx]);
        	prev2DptsIn.push_back(prev2Dpts[idx]);
        	 cur2DptsIn.push_back( cur2Dpts[idx]);
        }
         cur3Dpts=cur3DptsIn;
        prev3Dpts=prev3DptsIn;
         cur2Dpts=cur2DptsIn;
        prev2Dpts=prev2DptsIn;
	}
	void buildPointCould(){
		//Compute disparity and 3D point cloud
		sm->compute(curFrameL, curFrameR, curDisp);
        reprojectImageTo3D(curDisp, curPointCloud, Q, true);
        //bilateralFilterPointCloud(curPointCloud);

        // TO VISUALISE DISP
		//Mix disparity and previous left frame
        //dispC8 if to be visualised
        Mat disp8;
		Mat dispC8;
        prevDisp.convertTo(disp8, CV_8U);
		cvtColor(disp8,dispC8,CV_GRAY2BGR);
		addWeighted(dispC8,0.2,prevFrameL_c,0.8,0.0,dispC8);
		//Mix disparity and current left frame
        Mat cdisp8;
		Mat cdispC8;
		curDisp.convertTo(cdisp8,CV_8U);
		cvtColor(cdisp8,cdispC8,CV_GRAY2BGR);
		addWeighted(cdispC8,0.2,curFrameL_c,0.8,0.0,cdispC8);
		//Stack previous and current frame in a image to visualize
		cpImg=stackV(cdispC8,dispC8);
	}
	void updatePose(Mat &rr,Mat &tr){
		Mat rvec;//rotation vector
        rvec=rr; tll=tr;
        cv::Rodrigues(rvec,Rll);

        // curPointsL=cur2DptsIn;
        //prevPointsL=prev2DptsIn;
        projectionError(rvec,tll);

        //cv::transform(cur3DMat,prev3DMatAfter,cur3DMat);
		//erase x and z rotations since car only rotate on y
        rvec.at<float>(0.0)=0;
        rvec.at<float>(2.0)=0;

        // Update global pose Rgl and tgl
        Rodrigues(rvec,Rll);
		//Rodrigues(rvec,Rll);
        //cout <<"inliers after fitting PnPRansac="<< inliers.size() << endl;
		cout << "rvec"<< rvec.t() << endl;
		//cout << "Rll"<< Rll << endl;
		//cout << "tll"<< tll << endl;
		//cout << "tgl"<< tgl << endl;
		//cout << "Rgl"<< Rgl << endl;
		//update global transform
		Mat tll64;
		tll.at<float>(0,0)=0;
		tll.at<float>(1,0)=0;
		tll.convertTo(tll64,CV_64FC1);
		Mat dt=Rgl*tll64;
		//tgl = tgl + (Rgl*(scale*tll));
		tgl = tgl + dt;
		Rll.convertTo(Rll,CV_64FC1);
		Rgl = Rll*Rgl;
		//cout << "tgl"<< tgl << endl;
		//cout << "Rgl"<< Rgl << endl;
	}
	void refreshTrackingPoints(){
		//refresh tracking points
		cout << "#curPoints=" <<  curPointsL.size()<<endl;
		if(curPointsL.size()<1950){//match point if there are few points
			//Keypoints detection
			curKeypointsL.clear();
			bd.refreshDetection(curFrameL, curPointsL, curKeypointsL);
			vector<Point2f> rPoints;
			KeyPoint::convert(curKeypointsL, rPoints);
			curPointsL.insert(curPointsL.end(), rPoints.begin(), rPoints.end());
		}
		cout << "#curPoints" <<  curPointsL.size()<<endl;
	}
	void stepStereoOdometry(Mat& pcurFrameL_c,Mat& pcurFrameR_c){
		current2Previous(pcurFrameL_c,pcurFrameR_c);

		buildPointCould();

		opticalFlowPyrLKTrack();

		selectGoodPoints();

		Mat rr,tr;
		buildRelativePose(rr,tr);

        updatePose(rr,tr);

		refreshTrackingPoints();

		//Visualisation 2D
  		resize(cpImg, cpImg, Size(), 0.5,0.5);
        imshow("prevDisp",cpImg);

        //Visualization 3D
        myWindow.spinOnce(1, true);

	}
};

#endif /* DUALLKESTEREOVISUALODOMETRY_H_ */
