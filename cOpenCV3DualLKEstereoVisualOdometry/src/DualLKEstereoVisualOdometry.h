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
#include "rigid_transformation.h"

using namespace std;
using namespace cv;

string type2str(int type) {
 string r;

 uchar depth = type & CV_MAT_DEPTH_MASK;
 uchar chans = 1 + (type >> CV_CN_SHIFT);

 switch ( depth ) {
   case CV_8U:  r = "8U"; break;
   case CV_8S:  r = "8S"; break;
   case CV_16U: r = "16U"; break;
   case CV_16S: r = "16S"; break;
   case CV_32S: r = "32S"; break;
   case CV_32F: r = "32F"; break;
   case CV_64F: r = "64F"; break;
   default:     r = "User"; break;
 }

 r += "C";
 r += (chans+'0');

 return r;
}
/*
Mat centroidR(Mat &m){
	Mat t=m.row(0);
	int n=m.rows;
	for(int i=1;i<n;i++){
		t+=m.row(i);
	}
	t/=n;
	return t;
}
Mat centroidC(Mat &m){
	Mat t=m.col(0);
	int n=m.cols;
	for(int i=1;i<n;i++){
		t+=m.col(i);
	}
	t/=n;
	return t;
}
Mat centerR(Mat &M){
	Mat Ret;
	M.copyTo(Ret);
	Mat centroidM(centroidR(M));
	int n=M.rows;
	for(int i=0;i<n;i++){
		Ret.row(i)-=centroidM;
	}
	return Ret;
}
Mat centerC(Mat &M){
	Mat Ret;
	M.copyTo(Ret);
	Mat centroidM(centroidC(M));
	int n=M.cols;
	for(int i=0;i<n;i++){
		Ret.col(i)-=centroidM;
	}
	return Ret;
}

void rigidTransformation(Mat &A,Mat &B,Mat &R,Mat &t){
	/*http://nghiaho.com/?page_id=671
	Mat centroidA,centroidB;
	centroidA=centroidR(A);
	//cout << "centroidA" << endl;
	centroidB=centroidR(B);
	//cout << "centroidB" << endl;
	Mat AA=centerR(A);
	//cout << "AA" << endl;
	//cout << A.rows<<A.cols<<A.channels() << "AA"<<A<<endl;
	Mat BB=centerR(B);
	//cout << "BB" << endl;
	Mat H=AA.t()*BB;
	//cout << "H" << endl;
	Mat w, u, vt;
	SVD::compute(H, w, u, vt);
	//cout << "SVD::compute" << endl;
	R=vt.t()*u.t();
	//cout << "R" << endl;
	if(cv::determinant(R)<0.0){
		//cout << "if(cv::determinant" << endl;
		vt.row(2)*=-1;
		R=vt.t()*u.t();
	}
	t=-R*centroidA.t()+centroidB.t();
}
*/
class DualLKEstereoVisualOdometry {
public :
	string to_string(float i){
		string result;          // string which will contain the result
		ostringstream convert;   // stream used for the conversion
		convert.precision(2);
		convert << i;      // insert the textual representation of 'Number' in the characters in the stream
		result = convert.str();
		return result;
	}
	Mat stackH(Mat im1,Mat im2){
	    Size sz1 = im1.size();
	    Size sz2 = im2.size();
	    Mat im3(sz1.height, sz1.width+sz2.width, CV_8UC3);
	    Mat left(im3, Rect(0, 0, sz1.width, sz1.height));
	    im1.copyTo(left);
	    Mat right(im3, Rect(sz1.width, 0, sz2.width, sz2.height));
	    im2.copyTo(right);
	    //imshow("im3", im3);
	    return im3;
	}
	Mat stackV(Mat im1,Mat im2){
	    Size sz1 = im1.size();
	    Size sz2 = im2.size();
	    Mat im3(sz1.height+sz2.height, sz1.width, CV_8UC3);
	    Mat top(im3, Rect(0, 0, sz1.width, sz1.height));
	    im1.copyTo(top);
	    Mat down(im3, Rect(0, sz1.height, sz2.width, sz2.height));
	    im2.copyTo(down);
	    //imshow("im3", im3);
	    return im3;
	}
	Mat curFrameL,curFrameL_c, prevFrameL,prevFrameL_c, curFrameL_kp, prevFrameL_kp;
	Mat curFrameR,curFrameR_c, prevFrameR,prevFrameR_c, curFrameR_kp, prevFrameR_kp;
	vector<KeyPoint> curKeypointsL, prevKeypointsL, curGoodKeypointsL, prevGoodKeypointsL;
	vector<KeyPoint> curKeypointsR, prevKeypointsR, curGoodKeypointsR, prevGoodKeypointsR;
	Mat curDescriptorsL, prevDescriptorsL;
	Mat curDescriptorsR, prevDescriptorsR;
	vector<Point2f> curPointsL,prevPointsL;
	vector<Point2f> curPointsR,prevPointsR;
	vector<DMatch> goodMatchesL;
	vector<DMatch> goodMatchesR;

	Mat descriptors_1, descriptors_2,  img_matches;
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	Ptr<flann::IndexParams> indexParams;
	Ptr<flann::SearchParams> searchParams;
	Ptr<DescriptorMatcher> matcher;
	// relative scale
	double scale;
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
//public:
	DualLKEstereoVisualOdometry(Mat &pcurFrameL_c,Mat &pcurFrameR_c){
		pcurFrameL_c.copyTo(curFrameL_c);
		pcurFrameR_c.copyTo(curFrameR_c);
		cvtColor(curFrameL_c, curFrameL, CV_BGR2GRAY);
		cvtColor(curFrameR_c, curFrameR, CV_BGR2GRAY);

		sm=StereoBM::create(16*4,15);//9);
		sm->compute(curFrameL,curFrameR,curDisp);
		detector = ORB::create(1000);//number of features to detect
		extractor = ORB::create();
		indexParams = makePtr<flann::LshIndexParams> (6, 12, 1);
		searchParams = makePtr<flann::SearchParams>(50);
		matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);

		//detector->detect(curFrameL, curKeypointsL);
		binnedDetection(curFrameL, curKeypointsL);
		curPointsL.clear();
		KeyPoint::convert(curKeypointsL, curPointsL);
        //cornerSubPix(curFrameL, curPointsL, subPixWinSize, Size(-1,-1), termcrit);
		//detector->detect(curFrameR, curKeypointsR);
		binnedDetection(curFrameR, curKeypointsR);
		extractor->compute(curFrameL, curKeypointsL, curDescriptorsL);
		extractor->compute(curFrameR, curKeypointsR, curDescriptorsR);
		//Global transformations
		Rg  = (Mat_<double>(3, 3) << 1., 0., 0., 0., 1., 0., 0., 0., 1.);
		tg  = (Mat_<double>(3, 1) << 0., 0., 0.);
		Rgl = (Mat_<double>(3, 3) << 1., 0., 0., 0., 1., 0., 0., 0., 1.);
		tgl = (Mat_<double>(3, 1) << 0., 0., 0.);
		Rgr = (Mat_<double>(3, 3) << 1., 0., 0., 0., 1., 0., 0., 0., 1.);
		tgr = (Mat_<double>(3, 1) << 0., 0., 0.);
		//Local transformations
		Rll = (Mat_<double>(3, 3) << 1., 0., 0., 0., 1., 0., 0., 0., 1.);
		tll = (Mat_<double>(3, 1) << 0., 0., 0.);
		Rlr = (Mat_<double>(3, 3) << 1., 0., 0., 0., 1., 0., 0., 0., 1.);
		tlr = (Mat_<double>(3, 1) << 0., 0., 0.);

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
        reprojectImageTo3D(curDisp, curPointCloud, Q, true);
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
	void binDetect(Mat &img,int i,int j,vector<KeyPoint> &dkpt){
		int bwc=10;//bin with col
		int bhr=5;//bin height row
		int brows=img.rows/bhr;
		int bcols=img.cols/bwc;
		Rect r(j, i, bcols, brows);
		Mat imgBin(img, r);
		detector->detect(imgBin, dkpt);
		//relocate
		for(unsigned int k=0;k<dkpt.size();k++){
			dkpt[k].pt+=Point2f(j,i);
		}
	}
	void binnedDetection(Mat &img,vector<KeyPoint> &kpts){
		int bwc=10;//bin with col
		int bhr=5;//bin height row
		int brows=img.rows/bhr;
		int bcols=img.cols/bwc;
		kpts.clear();
		vector<KeyPoint> dkpt;
		Mat ims;
		for(int i=0;i<img.rows-brows;i+=brows/2)
			for(int j=0;j<img.cols-bcols;j+=bcols/2){
				binDetect(img,i,j,dkpt);
				/*
				Rect r(j, i, bcols, brows);
				Mat imgBin(img, r);
				detector->detect(imgBin, dkpt);
				//relocate
				for(unsigned int k=0;k<dkpt.size();k++){
					dkpt[k].pt+=Point2f(j,i);
				}
				*/
				//cv::drawKeypoints(img,dkpt,ims, Scalar(0,0,255));
				//imshow("imgBin",imgBin);
				//imshow("dkpt",ims);
				//waitKey(0);
				//cout << "#dkpt="<<dkpt.size()<<endl;
			    kpts.insert(kpts.end(), dkpt.begin(), dkpt.end());//append dkpt to kpts
				/*int n=5;
				if(dkpt.size()>5){
					for(int i=0;i<n;i++){
					kpts.push_back(dkpt.back());
					dkpt.pop_back();
					}
				}
				else{
				    kpts.insert(kpts.end(), dkpt.begin(), dkpt.end());//append dkpt to kpts
				}*/
			}
		//cv::drawKeypoints(img,kpts,ims, Scalar(0,0,255));
		//imshow("kpts",ims);
		//waitKey(0);
	}
	bool is2DpointInFrame(Point2f& p){
		return p.x>=0 && p.x<curFrameL.cols-1 && p.y>=0 && p.y<curFrameL.rows-1;
	}
	void stepStereoOdometry(Mat& pcurFrameL_c,Mat& pcurFrameR_c){
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

		//Compute disparity and 3D point cloud
		sm->compute(curFrameL, curFrameR, curDisp);
        reprojectImageTo3D(curDisp, curPointCloud, Q, true);

		//Mix disparity and previous left frame
        //dispC8 if to be visualized
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
		Mat cpImg=stackV(cdispC8,dispC8);

	    //Lucas-Kanade parameters
	    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
	    //Size subPixWinSize(10,10), winSize(31,31);
	    Size subPixWinSize(10,10), winSize(20,20);
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
		cout << "#prevPoints" << prevPointsL.size()<<endl;
		cout << "#curPointsL" <<  curPointsL.size()<<endl;
		if(curPointsL.size()<250){//match point if there are few points
	        //Keypoints detection
			binnedDetection( curFrameL,  curKeypointsL);
			binnedDetection(prevFrameL, prevKeypointsL);
			//detector->detect(curFrameR, curKeypointsR);
			extractor->compute( curFrameL,  curKeypointsL,  curDescriptorsL);
			extractor->compute(prevFrameL, prevKeypointsL, prevDescriptorsL);
			//extractor->compute(curFrameR, curKeypointsR, curDescriptorsR);
			//prev to cur left matches
			findGoodMatches(prevKeypointsL,prevDescriptorsL,
					        curKeypointsL, curDescriptorsL,
							goodMatchesL,
							prevGoodKeypointsL,curGoodKeypointsL);
			cout << "#prevKeypointsL="<<prevKeypointsL.size()<<endl;
			cout << "#curKeypointsL="<<curKeypointsL.size()<<endl;
			cout << "#prevGoodKeypointsL="<<prevGoodKeypointsL.size()<<endl;
			cout << "#curGoodKeypointsL="<<curGoodKeypointsL.size()<<endl;

			curPointsL.clear();
			KeyPoint::convert(curGoodKeypointsL, curPointsL);
			prevPointsL.clear();
			KeyPoint::convert(prevGoodKeypointsL, prevPointsL);
			cout << "***** Tracking ******"<<endl;
			cout << "#prevPoints" << prevPointsL.size()<<endl;
			cout << "#curPointsL" <<  curPointsL.size()<<endl;
		}
		//Photometry subpixel improvement
        //cornerSubPix(prevFrameL, prevPointsL, subPixWinSize, Size(-1,-1), termcrit);
        //cornerSubPix( curFrameL,  curPointsL, subPixWinSize, Size(-1,-1), termcrit);

		int cbad=0,nbad=0,dbad=0;
        prev3Dpts.clear();
        cur3Dpts.clear();
        vector<Point2f> cur2Dpts,prev2Dpts;
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
        	if(pz!=10000 && cz!=10000){
				Point3f dif3D=p3d-c3d;
				float d=sqrt(dif3D.dot(dif3D));
				//d should be velocity/time
				if (d<50e3/36000){
					cout << "dif3D="<<dif3D<<":"<<d<<endl;
					int disp=prevDisp.at<unsigned short>(p2df1)>>4;//16-bit fixed-point disparity map (where each disparity value has 4 fractional bits)
					Point2f p2f2=prevPointsL[i];
					p2f2.x+=disp;
					p2f2.y+=cpImg.rows/2;
					line(cpImg,ppx,p2f2,Scalar(0, 255, 0));
					if(pz<40.5 && pz>0.0){
						putText(cpImg,to_string(pz)+":"+to_string(d),ppx,1,1,Scalar(0, 255, 255));
						putText(cpImg,to_string(cz),c2df1,1,1,Scalar(255, 255, 0));
						prev3Dpts.push_back(p3d);
						 cur3Dpts.push_back(c3d);
						prev2Dpts.push_back(p2df1);
						 cur2Dpts.push_back(c2df1);
						//circle(cpImg,ppx ,10,Scalar(255, 0, 0));
						//circle(cpImg,c2df1,10,Scalar(255, 0, 0));
						line(cpImg,ppx,c2df1,Scalar(255,255,255));
					}
					else{
						putText(cpImg,to_string(cz),c2df1,1,1,Scalar(0, 64, 255));
						//circle(cpImg,ppx,3,Scalar(0, 128, 255));
						line(cpImg,ppx,c2df1,Scalar(0, 128, 255));
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
        cout <<"#prevPointsL"<<prevPointsL.size() << endl;
        cout <<"#cbad distance ="<<cbad<<endl;
        cout <<"#nbad too far  ="<<nbad<<endl;
        cout <<"#dbad disparity="<<dbad<<endl;
        cout <<"#PointsL left  ="<<prevPointsL.size()-cbad-nbad-dbad<<endl;
        cout <<"prev3Dpts="<< prev3Dpts.size() << endl;
        cout <<"cur2DforPnP="<< cur2Dpts.size() << endl;

		Mat rvec;//rotation vector
		vector<int> inliers;
		vector<Point2f> proj2DafterPnP;
		vector<Point3f> prevAfterFitting;
    	Mat  cur3DMat=Mat( cur3Dpts).reshape(1);
    	Mat prev3DMat=Mat(prev3Dpts).reshape(1);
        rigidTransformation(cur3DMat,prev3DMat,Rll,tll);
        Rodrigues(Rll,rvec);

		cv::projectPoints(cur3Dpts,rvec,tll,K,Mat(),proj2DafterPnP);
		//cv::transform(cur3Dpts,prevAfterFitting,Rll);
		float tdp=0;
		vector<Point3f> cur3DptsClean,prev3DptsClean;
		vector<Point2f> cur2DptsClean,prev2DptsClean;
		Point2f dif2D;
		//Point3f dif3D;
		for(unsigned int i=0;i<proj2DafterPnP.size();i++){
        	dif2D=prev2Dpts[i]-proj2DafterPnP[i];
        	//dif3D=prev3Dpts[i]-prevAfterFitting[i];
        	float dp=sqrt(dif2D.dot(dif2D));
        	//float dp3D=sqrt(dif3D.dot(dif3D));
        	tdp+=dp;
        	if(dp>5.0){
        		//cout <<prev2Dpts[i]<<":"<<dp<<"~" <<proj2DafterPnP[i] << endl;
        	}
        	else{
        		 cur3DptsClean.push_back( cur3Dpts[i]);
        		prev3DptsClean.push_back(prev3Dpts[i]);
        		 cur2DptsClean.push_back( cur2Dpts[i]);
        		prev2DptsClean.push_back(prev2Dpts[i]);
        	}
		}
		for(size_t i=0;i<proj2DafterPnP.size();i++){
    		Point2f p2daf1=proj2DafterPnP[i];
    		Point2f p2df1=prev2Dpts[i];
    		Point2f ppxa(p2daf1.x,p2df1.y+cpImg.rows/2);//pixel of prev in cpImg
    		Point2f ppx(p2df1.x,p2df1.y+cpImg.rows/2);//pixel of prev in cpImg
			line(cpImg,ppx,ppxa,Scalar(0, 0, 255));

		}
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
        //cv::transform(cur3DMat,prev3DMatAfter,cur3DMat);
		//erase x and z rotations since car only rotate on y
        //rvec.at<float>(0.0)=0;
        //rvec.at<float>(2.0)=0;

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
  		resize(cpImg, cpImg, Size(), 0.70,0.70);
        imshow("prevDisp",cpImg);
	}
};

#endif /* DUALLKESTEREOVISUALODOMETRY_H_ */
