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
#include "binned_detector.h"
#include "opencv3_util.h"

using namespace std;
using namespace cv;

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
//public:
	DualLKEstereoVisualOdometry(Mat &pcurFrameL_c,Mat &pcurFrameR_c):bd(3,10){
		cout << pcurFrameL_c.rows << ","<<pcurFrameL_c.cols<<endl;
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
		bd.binnedDetection(curFrameL, curKeypointsL);
		curPointsL.clear();
		KeyPoint::convert(curKeypointsL, curPointsL);
        //cornerSubPix(curFrameL, curPointsL, subPixWinSize, Size(-1,-1), termcrit);
		//detector->detect(curFrameR, curKeypointsR);
		bd.binnedDetection(curFrameR, curKeypointsR);
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
	bool is2DpointInFrame(Point2f& p){
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
				if (d<50e3/36000){
					//cout << "dif3D="<<dif3D<<":"<<d<<endl;
					int disp=prevDisp.at<unsigned short>(p2df1)>>4;//16-bit fixed-point disparity map (where each disparity value has 4 fractional bits)
					Point2f p2f2=prevPointsL[i];
					p2f2.x+=disp;
					p2f2.y+=cpImg.rows/2;
					line(cpImg,ppx,p2f2,Scalar(0, 255, 0));
					if(pz<40.5 && pz>0.0){
						putText(cpImg,toString(pz)+":"+toString(d),ppx,1,1,Scalar(0, 255, 255));
						putText(cpImg,toString(cz),c2df1,1,1,Scalar(255, 255, 0));
						prev3Dpts.push_back(p3d);
						 cur3Dpts.push_back(c3d);
						prev2Dpts.push_back(p2df1);
						 cur2Dpts.push_back(c2df1);
						//circle(cpImg,ppx ,10,Scalar(255, 0, 0));
						//circle(cpImg,c2df1,10,Scalar(255, 0, 0));
						line(cpImg,ppx,c2df1,Scalar(255,255,255));
					}
					else{
						putText(cpImg,toString(cz),c2df1,1,1,Scalar(0, 64, 255));
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

 		cur3Dpts= cur3DptsClean;
 		prev3Dpts= prev3DptsClean;
 		curPointsL=cur2DptsClean;
 		prevPointsL=prev2DptsClean;

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
		cpImg=stackV(cdispC8,dispC8);

		opticalFlowPyrLKTrack();

		selectGoodPoints();

		Mat rvec;//rotation vector
		vector<Point3f> prevAfterFitting;
    	//Mat  cur3DMat=Mat( cur3Dpts).reshape(1);
    	//Mat prev3DMat=Mat(prev3Dpts).reshape(1);
        //rigidTransformation(cur3DMat,prev3DMat,Rll,tll);
        //Rodrigues(Rll,rvec);

        Mat r,t,rr,tr,rri,tri;
        //cv::solvePnP(cur3Dpts,prev2Dpts,K,noArray(),r,t,0,SOLVEPNP_ITERATIVE);

        Mat inliers;
        cv::solvePnPRansac(cur3Dpts,prev2Dpts,K,noArray(),rr,tr,false,1000,1.0,0.99,inliers);
        vector<Point3f> prev3DptsIn,cur3DptsIn;
        vector<Point2f> prev2DptsIn, cur2DptsIn;
        for(int i=0;i<inliers.rows;i++){
        	int idx=inliers.at<int>(i,0);
        	 cur3DptsIn.push_back( cur3Dpts[idx]);
        	prev3DptsIn.push_back(prev3Dpts[idx]);
        	prev2DptsIn.push_back(prev2Dpts[idx]);
        	 cur2DptsIn.push_back( cur2Dpts[idx]);
        }
        rri=rr;
        tri=tr;
        cv::solvePnPRansac(cur3DptsIn,prev2DptsIn,K,noArray(),rri,tri,true,1000,0.5,0.99,inliers);
        rvec=rri; tll=tri;
        cv::Rodrigues(rvec,Rll);

        // curPointsL=cur2DptsIn;
        //prevPointsL=prev2DptsIn;
        projectionError(rvec,tll);

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

        //refresh traking points
		cout << "#curPoints" <<  curPointsL.size()<<endl;
		if(curPointsL.size()<1950){//match point if there are few points
	        //Keypoints detection
			curKeypointsL.clear();
			bd.refreshDetection(curFrameL, curPointsL, curKeypointsL);
			vector<Point2f> rPoints;
			KeyPoint::convert(curKeypointsL, rPoints);
		    curPointsL.insert(curPointsL.end(), rPoints.begin(), rPoints.end());
		}
		cout << "#curPoints" <<  curPointsL.size()<<endl;

		//Visualization
  		resize(cpImg, cpImg, Size(), 0.970,0.970);
        imshow("prevDisp",cpImg);
	}
};

#endif /* DUALLKESTEREOVISUALODOMETRY_H_ */
