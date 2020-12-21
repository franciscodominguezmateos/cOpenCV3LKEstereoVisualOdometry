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
#include "graph.h"
#include "stereo_camera.h"

using namespace std;
using namespace cv;

class DualLKEstereoVisualOdometry {
public :
	StereoCamera sc;
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

	//global rotation and translation
	Mat Rgl, tgl, Rglprev,tglprev;
	Mat Rgr, tgr, Rgrprev,tgrprev;
	Mat Rg,  tg , Rgprev ,tgprev;
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
	//Mat Q;
	//double baseLine;
	//temp attributes
	vector<vector<DMatch> > matches;
	//Mat E,mask;
	vector<Point3f> prev3Dpts,cur3Dpts;
	//Historic data
	vector< vector<Point2f> > pts2D;
	vector< vector<Point3f> > pts3D;

	//Visualization objects
	Mat cpImg;
    viz::Viz3d myWindow;
//public:
	DualLKEstereoVisualOdometry():bd(3,10,10),myWindow("Coordinate Frame"){	}
	DualLKEstereoVisualOdometry(double f,Point2f pp,double baseLine):sc(f,pp,baseLine),bd(3,10,10),myWindow("Coordinate Frame"){}
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
			Mat I  = (Mat_<double>(3, 3) << 1., 0., 0.,
					                     0., 1., 0.,
										 0., 0., 1.);
			Mat t0  = (Mat_<double>(3, 1) << 0., 0., 0.);
			Rg  = I;
			tg  = t0;
			Rgl = I;
			tgl = t0;
			Rgr = I;
			tgr = t0;
			//Local transformations
			Rll = I;
			tll = t0;
			Rlr = I;
			tlr = t0;

			// Stereo 3D data extraction
			sm=StereoBM::create(16*8,5);//9);
			sm->compute(curFrameL,curFrameR,curDisp);
	        reprojectImageTo3D(curDisp, curPointCloud, sc.Q, true);
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
		for(vector<DMatch> &vdm:matches){
			if(vdm.size()>1){
				if(vdm[0].distance/vdm[1].distance>0.6) vdm.clear();
			}
			else vdm.clear();
			if(!vdm.empty()) good_matches.push_back(vdm[0]);
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
	/*inline void getPoseFromEssentialMat(vector<Point2f> &point1, vector<Point2f> &point2,Mat &R,Mat &t){
		E = findEssentialMat(point2, point1, f, pp, RANSAC, 0.999, 1.0,mask);
		recoverPose(E, point2, point1, R, t, f, pp,mask);
	}*/
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
	//
	void selectPointsFromIndices(vector<int> indices,Scalar color=Scalar(0,255,0),int thick=2){
		//Clean points
		vector<Point3f> cur3DptsClean,prev3DptsClean;
		vector<Point2f> cur2DptsClean,prev2DptsClean;
		for(int &i:indices){
			 cur3DptsClean.push_back( cur3Dpts[i]);
			prev3DptsClean.push_back(prev3Dpts[i]);
			 cur2DptsClean.push_back( cur2Dpts[i]);
			prev2DptsClean.push_back(prev2Dpts[i]);
    		Point2f ppx(prev2Dpts[i].x,prev2Dpts[i].y+cpImg.rows/2);//pixel of prev in cpImg
			line(cpImg,ppx,cur2Dpts[i],color,thick);
		}
 		 cur3Dpts= cur3DptsClean;
 		prev3Dpts=prev3DptsClean;
 		 cur2Dpts= cur2DptsClean;
 		prev2Dpts=prev2DptsClean;
	}
	void buildGoodPoints(){
        prev3Dpts.clear();
         cur3Dpts.clear();
        prev2Dpts.clear();
         cur2Dpts.clear();
		int cbad=0,nbad=0,dbad=0;
        for(unsigned int i=0;i<prevPointsL.size();i++){
    		Point2f p2df1=prevPointsL[i];
    		Point2f c2df1= curPointsL[i];

    		//theses two points should be very close to each other
        	Point3f p3d=prevPointCloud.at<Point3f>(p2df1);
        	Point3f c3d= curPointCloud.at<Point3f>(c2df1);
        	float pz=p3d.z;
        	float cz=c3d.z;
        	//not disparity points have a z value of 1000
        	if(pz<10000 && cz<10000){
				float d=distPoint3f(p3d,c3d);
				//d should be velocity/time
				if (d<500e3/36000){
					if(pz<50 && pz>0.0){
						prev3Dpts.push_back(p3d);
						 cur3Dpts.push_back(c3d);
						prev2Dpts.push_back(p2df1);
						 cur2Dpts.push_back(c2df1);
					}
					else nbad++;
				}
				else cbad++;
        	}
        	else dbad++;
        }
        cout <<"#prevPointsL   ="<<prevPointsL.size() << endl;
        cout <<"#cbad distance ="<<cbad<<endl;
        cout <<"#nbad too far  ="<<nbad<<endl;
        cout <<"#dbad disparity="<<dbad<<endl;
        cout <<"#PointsL left  ="<<prevPointsL.size()-cbad-nbad-dbad<<endl;
	}
	Mat allDistances(vector<Point3f> &pts3D){
		int L=fmin(1000,pts3D.size());
		Mat r=Mat::zeros(L,L,CV_32F);
		for(int i=0;i<L;i++){
			for(int j=0;j<L;j++){
				float d=distPoint3f(pts3D[i],pts3D[j]);
				r.at<float>(i,j)=d;
			}
		}
		return r;
	}
	void rigidBodyConstraint(){
		Mat prevDist=allDistances(prev3Dpts);
		Mat currDist=allDistances(cur3Dpts);
		Mat dist;
		absdiff(prevDist,currDist,dist);
		//this doesn't work!!!
		//Mat r=(dist>0.1f);
		Mat r(dist.size(),dist.type());
		for(int i=0;i<dist.rows;i++){
			for(int j=0;j<dist.cols;j++){
				if(dist.at<float>(i,j)>0.20)
					r.at<float>(i,j)=0;
				else
					r.at<float>(i,j)=1;
			}
		}
		//printMat(r);
		Graph<float> g(r);
		vector<int> mClique=g.maxClique();
		cout << "maxClique="<<mClique.size()<<endl;
		if(mClique.size()<10) return;
		selectPointsFromIndices(mClique);
	}
	float msr2D(vector<Point2f> &vp0,vector<Point2f> &vp1){
		if(vp0.size()!=vp1.size())
			throw runtime_error("Both vectors have to have same size in msr2D");
		float tdp=0;
		for(unsigned int i=0;i<vp0.size();i++){
			float dp=distPoint2f(vp0[i],vp1[i]);
			tdp+=dp;
		}
		float msr=tdp/vp0.size();
		return msr;
	}
	vector<int> msr2Didx(vector<Point2f> &vp0,vector<Point2f> &vp1,float threshold=1.0){
		if(vp0.size()!=vp1.size())
			throw runtime_error("Both vectors have to have same size in msr2Didx");
		vector<int> vi;
		for(unsigned int i=0;i<vp0.size();i++){
			float dp=distPoint2f(vp0[i],vp1[i]);
			if(dp>threshold) vi.push_back(i);
		}
		return vi;
	}
	void projectionError(Mat &rvec,Mat &tll){
		vector<Point2f> prev2Dproj;
		projectPoints(cur3Dpts,rvec,tll,sc.K,Mat(),prev2Dproj);
		vector<int> indices=msr2Didx(prev2Dpts,prev2Dproj);
		//cv::transform(cur3Dpts,prevAfterFitting,Rll);
		float tdp=0;
		vector<Point3f> cur3DptsClean,prev3DptsClean;
		vector<Point2f> cur2DptsClean,prev2DptsClean;
		Point2f dif2D;
		//Point3f dif3D;
		for(unsigned int i=0;i<prev2Dproj.size();i++){
			//Visualization variables
    		Point2f p2daf1=prev2Dproj[i];
    		Point2f p2df1 =prev2Dpts[i];
    		Point2f ppxa(p2daf1.x,p2df1.y+cpImg.rows/2);//pixel of prev in cpImg
    		Point2f ppx(p2df1.x,p2df1.y+cpImg.rows/2);//pixel of prev in cpImg
    		//work out varibles
        	//dif3D=prev3Dpts[i]-prevAfterFitting[i];
        	float dp=distPoint2f(prev2Dpts[i],prev2Dproj[i]);
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
		tdp/=prev2Dproj.size();
		cout << "tdp0="<< tdp << endl;

		if(cur3DptsClean.size()>0){
			Mat cur3DMat1=Mat( cur3DptsClean).reshape(1);
			//cout << " cur3DMat1=" << cur3DptsClean.size()<< endl;
			Mat prev3DMat1=Mat(prev3DptsClean).reshape(1);
			//cout << "prev3DMat1=" << prev3DptsClean.size() <<endl;
			rigidTransformation(cur3DMat1,prev3DMat1,Rll,tll);
			Rodrigues(Rll,rvec);
	        //Mat inliers;
	        //cv::solvePnPRansac(cur3DptsClean,prev3DptsClean,sc.K,noArray(),rvec,tll,true,100,1.0,0.99,inliers);
			//cout << "rigidTransformation=" << endl;
			prev2Dproj.clear();
			cv::projectPoints(cur3DptsClean,rvec,tll,sc.K,Mat(),prev2Dproj);
			//cout << "projectPoints=" << endl;
			tdp=msr2D(prev2DptsClean,prev2Dproj);
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
        viz::WCameraPosition *cpw_frustum=new viz::WCameraPosition(Matx33d(sc.K),curFrameL_c,0.91); // Camera frustum
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
		curDisp      .copyTo(prevDisp);
		curPointCloud.copyTo(prevPointCloud);
		curFrameL    .copyTo(prevFrameL);		      curFrameR.copyTo(prevFrameR);
		curFrameL_c  .copyTo(prevFrameL_c);	        curFrameR_c.copyTo(prevFrameR_c);
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
        cv::solvePnPRansac(cur3Dpts,prev2Dpts,sc.K,noArray(),rr,tr,false,100,1.5,0.99,inliers);
        vector<int> indices;
        for(int i=0;i<inliers.rows;i++)
        	indices.push_back(inliers.at<int>(i,0));
        selectPointsFromIndices(indices);
        /*
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
        */
        cv::solvePnPRansac(cur3Dpts,prev2Dpts,sc.K,noArray(),rr,tr,true,100,1.0,0.99,inliers);
        for(int i=0;i<inliers.rows;i++)
        	indices.push_back(inliers.at<int>(i,0));
        selectPointsFromIndices(indices);
        /*prev3DptsIn.clear();
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
        prev2Dpts=prev2DptsIn;*/
	}
	void buildPointCould(){
		//Compute disparity and 3D point cloud
		sm->compute(curFrameL, curFrameR, curDisp);
        reprojectImageTo3D(curDisp, curPointCloud, sc.Q, true);
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
        //rvec.at<float>(0.0)=0;
        //rvec.at<float>(2.0)=0;

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
		//tll.at<float>(0,0)=0;
		//tll.at<float>(1,0)=0;
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
		if(curPointsL.size()<90){//match point if there are few points
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

		buildGoodPoints();

		rigidBodyConstraint();

		Mat rr,tr;
		buildRelativePose(rr,tr);

        updatePose(rr,tr);

		refreshTrackingPoints();

		//Visualisation 2D
  		resize(cpImg, cpImg, Size(), 0.95,0.95);
        imshow("prevDisp",cpImg);

        //Visualization 3D
        myWindow.spinOnce(1, true);

	}
};

#endif /* DUALLKESTEREOVISUALODOMETRY_H_ */
