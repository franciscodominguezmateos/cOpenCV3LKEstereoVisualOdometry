
/*
The MIT License
Copyright (c) 2015 Satyaki Chakraborty

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <sstream>
#include "iostream"
#include "vector"

#include "DualLKEstereoVisualOdometry.h"

#define PI 3.14159265
#define minFeatures 35

using namespace std;
using namespace cv;

vector<Mat> getKittiPoses(string fileName){
	vector<Mat> vm;
	vector<string> lines=getLinesFromFile(fileName);
	for(unsigned int i=0;i<lines.size();i++){
		string line=lines[i];
		vector<string> words=split(line);
		double d[12];
		for(unsigned int i=0;i<words.size();i++){
			d[i]=atof(words[i].c_str());
		}
		Mat m = (Mat_<double>(3, 4) << d[0],d[1],d[2],d[3],
				                       d[4],d[5],d[6],d[7],
				                       d[8],d[9],d[10],d[11]);
		vm.push_back(m);
	}
	return vm;
}
string get_sequence(int n) {
	if (n==0) return "000000";
	else if (n/10 == 0) return "00000"+toString(n);
	else if (n/100 == 0) return "0000"+toString(n);
	else if (n/1000 == 0) return "000"+toString(n);
	else if (n/10000 == 0) return "00"+toString(n);
	return NULL;
}
int main(int argc, char** argv) {

	if (argc< 3) {
		cout << "Enter path to data.. ./odo <path> <numFiles>\n";
		return -1;
	}
	
	if (argv[1][strlen(argv[1])-1] == '/') {
		argv[1][strlen(argv[1])-1] = '\0';
	}
	string path = string(argv[1]);
	int SEQ_MAX = atoi(argv[2]);

	//int seq_id = 0, scene_id = 0;
	Mat top_view2 = Mat::zeros(400, 400, CV_8UC3);
	Mat top_view3 = Mat::zeros(400, 400, CV_8UC3);
	Mat cur_frame_c2,cur_frame_c3;
	Mat t2_,t3_;
	cur_frame_c2 = imread(path+"/image_2/000000.png");
	cur_frame_c3= imread(path+"/image_3/000000.png");
	vector<Mat> vm=getKittiPoses(path+"/../poses/00.txt");
	DualLKEstereoVisualOdometry dsvo2(cur_frame_c2,cur_frame_c3);

	float l=0;//trajectory length
	for (int i=1; i<=SEQ_MAX; i+=1) {
		t2_= (Mat_<double>(3, 1) << vm[i].at<double>(0,3),
				                    vm[i].at<double>(1,3),
									vm[i].at<double>(2,3));
		l+=abs(t2_.at<double>(2,0));
		//cout << "gpos:" << t2_.t() <<":"<<l<< endl;
    	circle(top_view2, Point(200+t2_.at<double>(0, 2)/2.0, (200+t2_.at<double>(0, 0)/2.0)), 3, Scalar(0, 255, 255), -1);
		circle(top_view2, Point(200+t2_.at<double>(0, 2)/2.0, (200+t2_.at<double>(0, 0)/2.0)), 2, Scalar(0,   0, 255), -1);
	}
	l=0;
	for (int i=1; i<=SEQ_MAX; i+=1) {
		string nf2=path+"/image_2/"+get_sequence(i)+".png";
		string nf3=path+"/image_3/"+get_sequence(i)+".png";
		cout << nf2 << endl;
		cur_frame_c2 = imread(nf2);
		cur_frame_c3 = imread(nf3);
		dsvo2.stepStereoOdometry(cur_frame_c2,cur_frame_c3);
		t2_=dsvo2.tgl;
		cout << "gpos:" << t2_.t() << endl;
		l+=abs(dsvo2.tll.at<float>(2,0));
		cout << " i================="<<i<<"lpos:" << dsvo2.tll.t() <<":"<<l<< endl;
		circle(top_view2, Point(200+t2_.at<double>(0, 2)/2.0, (200+t2_.at<double>(0, 0)/2.0)), 3, Scalar(0, 255, 0), -1);
		circle(top_view2, Point(200+t2_.at<double>(0, 2)/2.0, (200+t2_.at<double>(0, 0)/2.0)), 2, Scalar(0, 0, 255), -1);
  		imshow("Top view2", top_view2);
  		if(i%25==0)
  			dsvo2.showCloud(get_sequence(i),dsvo2.Rgl,dsvo2.tgl);
 		if (waitKey(1) == 27) break;
	}
	return 0;
}

