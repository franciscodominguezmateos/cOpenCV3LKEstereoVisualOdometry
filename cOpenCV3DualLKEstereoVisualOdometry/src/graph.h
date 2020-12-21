/*
 * graph.h
 *
 *  Created on: 19 Dec 2020
 *      Author: Francisco Dominguez
 *  This graph if a help to detect inlier with maximoum cliques
 *  of connected keypoints
 */

#pragma once
//this default value doesn't seem to work
template<typename T=float>
class Graph{
	//Adjacency matrix
	Mat G;
public:
	Graph(Mat g):G(g){}
	T &at(int i,int j){return G.at<T>(i,j);}
	int getMaxConnectedNodeIdx(){
		Mat v;
		double minv,maxv;
		int minIdx[2],maxIdx[2];
		reduce(G,v,0,CV_REDUCE_SUM);
		minMaxIdx(v,&minv,&maxv,minIdx,maxIdx);
		cout << minv<<","<<maxv<<endl;
		cout <<minIdx[0]<<","<<minIdx[1]<<endl;
		cout <<maxIdx[0]<<","<<maxIdx[1]<<endl;
		cout <<v.at<T>(maxIdx[0],maxIdx[1])<<endl;
		return maxIdx[1];
	}
	void updateClique(vector<int> &potentialNodes,vector<int> &clique){
		int maxNumMatches=0;
		int curr_max=0;
		for(int &i:potentialNodes){
				int numMatches=0;
				for(int &j:potentialNodes){
					if(this->at(i,j)==1)
						numMatches+=1;
				}
				if(numMatches>=maxNumMatches){
					curr_max=i;
					maxNumMatches=numMatches;
				}
		}
		if(maxNumMatches!=0){
			clique.push_back(curr_max);
		}
	}
	vector<int> findPotentialNodes(vector<int> &clique){
		Mat newSet=G.col(clique[0]);
		if(clique.size()>1){
			for(int i=1;i<clique.size();i++){
				Mat c=G.col(clique[i]);
				newSet = newSet.mul(c);
			}
		}
		//nodes in clique are not potentialNodes
		for(int i=0;i<clique.size();i++){
			newSet.at<T>(clique[i],0)=0;
		}
		vector<int> vpn;
		for(int i=0;i<newSet.rows;i++)
			if(newSet.at<T>(i,0)==1)
				vpn.push_back(i);
		return vpn;
	}
	vector<int> maxClique(){
		vector<int> clique;
		int idx=this->getMaxConnectedNodeIdx();
		clique.push_back(idx);
		vector<int> potentialNodes=this->findPotentialNodes(clique);
		while(potentialNodes.size()!=0){
			this->updateClique(potentialNodes,clique);
			potentialNodes=this->findPotentialNodes(clique);
		}
		return clique;
	}
};




