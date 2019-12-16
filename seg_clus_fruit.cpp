#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;

Mat dst;
Mat src;
Mat thresholded;
Mat hue;

/** @function main */
int main( int argc, char** argv )
{

vector<Mat> hsv(3);
vector<Mat> bgr(3);
Mat detected_edges;


		if (argc>1){
  /// Load an image
  src = imread( argv[1] );

}else{ src = imread("./image-seg/fruit.jpg");
	
}
  if( !src.data )
  { return -1; }

cout	<<src.cols<<endl;
cout	<<src.rows<<endl;
	int limiar=128;
	
	/// Create hsv
	Mat full_hsv;
	cvtColor(src,full_hsv,CV_BGR2HSV);
	split(full_hsv, hsv);
	split(src, bgr);
	//namedWindow("h", WINDOW_NORMAL);
	//imshow("h",hsv[0]);
	//waitKey(0);
	
	//namedWindow("s", WINDOW_NORMAL);
	//imshow("s",hsv[1]);
	//waitKey(0);

	int num_classes=9;

	vector<Rect> interest(num_classes);
	interest[0]=Rect(100,50,2300,200);//fundo
	interest[1]=Rect(2200,1750,200,200);//chao
	interest[2]=Rect(1600,550,80,450);//fruta vermelha
	interest[3]=Rect(720,600,80,300);//fruta verde
	interest[4]=Rect(1750,420,350,300);//frutas amareladas
	interest[5]=Rect(800,1300,300,300);//cesta e madeira
	interest[6]=Rect(1320,1530,300,100);//alho
	interest[7]=Rect(1220,1850,250,80);//pano
	interest[8]=Rect(600,1730,250,50);//saca rolhas
	vector<Scalar> color(num_classes);
	color[0]=Scalar(0,0,0);// fundo é preto
	color[1]=Scalar(128,128,128);// chao cinza medio
	color[2]=Scalar(0,0,255);// 
	color[3]=Scalar(0,200,0);// 
	color[4]=Scalar(0,255,255);// 
	color[5]=Scalar(106,106,220);// 
	color[6]=Scalar(255,255,255);//alho
	color[7]=Scalar(255,0,0);//pano
	color[8]=Scalar(200,200,200);//saca rolhas
	
	Mat show;
	src.copyTo(show);
	
	for (int i=0;i<num_classes;i++){
		rectangle(show,Point(interest[i].x,interest[i].y),Point(interest[i].x+interest[i].width,interest[i].y+interest[i].height),color[i],8);
	}
	
	namedWindow("interests", WINDOW_NORMAL);
	imshow("interests",show);
	imwrite("./clustering_chosen/interest_points_fruit.png",show);
	//BGR
	vector<Scalar> centroideBGR(num_classes);
	vector<Scalar> centroideBGR_novo(num_classes);
	cout<<"Centroides BGR:"<<endl;
	for (int i=0;i<num_classes;i++){
		Mat roi(src,interest[i]);
		centroideBGR[i]=mean(roi);	
		cout<<centroideBGR[i]<<endl;	
		
	}
	
	
	float distancia_aceitavel=sqrt(3.0);
	vector<int> num_in_class(num_classes);
	
	
	hsv[0].copyTo(hue);
	vector<Mat> segmentedBGR(3);
	vector<Mat> segmentedHSV(3);
	segmentedBGR[0].create( hue.size(), hue.type() );
	segmentedBGR[1].create( hue.size(), hue.type() );
	segmentedBGR[2].create( hue.size(), hue.type() );
	
	 
	bool repeat=true;
	int iteracao=1;
	while (repeat){
		for (int i=0;i<num_classes;i++){
			centroideBGR_novo[i]=Scalar(0,0,0,0);	
			num_in_class[i]=0;
		}
		for(int i=0; i<src.rows;i++){
			uchar* rowiB = segmentedBGR[0].ptr<uchar>(i);
			uchar* rowiG = segmentedBGR[1].ptr<uchar>(i);
			uchar* rowiR = segmentedBGR[2].ptr<uchar>(i);
			
			for(int j=0; j<src.cols; j++){
				Scalar bgrValue = Scalar(bgr[0].at<uchar>(Point(j, i)),bgr[1].at<uchar>(Point(j, i)),bgr[2].at<uchar>(Point(j, i)),0);
				float min_dist=99999999;
				int classe=-1;
				for (int k=0;k<num_classes;k++){
					Scalar diff= centroideBGR[k]-bgrValue;					
					if (norm(diff)<min_dist){
						min_dist=norm(diff);
						classe=k;
					}
				}
				centroideBGR_novo[classe]+=bgrValue;
				num_in_class[classe]++;
				rowiB[j]=color[classe][0];
				rowiG[j]=color[classe][1];
				rowiR[j]=color[classe][2];
			}
		}
		//recalcular centroides
		repeat=false;
		for (int i=0;i<num_classes;i++){
			centroideBGR_novo[i]=Scalar(1.0*centroideBGR_novo[i](0)/num_in_class[i],1.0*centroideBGR_novo[i](1)/num_in_class[i],1.0*centroideBGR_novo[i](2)/num_in_class[i],0);
			if(norm(centroideBGR_novo[i]-centroideBGR[i])>distancia_aceitavel){
				repeat=true;	
				cout<<"centroide da classe "<<i<<" alterado de "<<centroideBGR[i]<<" para "<<centroideBGR_novo[i]<<endl;
				centroideBGR[i]=centroideBGR_novo[i];				
			}			
		}
		
		
	
	Mat colorSegBGR;
	
	merge(segmentedBGR,colorSegBGR);
	namedWindow("bgr seg", WINDOW_NORMAL);
	imshow("bgr seg",colorSegBGR);
	waitKey(100);
	std::string s;
	std::stringstream out;
	out << iteracao;
	s = out.str();
	imwrite("./clustering_chosen/fruit_BGR_step_"+s+".png",colorSegBGR);
	
	cout<<"BGR segmentado, iteracao="<<iteracao<<endl<<endl<<endl;	
	iteracao++;
	}
	
	
	
	


//HSV
vector<Scalar> centroideHSV(num_classes);
vector<Scalar> centroideHSV_novo(num_classes);
	cout<<"Centroides HSV:"<<endl;
	for (int i=0;i<num_classes;i++){
		Mat roiHSV(full_hsv,interest[i]);
		centroideHSV[i]=mean(roiHSV);
		cout<<centroideHSV[i]<<endl;
	}


	segmentedHSV[0].create( hue.size(), hue.type() );
	segmentedHSV[1].create( hue.size(), hue.type() );
	segmentedHSV[2].create( hue.size(), hue.type() );
	
	
	repeat=true;
	iteracao=1;
	while (repeat){
		for (int i=0;i<num_classes;i++){
			centroideHSV_novo[i]=Scalar(0,0,0,0);	
			num_in_class[i]=0;
		}
	
	for(int i=0; i<src.rows;i++){
		uchar* rowiB = segmentedHSV[0].ptr<uchar>(i);
		uchar* rowiG = segmentedHSV[1].ptr<uchar>(i);
		uchar* rowiR = segmentedHSV[2].ptr<uchar>(i);
		for(int j=0; j<src.cols; j++){
			float min_dist=99999999;
			int classe=-1;
			Scalar bgrValue = Scalar(hsv[0].at<uchar>(Point(j, i)),hsv[1].at<uchar>(Point(j, i)),hsv[2].at<uchar>(Point(j, i)),0);
			for (int k=0;k<num_classes;k++){
				Scalar diff= centroideHSV[k]-bgrValue;
				
				if (norm(diff)<min_dist){
					min_dist=norm(diff);
					classe=k;
				}
			}
			centroideHSV_novo[classe]+=bgrValue;
			num_in_class[classe]++;
			rowiB[j]=color[classe][0];
			rowiG[j]=color[classe][1];
			rowiR[j]=color[classe][2];
		}
	}
	
	
		//recalcular centroides
		repeat=false;
		for (int i=0;i<num_classes;i++){
			centroideHSV_novo[i]=Scalar(1.0*centroideHSV_novo[i](0)/num_in_class[i],1.0*centroideHSV_novo[i](1)/num_in_class[i],1.0*centroideHSV_novo[i](2)/num_in_class[i],0);
			if(norm(centroideHSV_novo[i]-centroideHSV[i])>distancia_aceitavel){
				repeat=true;	
				cout<<"centroide da classe "<<i<<" alterado de "<<centroideHSV[i]<<" para "<<centroideHSV_novo[i]<<endl;
				centroideHSV[i]=centroideHSV_novo[i];				
			}			
		}
		
		
	
	Mat colorSegHSV;
	
	merge(segmentedHSV,colorSegHSV);
	namedWindow("hsv seg", WINDOW_NORMAL);
	imshow("hsv seg",colorSegHSV);
	
	std::string s;
	std::stringstream out;
	out << iteracao;
	s = out.str();
	imwrite("./clustering_chosen/fruit_HSV_step_"+s+".png",colorSegHSV);
	waitKey(100);
	cout<<"HSV segmentado, iteracao="<<iteracao<<endl<<endl<<endl;	
	iteracao++;
	}
	
  }

