#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;


vector<vector<Rect> > regions;
Mat dst;
Mat src;
Mat thresholded;
Mat hue;
int level=0;
int region_count=0;
int level_max;
vector<Rect> segmentation(int limiar1, int limiar2, int x,int y,int w, int h, Scalar cor){
	vector<Rect> temp;	
	
	bool print=false;
	
	level++;
	if(w==0||h==0){
		level--;
		return temp;
	}
		
	float mult=1.0;//-((level-1)*0.05);
		
	Rect roi1=Rect(x,y,w/2,h/2);
	Rect roi2=Rect(x+w/2,y,w-w/2,h/2);
	Rect roi3=Rect(x,y+h/2,w/2,h-h/2);
	Rect roi4=Rect(x+w/2,y+h/2,w-w/2,h-h/2);
	if(h>300){
	Mat dst_copy;
		dst.copyTo(dst_copy);
		rectangle(dst_copy, roi1, Scalar(255));
		rectangle(dst_copy, roi2, Scalar(255));
		rectangle(dst_copy, roi3, Scalar(255));
		rectangle(dst_copy, roi4, Scalar(255));
		namedWindow("dst",WINDOW_NORMAL);
		imshow("dst",dst_copy);
		waitKey(1);
	}
	Mat region1(hue, roi1);
	Mat region2(hue, roi2);
	Mat region3(hue, roi3);
	Mat region4(hue, roi4);

	Mat thresh1;
	thresh1.create( region1.size(), region1.type() );
	Mat thresh2;
	thresh2.create( region2.size(), region2.type() );
	Mat thresh3;
	thresh3.create( region3.size(), region3.type() );
	Mat thresh4;
	thresh4.create( region4.size(), region4.type() );

	
	threshold( region1,thresh1,limiar1,255,0 );		
	Mat aux;
	threshold( region1,aux,limiar2,255,0 );			
	thresh1=thresh1-aux;
	if(sum(thresh1)[0]/255>=thresh1.cols*thresh1.rows*mult || sum(thresh1)[0]==0){//se ou tudo branco ou tudo preto
	
		Mat dst1(dst,roi1);
		if(sum(thresh1)[0]==0)
		;//dst1=Scalar(0);
		else{
		dst1.setTo(cor);
		temp.push_back(roi1);		
	}
	}else{//cout<<"segmentando regiao 1"<<endl;
		vector<Rect> connected_inside=segmentation(limiar1, limiar2,roi1.x,roi1.y,roi1.width,roi1.height,cor);
		temp.insert(temp.end(), connected_inside.begin(), connected_inside.end());
		//cout<<"regiao 1 segmentada"<<endl;
	}
	
	
	threshold( region2,thresh2,limiar1,255,0 );
	threshold( region2,aux,limiar2,255,0 );			
	thresh2=thresh2-aux;
	if(sum(thresh2)[0]/255>=thresh2.cols*thresh2.rows*mult || sum(thresh2)[0]==0){//se ou tudo branco ou tudo preto
		Mat dst2(dst,roi2);
		if(sum(thresh2)[0]==0)
		;//dst2=Scalar(0);
		else{
		dst2.setTo(cor);
		temp.push_back(roi2);
	}
	}else{
		//cout<<"segmentando regiao 2"<<endl;
		vector<Rect> connected_inside=segmentation(limiar1,limiar2,roi2.x,roi2.y,roi2.width,roi2.height,cor);
		temp.insert(temp.end(), connected_inside.begin(), connected_inside.end());
		//cout<<"regiao 2 segmentada"<<endl;
	}
	threshold( region3,thresh3,limiar1,255,0 );
	threshold( region3,aux,limiar2,255,0 );			
	thresh3=thresh3-aux;
	if(sum(thresh3)[0]/255>=thresh3.cols*thresh3.rows*mult || sum(thresh3)[0]==0){//se ou tudo branco ou tudo preto
		Mat dst3(dst,roi3);
		if(sum(thresh3)[0]==0)
		;//dst3=Scalar(0);
		else{
		dst3.setTo(cor);			
		temp.push_back(roi3);
	}
	}else{
		//cout<<"segmentando regiao 3"<<endl;
		vector<Rect> connected_inside=segmentation(limiar1,limiar2,roi3.x,roi3.y,roi3.width,roi3.height,cor);
		temp.insert(temp.end(), connected_inside.begin(), connected_inside.end());
		//cout<<"regiao 3 segmentada"<<endl;
	}
	
	threshold( region4,thresh4,limiar1,255,0 );
	threshold( region4,aux,limiar2,255,0 );			
	thresh4=thresh4-aux;
	if(sum(thresh4)[0]/255>=thresh4.cols*thresh4.rows*mult || sum(thresh4)[0]==0){//se ou tudo branco ou tudo preto
		Mat dst4(dst,roi4);
		if(sum(thresh4)[0]==0)
		;//dst4=Scalar(0);
		else{
		dst4.setTo(cor);			
		temp.push_back(roi4);
	}
	}else{
		//cout<<"segmentando regiao 4"<<endl;
		vector<Rect> connected_inside=segmentation(limiar1,limiar2,roi4.x,roi4.y,roi4.width,roi4.height,cor);
		temp.insert(temp.end(), connected_inside.begin(), connected_inside.end());
		//cout<<"regiao 4 segmentada"<<endl;
	}
			
		level--;
		//regions.push_back(temp);
		return temp;
	}















/** @function main */
int main( int argc, char** argv )
{

vector<Mat> hsv(3);
Mat detected_edges;


		if (argc>1){
  /// Load an image
  src = imread( argv[1] );

}else{ src = imread("./image-seg/wine.jpg");
	
}
  if( !src.data )
  { return -1; }

	/// Create hsv
	Mat full_hsv;
	cvtColor(src,full_hsv,CV_BGR2HSV);
	split(full_hsv, hsv);
	namedWindow("fonte", WINDOW_NORMAL);
	imshow("fonte",hsv[0]);
		
	hsv[0].copyTo(hue);
	thresholded.create( hue.size(), hue.type() );
	dst.create( src.size(), src.type() );
	dst=Scalar(0);
	
	
	
	    int num_classes=7;
vector<Scalar> color(num_classes);
	color[0]=Scalar(255,0,255);//magenta
	color[1]=Scalar(255,255,255);// branco
	color[2]=Scalar(128,128,128);// cinza
	color[3]=Scalar(0,200,0);// verde
	color[4]=Scalar(0,255,255);// amarelo
	color[5]=Scalar(0,0,255);// vermelho
	color[6]=Scalar(255,0,0);// azul
	
	vector<int>thresholds(num_classes+1);
	thresholds[0]=0;
thresholds[1]=13;
thresholds[2]=65;
thresholds[3]=115;
thresholds[4]=123;
thresholds[5]=133;
thresholds[6]=152;
thresholds[7]=179;


		namedWindow("fonte", WINDOW_NORMAL);
	imshow("fonte",hsv[2]);
	for(int i=0; i<thresholds.size()-1; i++){
		cout<<"segmentando, regra: hue entre "<<thresholds[i]<<" e "<<thresholds[i+1]<<endl<<"Pintando de "<<color[i]<<endl;
		segmentation(thresholds[i],thresholds[i+1],0,0,src.cols,src.rows,color[i]);
	}

	namedWindow("resultado", WINDOW_NORMAL);
	imshow("resultado",dst);
	
	imwrite("./resultado_regioes/wine_regioes_hue.png",dst);
    hsv[2].copyTo(hue);
	
	
thresholds[0]=0;
	thresholds[1]=30;
	thresholds[2]=100;
	thresholds[3]=155;
	thresholds[4]=200;
	thresholds[5]=237;
	thresholds[6]=248;
	thresholds[7]=255;
dst=Scalar(0);
	for(int i=0; i<thresholds.size()-1; i++){
		cout<<"segmentando, regra: value entre "<<thresholds[i]<<" e "<<thresholds[i+1]<<endl<<"Pintando de "<<color[i]<<endl;
		segmentation(thresholds[i],thresholds[i+1],0,0,src.cols,src.rows,color[i]);
	}

	namedWindow("resultado por value", WINDOW_NORMAL);
	imshow("resultado por value",dst);
	
	imwrite("./resultado_regioes/wine_regioes_value.png",dst);
	


    waitKey(0);
    
    
  return 0;
  }

