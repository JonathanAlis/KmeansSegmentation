#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;

using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
string window_name = "Edge Map";


float precision(Mat im1, Mat im2){
   int tam=im1.rows*im2.cols;
   Mat xored;
   bitwise_xor(im1,im2,xored);
   int miss=sum(xored)[0]/255;
   int hit=tam-miss;
   cout<<100.0*hit/tam<<endl;
   return 100.0*hit/tam;
	}

void toDistinctGray( const Mat& colored, Mat& gray,vector<Scalar> colorMap)
{
		vector<Mat> bgr(3);
		
		split(colored, bgr);
		int numColors=colorMap.size();
		vector<Scalar> grayMap;
		for (int i=0;i<numColors;i++){
			grayMap.push_back(Scalar(255*i/(numColors-1),0,0,0));
		}
        gray.create( bgr[0].size(), CV_8UC1 );
        gray = Scalar::all(0);

    for( int i = 0; i < bgr[0].rows; i++ ){
		uchar* rowi = gray.ptr<uchar>(i);
        for( int j = 0; j < bgr[0].cols; j++ ){
			for (int k=0;k<numColors;k++){
				
				Scalar bgrValue = Scalar(bgr[0].at<uchar>(Point(j, i)),bgr[1].at<uchar>(Point(j, i)),bgr[2].at<uchar>(Point(j, i)),0);
				if (norm(colorMap[k]-bgrValue)==0){
					rowi[j]=grayMap[k][0];
				}
			}
        }
	}
}

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(5,5) );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);
  detected_edges.copyTo( dst);
  //src.copyTo( dst, detected_edges);
  imshow( window_name, dst );
  
 }


/** @function main */
int main( int argc, char** argv )
{
Mat limiar_h;
Mat gray_limiar_h;
Mat limiar_v;
Mat gray_limiar_v;
Mat regions_h;
Mat gray_regions_h;
Mat regions_v;
Mat gray_regions_v;
Mat cluster1_hsv;
Mat gray_cluster1_hsv;
Mat cluster1_bgr;
Mat gray_cluster1_bgr;
Mat cluster2_hsv;
Mat gray_cluster2_hsv;
Mat cluster2_bgr;
Mat gray_cluster2_bgr;

string name="fruit";
int num_classes=7;
vector<Scalar> color(num_classes);
	color[0]=Scalar(0,0,0);// fundo vai ser preto
	color[1]=Scalar(255,255,255);// chao cinza medio
	color[2]=Scalar(128,128,128);// prata cinza
	color[3]=Scalar(0,200,0);// 
	color[4]=Scalar(0,255,255);// 
	color[5]=Scalar(0,0,255);// 
	color[6]=Scalar(150,150,255);//
	
src = imread("./image-seg/"+name+".jpg");
limiar_h = imread("./resultado_limiar/"+name+"_limiar_hue.png");
toDistinctGray(limiar_h,gray_limiar_h,color);
namedWindow("lim hue",WINDOW_NORMAL);
imshow("lim hue",gray_limiar_h);
waitKey(10);

limiar_v = imread("./resultado_limiar/"+name+"_limiar_value.png");
toDistinctGray(limiar_h,gray_limiar_v,color);
namedWindow("lim value",WINDOW_NORMAL);
imshow("lim value",gray_limiar_v);
waitKey(10);

regions_h = imread("./resultado_regioes/"+name+"_regioes_hue.png");
toDistinctGray(regions_h,gray_regions_h,color);
namedWindow("regions hue",WINDOW_NORMAL);
imshow("regions hue",gray_regions_h);
waitKey(10);

regions_v = imread("./resultado_regioes/"+name+"_regioes_value.png");
toDistinctGray(regions_v,gray_regions_v,color);
namedWindow("regions value",WINDOW_NORMAL);
imshow("regions value",gray_regions_v);
waitKey(10);

cluster1_bgr = imread("./clustering_random/"+name+"_BGR_final.png");
toDistinctGray(cluster1_bgr,gray_cluster1_bgr,color);
namedWindow("cluster1 bgr",WINDOW_NORMAL);
imshow("cluster1 bgr",gray_cluster1_bgr);
waitKey(10);
cluster1_hsv = imread("./clustering_random/"+name+"_HSV_final.png");
toDistinctGray(cluster1_hsv,gray_cluster1_hsv,color);
namedWindow("cluster1 hsv",WINDOW_NORMAL);
imshow("cluster1 hsv",gray_cluster1_hsv);
waitKey(10);
cluster2_bgr = imread("./clustering_chosen/"+name+"_BGR_final.png");
toDistinctGray(cluster2_bgr,gray_cluster2_bgr,color);
namedWindow("cluster2 bgr",WINDOW_NORMAL);
imshow("cluster2 bgr",gray_cluster2_bgr);
waitKey(10);
cluster2_hsv = imread("./clustering_chosen/"+name+"_HSV_final.png");
toDistinctGray(cluster2_hsv,gray_cluster2_hsv,color);
namedWindow("cluster2 hsv",WINDOW_NORMAL);
imshow("cluster2 hsv",gray_cluster2_hsv);
waitKey(10);

  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Create a window
  namedWindow( window_name, WINDOW_NORMAL );

  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

  /// Show the image
  CannyThreshold(0, 0);

  
  while(1){
	  
	  cout<<"Aperte um botão para fazer a comparacao, a partir do limiar escolhido na barra."<<endl<<"Aperte 'q' para sair."<<endl;  
  char q= waitKey(0);
  if (q=='q') return 0;
  
  cout<<endl<<"limiar menor do canny="<<lowThreshold<<endl;
  cout<<"limiar maior do canny="<<ratio*lowThreshold<<endl;
  
  Canny( gray_limiar_v, limiar_v, lowThreshold, lowThreshold*ratio, kernel_size );
  imshow("lim value",limiar_v);
  cout<<"limiar value precision=";
  precision(limiar_v,dst);
  
  Canny( gray_limiar_h, limiar_h, lowThreshold, lowThreshold*ratio, kernel_size );
  imshow("lim hue",limiar_h);
  cout<<"limiar hue precision=";
  precision(limiar_h,dst);
  
  Canny( gray_regions_v, regions_v, lowThreshold, lowThreshold*ratio, kernel_size );
  imshow("regions value",regions_v);
  cout<<"regions value precision=";
  precision(regions_v,dst);
  
  Canny( gray_regions_h, regions_h, lowThreshold, lowThreshold*ratio, kernel_size );
  imshow("regions hue",regions_h);
  cout<<"regions hue precision=";
  precision(regions_h,dst);
  
  
  Canny( gray_cluster1_bgr, cluster1_bgr, lowThreshold, lowThreshold*ratio, kernel_size );
  imshow("cluster1 bgr",cluster1_bgr);
  cout<<"cluster random bgr precision=";
  precision(cluster1_bgr,dst);
  
  Canny( gray_cluster1_hsv, cluster1_hsv, lowThreshold, lowThreshold*ratio, kernel_size );
  imshow("cluster1 hsv",cluster1_hsv);
  cout<<"cluster random hsv precision=";
  precision(cluster1_bgr,dst);
  
  Canny( gray_cluster2_bgr, cluster2_bgr, lowThreshold, lowThreshold*ratio, kernel_size );
  imshow("cluster2 bgr",cluster2_bgr);
  cout<<"cluster chosen bgr precision=";
  precision(cluster2_bgr,dst);  
  
  Canny( gray_cluster2_hsv, cluster2_hsv, lowThreshold, lowThreshold*ratio, kernel_size );
  imshow("cluster2 hsv",cluster2_hsv);
  cout<<"cluster chosen hsv precision=";
  precision(cluster2_bgr,dst);
	waitKey(0);
}
  return 0;
  }
