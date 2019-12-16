#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;



int drawPeaks(Mat &histImage, vector<int>& peaks, int hist_size = 256, Scalar color = Scalar(0, 0, 255))
{
    int bin_w = cvRound( (double) histImage.cols / hist_size );
    for(size_t i = 0; i < peaks.size(); i++)
        line(histImage, Point(bin_w * peaks[i], histImage.rows), Point(bin_w * peaks[i], 0), color);

    imshow("Peaks", histImage);
    return EXIT_SUCCESS;
}

int drawThresholds(Mat &histImage, vector<int>& peaks, int hist_size = 256, Scalar color = Scalar(255, 0, 0))
{
    int bin_w = cvRound( (double) histImage.cols / hist_size );
    for(size_t i = 0; i < peaks.size(); i++)
        line(histImage, Point(bin_w * peaks[i], histImage.rows), Point(bin_w * peaks[i], 0), color);

    imshow("Thresholds", histImage);
    return EXIT_SUCCESS;
}

Mat drawHistogram(Mat &hist, bool show=true, int hist_h = 400, int hist_w = 1024, int hist_size = 256, Scalar color = Scalar(255, 255, 255), int type = 2)
{
    int bin_w = cvRound( (double) hist_w/hist_size );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    switch (type) {
    case 1:
        for(int i = 0; i < histImage.cols; i++)
        {
            const unsigned x = i;
            const unsigned y = hist_h;

            line(histImage, Point(bin_w * x, y),
                 Point(bin_w * x, y - cvRound(hist.at<float>(i))),
                 color);
        }

        break;
    case 2:
        for( int i = 1; i < hist_size; ++i)
        {
            Point pt1 = Point(bin_w * (i-1), hist_h);
            Point pt2 = Point(bin_w * i, hist_h);
            Point pt3 = Point(bin_w * i, hist_h - cvRound(hist.at<float>(i)));
            Point pt4 = Point(bin_w * (i-1), hist_h - cvRound(hist.at<float>(i-1)));
            Point pts[] = {pt1, pt2, pt3, pt4, pt1};

            fillConvexPoly(histImage, pts, 5, color);
        }
        break;
    default:
        for( int i = 1; i < hist_size; ++i)
        {
            line( histImage, Point( bin_w * (i-1), hist_h - cvRound(hist.at<float>(i-1))) ,
                             Point( bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
                             color, 1, 8, 0);
        }

        break;
    }
if (show){
    imshow("Histogram", histImage);
}
    return histImage;
}

struct Length
{
    int pos1;
    int pos2;
    int size()
    {
        return pos2 - pos1 +1;
    }
};

struct PeakInfo
{
    int pos;
    int left_size;
    int right_size;
    float value;
};





/** @function main */
int main( int argc, char** argv )
{

Mat src, src_gray;
vector<Mat> hsv(3);
Mat dst, detected_edges;


if (argc>1){
  /// Load an image
  src = imread( argv[1] );

}else{ src = imread("./image-seg/fruit.jpg");
	
}
  if( !src.data )
  { return -1; }
	
	
	/// Create hsv
	Mat full_hsv;
	cvtColor(src,full_hsv,CV_BGR2HSV);
	split(full_hsv, hsv);
	namedWindow("fonte", WINDOW_NORMAL);
	imshow("fonte",hsv[0]);
	


    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist;



    /// Compute the histograms of h:

    calcHist( &hsv[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	cout<<"histograma do hue calculado"<<endl;

	drawHistogram(b_hist);
	
int num_classes=9;
vector<int> peaks(num_classes);
peaks[0]=0;
peaks[1]=6;
peaks[2]=14;
peaks[3]=20;
peaks[4]=46;
peaks[5]=103;
peaks[6]=165;
peaks[7]=173;
peaks[8]=177;
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
    cout<<peaks.size()<<endl;
    
 Mat histImg = drawHistogram(b_hist,false);
    drawPeaks(histImg, peaks);
    
    if(peaks.size()<1){
		
		cout<<"não foram encontrados picos suficientes"<<endl;
		waitKey(0);
		return -1;
		}
    for(int i=0; i<peaks.size(); i++){
		cout<<peaks[i]<<endl;
	}

	vector<int>thresholds;
		cout<<"limiares"<<endl;
	for(int i=0; i<peaks.size()-1; i++){
		thresholds.push_back((peaks[i]+peaks[i+1])/2);
		cout<<thresholds[i]<<endl;
	}
	thresholds.push_back(255);
	drawThresholds(histImg, thresholds);
	
	vector<Mat> segmented(thresholds.size());
	vector<Mat> diff(thresholds.size());
	Mat seg, dst_final;
	seg.create( hsv[0].size(), hsv[0].type() );
	

	for (int k=0;k<thresholds.size();k++){
		
		threshold( hsv[0],segmented[k], thresholds[k],255,1 );//(k*255/(thresholds.size()-1)) 
		
		if(k!=0){
			diff[k]=segmented[k]-segmented[k-1];
			
			}else{
				diff[k]=segmented[k];	
			}
		
		addWeighted( seg, 1, segmented[k], 1, 0.0, seg);
	
	}
	//threshold( seg,diff[0],128,255,1 );

	
	dst_final.create( src.size(), src.type() );
	Mat result=Mat::zeros(Size(seg.cols, seg.rows), CV_8UC1);
	for (int k=0;k<thresholds.size();k++){		
		addWeighted( result, 1, diff[k], (k*1.0/(thresholds.size())), 0.0, result);
		dst = Scalar::all(0);
		Mat cor;
		cor.create( src.size(), src.type() );
		cor.setTo(color[k]);
		cor.copyTo( dst, diff[k]);		
		addWeighted( dst_final, 1, dst, 1, 0.0, dst_final);
		
	}	

       	
    namedWindow("resultado hue", WINDOW_NORMAL);
	imshow("resultado hue",dst_final);
	imwrite("./resultado_limiar/fruit_limiar_hue.png",dst_final);

    
    
    cout<<"aperte uma tecla para mostrar o proximo passo(segmentacao por limiar pelo value)"<<endl;
    waitKey(0);
    
    //VALUE
    namedWindow("fonte", WINDOW_NORMAL);
	imshow("fonte",hsv[2]);
    
    
    
    calcHist( &hsv[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	cout<<"histograma do hue calculado"<<endl;

	drawHistogram(b_hist);


peaks[0]=0;
peaks[1]=23;
peaks[2]=27;
peaks[3]=30;
peaks[4]=35;
peaks[5]=63;
peaks[6]=135;
peaks[7]=193;
peaks[8]=250;

 histImg = drawHistogram(b_hist,false);
    drawPeaks(histImg, peaks);
    
    if(peaks.size()<1){		
		cout<<"não foram encontrados picos suficientes"<<endl;
		waitKey(0);
		return -1;
		}
    for(int i=0; i<peaks.size(); i++){
		cout<<peaks[i]<<endl;
	}

	thresholds.clear();
		cout<<"limiares"<<endl;
	for(int i=0; i<peaks.size()-1; i++){
		thresholds.push_back((peaks[i]+peaks[i+1])/2);
		cout<<thresholds[i]<<endl;
	}
	thresholds.push_back(255);
	drawThresholds(histImg, thresholds);
	
	vector<Mat> segmented2(thresholds.size());
	vector<Mat> diff2(thresholds.size());
	
	seg=Scalar(0);
	for (int k=0;k<thresholds.size();k++){		
		threshold( hsv[2],segmented2[k], thresholds[k],255,1 );//(k*255/(thresholds.size()-1)) 
		
		if(k!=0){
			diff2[k]=segmented2[k]-segmented2[k-1];
			
			}else{
				diff2[k]=segmented2[k];	
			}
		
		addWeighted( seg, 1, segmented2[k], 1, 0.0, seg);
	
	}
	//threshold( seg,diff[0],128,255,1 );

	
	dst_final.create( src.size(), src.type() );
	result=Mat::zeros(Size(seg.cols, seg.rows), CV_8UC1);
	for (int k=0;k<thresholds.size();k++){		
		addWeighted( result, 1, diff[k], (k*1.0/(thresholds.size())), 0.0, result);
		dst = Scalar::all(0);
		Mat cor;
		cor.create( src.size(), src.type() );
		cor.setTo(color[k]);
		cor.copyTo( dst, diff[k]);		
		addWeighted( dst_final, 1, dst, 1, 0.0, dst_final);
		
	}	

       	
    namedWindow("resultado value", WINDOW_NORMAL);
	imshow("resultado value",dst_final);
	imwrite("./resultado_limiar/fruit_limiar_value.png",dst_final);
    waitKey(0);
    
    
    return 0;
    
    
  }
