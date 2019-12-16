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

PeakInfo peakInfo(int pos, int left_size, int right_size, float value)
{
    PeakInfo output;
    output.pos = pos;
    output.left_size = left_size;
    output.right_size = right_size;
    output.value = value;
    return output;
}

vector<PeakInfo> findPeaks(InputArray _src, int window_size)
{
    Mat src = _src.getMat();

    Mat slope_mat = src.clone();

    // Transform initial matrix into 1channel, and 1 row matrix
    Mat src2 = src.reshape(1, 1);

    int size = window_size / 2;

    Length up_hill, down_hill;
    vector<PeakInfo> output;

    int pre_state = 0;
    int i = size;

    while(i < src2.cols - size)
    {
        float cur_state = src2.at<float>(i + size) - src2.at<float>(i - size);

        if(cur_state > 0)
            cur_state = 2;
        else if(cur_state < 0)
            cur_state = 1;
        else cur_state = 0;

        // In case you want to check how the slope looks like
        slope_mat.at<float>(i) = cur_state;

        if(pre_state == 0 && cur_state == 2)
            up_hill.pos1 = i;
        else if(pre_state == 2 && cur_state == 1)
        {
            up_hill.pos2 = i - 1;
            down_hill.pos1 = i;
        }

        if((pre_state == 1 && cur_state == 2) || (pre_state == 1 && cur_state == 0))
        {
            down_hill.pos2 = i - 1;
            int max_pos = up_hill.pos2;
            if(src2.at<float>(up_hill.pos2) < src2.at<float>(down_hill.pos1))
                max_pos = down_hill.pos1;

            PeakInfo peak_info = peakInfo(max_pos, up_hill.size(), down_hill.size(), src2.at<float>(max_pos));

            output.push_back(peak_info);
        }
        i++;
        pre_state = (int)cur_state;
    }
    return output;
}

vector<int> getLocalMaximum(InputArray _src, int smooth_size = 9, int neighbor_size = 3, float peak_per = 0.5) //if you play with the peak_per attribute value, you can increase/decrease the number of peaks found
{
    Mat src = _src.getMat().clone();

    vector<int> output;
    GaussianBlur(src, src, Size(smooth_size, smooth_size), 0);
    vector<PeakInfo> peaks = findPeaks(src, neighbor_size);

    double min_val, max_val;
    minMaxLoc(src, &min_val, &max_val);

    for(size_t i = 0; i < peaks.size(); i++)
    {
        if(peaks[i].value > max_val * peak_per && peaks[i].left_size >= 2 && peaks[i].right_size >= 2)
            output.push_back(peaks[i].pos);
    }

    Mat histImg = drawHistogram(src);
    drawPeaks(histImg, output);
    return output;
}




/** @function main */
int main( int argc, char** argv )
{

Mat src, src_gray;
vector<Mat> hsv(3);
Mat dst, detected_edges;


if (argc>1){
  /// Load an image
  src = imread( argv[1] );

}else{ src = imread("./image-seg/wine.jpg");
	
}
  if( !src.data )
  { return -1; }
	dst.create( src.size(), src.type() );
	
	
	/// Create hsv
	Mat full_hsv;
	cvtColor(src,full_hsv,CV_BGR2HSV);
	split(full_hsv, hsv);
	namedWindow("h", WINDOW_NORMAL);
	imshow("h",hsv[0]);


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
	
int num_classes=7;
vector<int> peaks(num_classes);
peaks[0]=6;
peaks[1]=20;
peaks[2]=110;
peaks[3]=120;
peaks[4]=126;
peaks[5]=140;
peaks[6]=165;
//peaks[7]=173;
vector<Scalar> color(num_classes);
	color[0]=Scalar(255,0,255);//magenta
	color[1]=Scalar(255,255,255);// branco
	color[2]=Scalar(128,128,128);// cinza
	color[3]=Scalar(0,200,0);// verde
	color[4]=Scalar(0,255,255);// amarelo
	color[5]=Scalar(0,0,255);// vermelho
	color[6]=Scalar(255,0,0);// azul
	
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

       	
           	namedWindow("resultado", WINDOW_NORMAL);
	imshow("resultado",dst_final);
	imwrite("./resultado_limiar/wine_limiar_hue.png",dst_final);
    waitKey(0);
    
    
       
    //VALUE
    namedWindow("fonte", WINDOW_NORMAL);
	imshow("fonte",hsv[2]);
    
    
    
    calcHist( &hsv[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	cout<<"histograma do hue calculado"<<endl;

	drawHistogram(b_hist);


peaks[0]=0;
peaks[1]=60;
peaks[2]=140;
peaks[3]=170;
peaks[4]=230;
peaks[5]=245;
peaks[6]=255;

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
	imwrite("./resultado_limiar/wine_limiar_value.png",dst_final);
    waitKey(0);
    
    
    return 0;
    
    
  }
