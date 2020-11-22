//
//  capstone.hpp
//  CV test
//
//  Created by Jan Ferbr on 19/11/2020.
//  Copyright © 2020 Jan Ferbr. All rights reserved.
//

#ifndef capstone_hpp
#define capstone_hpp

#include <stdio.h>
#include <iostream>
#include <opencv/cv.hpp>
#include <math.h>

using namespace cv;
using namespace std;

#endif /* capstone_hpp */

int MinMax(Mat & histogram)
{
    Mat histogramAux(256, 1, CV_32FC1), normalized_histAux;
    int i, max1, max2, min;
    max1 = 0;
    for (i = 1; i < 256; ++i){
        if(histogram.at<float>(max1) < histogram.at<float>(i)){
            max1 = i;
        }
    }
    for (i = 0; i < 256; ++i){
        histogramAux.at<float>(i) = histogram.at<float>(i) * abs(i - max1);
    }
    max2 = 0;
    for (i = 1; i < 256; ++i){
        if (histogramAux.at<float>(max2) < histogramAux.at<float>(i)){
            max2 = i;
        }
    }
    //cout << "max1 " << max1 << " max2 " << max2 << endl;
    if (max1 < max2){
        min = max1;
        for (i = min + 1; i < max2; ++i)
            if(histogram.at<float>(i) < histogram.at<float>(min))
                min = i;
    } else {
        min = max2;
        for (i = min + 1; i < max1; ++i){
            if (histogram.at<float>(i) < histogram.at<float>(min)){
                min = i;
            }
        }
    }
    //cout << "min " << min << endl;
    return (min);
}


Mat make_Histogram(Mat img)
{
    int histogram_size =256;
    float histogram_range[] = {0,255};
    const float* histogram_ranges[] = {histogram_range};
    Mat histogram;

    calcHist(&img,1,0,Mat(),histogram,1,&histogram_size,histogram_ranges);

    for(int i=0;i<histogram_size;i++){
        float bin_value=histogram.at<float>(i);
    }

    int bin_width =2;
    int histogram_width =512;
    int histogram_height =400;
    Mat normalized_histogram;

    Mat image_histogram(histogram_height,histogram_width,CV_8UC1,Scalar(255,0,0));
    normalize(histogram,normalized_histogram,0,histogram_height,NORM_MINMAX,-1,Mat());

    for(int i = 1; i <histogram_size;i++)
    {
        Point p1(bin_width*(i-1),histogram_height-cvRound(normalized_histogram.at<float>(i-1)));
        Point p2(bin_width*(i),histogram_height-cvRound(normalized_histogram.at<float>(i)));
        line(image_histogram,p1,p2,Scalar(0,0,0),2);
        }
    return histogram;
}

Mat make_Sat_Hist(Mat img){
    Mat red;
    Mat HSV;
    Mat green;
    Mat blue;
    Mat splitted[] = {red,green,blue};
    Mat sat_Hist;
    cvtColor(img, HSV ,CV_BGR2HSV);
    split(HSV, splitted);
    sat_Hist=make_Histogram(splitted[1]);
    int threshold_Val = MinMax(sat_Hist);
    Mat sat_Thresh(splitted[1].size(),CV_8UC1),res;
    cv::threshold(splitted[1], sat_Thresh, double(threshold_Val), double(255), cv::THRESH_OTSU);
    return sat_Thresh;
}

int get_Colour(Mat img,Point point){
    Mat red;
    Mat green;
    Mat blue;
    Mat RGB;
    Mat splitted[] = {red,green,blue};
    int res=0;
    
    cvtColor(img, RGB ,CV_BGR2RGB);
    split(RGB,splitted);
    
    imshow("R", splitted[0]);
    imshow("G", splitted[1]);
    imshow("B", splitted[2]);
    
    int red_value = splitted[0].at<uchar>(point);
    //cout<<"\n red value is:"<<red_value;
    int green_value = splitted[1].at<uchar>(point);
    //cout<<"\n green value is:"<<green_value;
    int blue_value = splitted[2].at<uchar>(point);
    //cout<<"\n blue value is:"<<blue_value;
    
    if (red_value > 100 && green_value < 100 && blue_value < 100){
        res=1;
        cout<<"\nobject is red";
    }
    else if (green_value > 100 && red_value < 100 && blue_value < 100){
        res=2;
        cout<<"\nobject is green";
    }
    else if (blue_value > 100 && red_value < 100 && green_value < 100){
        res=3;
        cout<<"\nobject is blue";
    }
    else
    {
        cout<<"\nOBJECT WITH UNIDENTIFIABLE COLOUR";
    }
    return res;
}


