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
#include <opencv/ml.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <math.h>

#define NCLASS1 3
#define NCLASS2 4
#define NCLASS3 13
#define TRAIN_SAMPLES (NCLASS1+NCLASS2+NCLASS3)

using namespace cv;
using namespace std;

#endif /* capstone_hpp */

/*
static Ptr<ml::TrainData> prep_train_data(const Mat& data, const Mat& responses, int ntrain_samples){
    Mat sample_idx= Mat::zeros(1, data.rows,CV_8U);
    Mat train_samples=sample_idx.colRange(0,ntrain_samples);
    train_samples.setTo(Scalar::all(1));
    
    int nvars = data.cols;
    Mat var_type(nvars + 1,1,CV_8U);
    var_type.setTo(Scalar::all(ml::VAR_ORDERED));
    var_type.at<uchar>(nvars)=ml::VAR_CATEGORICAL;
    
    return ml::TrainData::create(data,ml::ROW_SAMPLE,responses,noArray(),sample_idx,noArray(),var_type);
}

Mat prepare_training_features(){
    Mat image,binary_img,bullshit;
    vector<vector<Point>> contours;
    
    int n_contours,k=0;
    double perimeter,area;
    
    Mat train_features(TRAIN_SAMPLES,2,CV_32FC1);
    Mat train_labels(TRAIN_SAMPLES,1,CV_32FC1);
    
    string classes_names[3] = {"Nail","Screw","Thumbtack"};
    string imgs_names[3] = {"Clase1.tif","Clase2.tif","Clase3.tif"};
    for(int i=0 ; i<3 ; i++){
        string route = ("/Users/dinokfenicky/downloads/lesson_9/" + (imgs_names[i]));
        image=imread(route,IMREAD_GRAYSCALE);
        if(!image.data){
            cerr<<"ERROR LOADING TRAINING"<<endl;
            return bullshit;
        }
        
        binary_img=Mat(image.size(),CV_8UC1);
        threshold(image,binary_img,128,255,THRESH_BINARY);
        
        //imshow("training",binary_img);
        //waitKey(0);
        
        findContours(binary_img.clone(), contours, RETR_LIST,CHAIN_APPROX_NONE);
        n_contours = contours.size();
        
        cout<<"Total number of "<<imgs_names[i]<<":"<<n_contours<<endl;
        
        for (size_t idx =0;idx<n_contours;idx++,k++){
            area=contourArea(contours[idx],false);
            perimeter=arcLength(contours[idx], true);
            
            train_features.at<float>(k,0)=perimeter;
            train_features.at<float>(k,1)=area;
            train_labels.at<float>(k)=i;
            
        }
    }
    return train_features;
}

Mat prepare_training_labels(){
    Mat image,binary_img,bullshit;
    vector<vector<Point>> contours;
    
    int n_contours,k=0;
    double perimeter,area;
    
    Mat train_features(TRAIN_SAMPLES,2,CV_32FC1);
    Mat train_labels(TRAIN_SAMPLES,1,CV_32FC1);
    
    string classes_names[3] = {"Nail","Screw","Thumbtack"};
    string imgs_names[3] = {"Clase1.tif","Clase2.tif","Clase3.tif"};
    for(int i=0 ; i<3 ; i++){
        string route = ("/Users/dinokfenicky/downloads/lesson_9/" + (imgs_names[i]));
        image=imread(route,IMREAD_GRAYSCALE);
        if(!image.data){
            cerr<<"ERROR LOADING TRAINING"<<endl;
            return bullshit;
        }
        
        binary_img=Mat(image.size(),CV_8UC1);
        threshold(image,binary_img,128,255,THRESH_BINARY);
        
        //imshow("training",binary_img);
        //waitKey(0);
        
        findContours(binary_img.clone(), contours, RETR_LIST,CHAIN_APPROX_NONE);
        n_contours = contours.size();
        
        cout<<"Total number of "<<imgs_names[i]<<":"<<n_contours<<endl;
        
        for (size_t idx =0;idx<n_contours;idx++,k++){
            area=contourArea(contours[idx],false);
            perimeter=arcLength(contours[idx], true);
            
            train_features.at<float>(k,0)=perimeter;
            train_features.at<float>(k,1)=area;
            train_labels.at<float>(k)=i;
            
        }
    }
    return train_labels;
}



void create_bayes(Mat train_features,Mat train_labels){
    int ntrain_samples = (int)(TRAIN_SAMPLES);
    int n_contours;
    double perimeter,area;
    vector<vector<Point>> contours;
    string classes_names[3] = {"Nail","Screw","Thumbtack"};
    
    Ptr<ml::NormalBayesClassifier>bayes = ml::NormalBayesClassifier::create();
    Ptr<ml::TrainData>tdata = prep_train_data(train_features, train_labels, ntrain_samples);
    bayes->train(tdata);
    //test
    Mat test_features(1,2,CV_32FC1);
    Mat image,binary_img;
    image=imread("/Users/dinokfenicky/downloads/lesson_9/test.tif",IMREAD_GRAYSCALE);
    if(!image.data){
        cerr<<"ERROR LOADING TRAINING"<<endl;
        //return!!!
    }
    binary_img = Mat(image.size(),CV_8UC1);
    threshold(image,binary_img,128,255,THRESH_BINARY);
    
    Mat color_img(image.size(),CV_8UC3);
    cvtColor(image, color_img, CV_GRAY2BGR);
    
    findContours(binary_img.clone(), contours, RETR_LIST,CHAIN_APPROX_NONE);
    n_contours = contours.size();
    cout<<"Total number of pieces"<<":"<<n_contours<<endl;
    Mat test_predcition(n_contours,1,CV_32FC1);
    
    for(size_t idx=0;idx<n_contours;idx++){
        area=contourArea(contours[idx],false);
        perimeter=arcLength(contours[idx], true);
        
        test_features.at<float>(0,0)=perimeter;
        test_features.at<float>(0,1)=area;
        
        test_predcition.at<float>(idx)=bayes->predict(test_features);
        
        cvtColor(image,color_img, CV_GRAY2BGR);
        Scalar color(0,0,255);
        drawContours(color_img, contours, idx, color,2);
        imshow("Test",color_img);
        
        cout<<"Highlited piece is a "<<classes_names[(int)test_predcition.at<float>(idx)]<<endl;
        waitKey(0);
    }
    
    
}

*/

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

Mat make_Colour_Thresh(Mat img,int colour_code){ //0 for red , 1 for blue, 2 for green
    Mat HSV,H,S,V;
    Mat splitted[]={H,S,V};
    cvtColor(img, HSV ,CV_BGR2HSV);
    Mat mask1,mask2;
    //split(HSV, splitted);
    //imshow("V",splitted[1]);
    if (colour_code==0){
        inRange(HSV, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
        inRange(HSV, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);
        mask1 = mask1 + mask2;
    }
    if (colour_code==1){
        inRange(HSV, Scalar(100, 120, 70), Scalar(140, 255, 255), mask1);
    }
    if (colour_code==2){
        inRange(HSV, Scalar(40, 120, 70), Scalar(80, 255, 255), mask1);
    }
    
    return mask1;
}

void Erosion( int erosion_elem, int erosion_size, Mat src,void*)
{
    Mat erosion_dst;
    int erosion_type = 0;
    if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
    else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
    else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement( erosion_type, Size( 2*erosion_size + 1, 2*erosion_size+1 ),Point( erosion_size, erosion_size ) );
    erode( src, src, element );
    imshow( "Erosion Demo", src );
}

