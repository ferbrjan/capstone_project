//
//  capstone.cpp
//  CV test
//
//  Created by Jan Ferbr on 19/11/2020.
//  Copyright © 2020 Jan Ferbr. All rights reserved.
//

#include "capstone.hpp"
#include <iostream>
#include <opencv/cv.hpp>
#include "opencv2/imgcodecs.hpp"
#include <math.h>
#include <chrono>
#include <time.h>


using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    //Declaration of variables
    Mat img;
    Mat imgRGB;
    Mat img_res;
    Mat grayscale,HSV,H,S,V;
    Mat splitted[]={H,S,V};
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    //Open video
    VideoCapture cap("/Users/dinokfenicky/desktop/two_balls_1.mp4");//To use the webcam just use VideoCapture cap(0);
    //Jan's route to video pls dont erase :DD "/Users/dinokfenicky/desktop/two_balls_1.mp4"
    
    //Errors?
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    
    cap>>img;
    cv::resize(img, img, cv::Size(), 0.4, 0.4);
    
    Point p;
    namedWindow("Select ball", 1);
    setMouseCallback("Select ball", CallBackFunc, &p);
    imshow("Select ball", img);
    waitKey(0);
    destroyWindow("Select ball");
    
    while(1){
        
        auto start = std::chrono::system_clock::now();
        
        //take a frame from video capture
        cap>>img;
        
        //if the frame is empty, break
        if (img.empty())
            break;

        //resize img to fit the screen
        cv::resize(img, img, cv::Size(), 0.4, 0.4);
        
        //Thresholding
        Mat thresh=make_Colour_Thresh(img,p);
        imshow("adsadasd", thresh);
        
        //Hough circles detection (make this a function??)
        //Detects balls of all colours!!!
        cvtColor(img, grayscale, CV_BGR2GRAY);
        GaussianBlur( grayscale, grayscale, Size(9, 9), 3, 3 );
        vector<Vec3f> circles;
        HoughCircles(grayscale, circles, HOUGH_GRADIENT, 1,thresh.rows/16,100, 30, 10, 50);
        int objectcnt=0;
        for( size_t i = 0; i < circles.size(); i++ )
        {
            Vec3i c = circles[i];
            Point center = Point(c[0], c[1]);
            //cout<<"\n"<<((int)thresh.at<uchar>(center));
            if ((int)thresh.at<uchar>(center)==255)
            {
                // circle center
                objectcnt++;
                circle( img, center, 1, Scalar(0,100,100), 3, LINE_AA);
                // circle outline
				objectcnt++;
                int radius = c[2];
                circle( img, center, radius, Scalar(255,0,255), 3, LINE_AA);
                char str[200];
                double cx = center.x;
                double cy = center.y;

                sprintf(str,"[%f , %f] is centre %i",cx, cy, objectcnt);
                putText(img, str, Point2f(10,20+10*(objectcnt+2)), FONT_HERSHEY_PLAIN, 0.8, Scalar(255,0,0));
                circle(img, Point(cx,cy), 5, Scalar (rand() & 255,rand() & 255,rand() & 255),FILLED);
            }
        }
        
        //Contours + drawings in the pic
        findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        int cnt=0;
        for( size_t i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( 255,0,255);
            double perimeter = arcLength(contours[i], true);
            double area = contourArea(contours[i]);
            //cout<<perimeter,area;
            double compactness = (perimeter*perimeter)/area;
            if (compactness>0 && compactness<15 && area>100){
                cnt++;
                drawContours( img, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
                //DOES NOT WORK FOR WHITE
            }
        }
        
        //FPS
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end-start;
        int FPS = 1 / diff.count();
        char str[200];
        sprintf(str,"FPS: %i",FPS);
        putText(img, str, Point2f(10,20+10), FONT_HERSHEY_PLAIN, 0.8, Scalar(255,0,0));
        
        //Results
        namedWindow("original",CV_WINDOW_AUTOSIZE);
        imshow("original",img);
        
        //press esc to exit video
        char c=(char)waitKey(25);
        if(c==27)
            break;
    }
    destroyWindow("original");
    
    //Mat t_labels,t_features;
    //t_features=prepare_training_features();
    //t_labels=prepare_training_labels();
    
    //cout<<"Labels are"<<t_labels<<"\nFeatures are"<<t_features<<endl;
    
    //create_bayes(t_features, t_labels);
}
