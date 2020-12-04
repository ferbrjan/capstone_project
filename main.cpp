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
#include <math.h>
#include <time.h>


using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    //Declaration of variables
    Mat img;
    Mat imgRGB;
    Mat img_res;
    Mat labels,stats,centroids;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    //Open video
    VideoCapture cap("/Users/dinokfenicky/desktop/red_ball_1.mp4");
    
    //Errors?
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

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
        Mat thresh=make_Colour_Thresh(img,0);
        imshow("adsadasd", thresh);
        
        //Find different stats of objects
        connectedComponentsWithStats(thresh, labels, stats, centroids,8,CV_32S);
        
        //number of all objects NEEDS TO BE EDITED TO DETECT BALLS ONLY (RED,GREEN,ETC)
        int object_cnt=stats.rows;
        
        //Contours + drawings in the pic
        findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        int cnt=0;
        for( size_t i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( 0,255,0);
            double perimeter = arcLength(contours[i], true);
            double area = contourArea(contours[i]);
            //cout<<perimeter,area;
            double compactness = (perimeter*perimeter)/area;
            if (compactness>0 && compactness<15 && area>100){
                cnt++;
                drawContours( img, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
                Rect rectangl=boundingRect(contours[i]);
                double cx = rectangl.x + rectangl.width/2;
                double cy = rectangl.y + rectangl.height/2;
                char str[200];
                sprintf(str,"[%f , %f] is centre",cx, cy);
                putText(img, str, Point2f(10,20+10*(cnt+1)), FONT_HERSHEY_PLAIN, 0.8, Scalar(255,0,0));
                circle(img, Point(cx,cy), 5, Scalar (rand() & 255,rand() & 255,rand() & 255),FILLED);
            }
        }
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
