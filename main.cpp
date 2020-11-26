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
    VideoCapture cap("/Users/dinokfenicky/desktop/red_ball_3.mp4");
    
    //Errors?
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    while(1){
        //take a frame from video capture
        cap>>img;
        
        //FPS?!?!
        //CV_CAP_PROP_FPS
        
        //if the frame is empty, break
        if (img.empty())
            break;

        //resize img to fit the screen
        cv::resize(img, img, cv::Size(), 0.25, 0.25);
        
        //Thresholding
        Mat sat_Thresh=make_Sat_Hist(img);
        
        //Find different stats of objects
        connectedComponentsWithStats(sat_Thresh, labels, stats, centroids,8,CV_32S);
        
        //number of all objects NEEDS TO BE EDITED TO DETECT BALLS ONLY (RED,GREEN,ETC)
        int object_cnt=stats.rows;
        
        //Text
        for (int i=1;i<object_cnt;i++){
            char str[200];
            sprintf(str,"[%f , %f] is centre",centroids.at<double>(i, 0), centroids.at<double>(i, 1));
            putText(img, str, Point2f(10,20+10*i), FONT_HERSHEY_PLAIN, 0.8, Scalar(255,0,0));
            
            //Colour? works for red only rn, different thresholds needed for green and blue (thresholding???)
            int colour=get_Colour(img, Point(centroids.at<double>(i, 0),centroids.at<double>(i, 1)));
            //cout<<"\n"<<colour;
        }
        
        //Contours
        findContours(sat_Thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        for( size_t i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( 0,255,0);
            drawContours( img, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
        }
        
        //Results
        namedWindow("original",CV_WINDOW_AUTOSIZE);
        imshow("original",img);
        
        //press esc to exit video
        char c=(char)waitKey(25);
        if(c==27)
            break;
        
    }
    destroyWindow("original");
    
    Mat t_labels,t_features;
    t_features=prepare_training_features();
    t_labels=prepare_training_labels();
    
    cout<<"Labels are"<<t_labels<<"\nFeatures are"<<t_features<<endl;
    
    create_bayes(t_features, t_labels);
}
