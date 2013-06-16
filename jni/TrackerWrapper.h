#include <FaceTracker/Tracker.h>
#include <DetectionBasedTracker_jni.h>
#include <opencv/highgui.h>
#include <iostream>
#include <android/log.h>

using namespace cv;

#define LOG_TAG "FaceDetection/DetectionBasedTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

//=============================================================================
void Draw(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi)
{
  int i,n = shape.rows/2; cv::Point p1,p2; cv::Scalar c;

  //draw triangulation
  c = CV_RGB(0,0,0);
  for(i = 0; i < tri.rows; i++){
    if(visi.at<int>(tri.at<int>(i,0),0) == 0 ||
       visi.at<int>(tri.at<int>(i,1),0) == 0 ||
       visi.at<int>(tri.at<int>(i,2),0) == 0)continue;
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
       shape.at<double>(tri.at<int>(i,0)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
       shape.at<double>(tri.at<int>(i,1)+n,0));
    cv::line(image,p1,p2,c);
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
       shape.at<double>(tri.at<int>(i,0)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
       shape.at<double>(tri.at<int>(i,2)+n,0));
    cv::line(image,p1,p2,c);
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
       shape.at<double>(tri.at<int>(i,2)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
       shape.at<double>(tri.at<int>(i,1)+n,0));
    cv::line(image,p1,p2,c);
  }
  //draw connections
  c = CV_RGB(0,0,255);
  for(i = 0; i < con.cols; i++){
    if(visi.at<int>(con.at<int>(0,i),0) == 0 ||
       visi.at<int>(con.at<int>(1,i),0) == 0)continue;
    p1 = cv::Point(shape.at<double>(con.at<int>(0,i),0),
       shape.at<double>(con.at<int>(0,i)+n,0));
    p2 = cv::Point(shape.at<double>(con.at<int>(1,i),0),
       shape.at<double>(con.at<int>(1,i)+n,0));
    cv::line(image,p1,p2,c,1);
  }
  //draw points
  for(i = 0; i < n; i++){    
    if(visi.at<int>(i,0) == 0)continue;
    p1 = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
    c = CV_RGB(255,0,0); cv::circle(image,p1,2,c);
  }return;
}


void test1(cv::Mat img){
  for ( int i = 0; i < img.rows / 2; i ++ )
    for ( int j = 0; j < img.cols / 2; j ++ )
      img.at<uchar>(i,j) = 225;
}

class TrackerWrapper{
public:
  //parse command line arguments
  char ftFile[256],conFile[256],triFile[256];
  bool fcheck; double scale; int fpd; bool show;
  //set other tracking parameters
  std::vector<int> wSize1; 
  std::vector<int> wSize2; 
  int nIter; double clamp,fTol; 
  FACETRACKER::Tracker *model;
  cv::Mat tri;
  cv::Mat con;
  
  //initialize camera and display window
  cv::Mat frame,gray,im; double fps; char sss[256]; std::string text; 
  
  int64 t1,t0; int fnum;
  

  //loop until quit (i.e user presses ESC)
  bool failed;

  TrackerWrapper(){

    nIter = 5; clamp = 3; fTol = 0.01;
    wSize1 = std::vector<int>(1);
    wSize2 = std::vector<int>(3);
    fcheck = false;
    scale = 1; fpd = -1; show = true;
    fps = 0; fnum = 0; t0 = cvGetTickCount();
    failed = true;
    wSize1[0] = 7; wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;

    model = new FACETRACKER::Tracker("/storage/sdcard1/face2.tracker");
    tri = FACETRACKER::IO::LoadTri("/storage/sdcard1/face.tri");
    con = FACETRACKER::IO::LoadCon("/storage/sdcard1/face.con");

  }

  
  void tick(cv::Mat* I){

    im = *I;

    //track this image
    std::vector<int> wSize; if(failed)wSize = wSize2; else wSize = wSize1;

    if(model->Track(im,wSize,fpd,nIter,clamp,fTol,fcheck) == 0){
      int idx = model->_clm.GetViewIdx();
      failed = false;
      Mat m = model->_clm._visi[idx];
      Draw(im,model->_shape,con,tri,m);
    }else{
      if(show){cv::Mat R(im,cvRect(0,0,150,50)); R = cv::Scalar(0,0,255);}
      model->FrameReset(); failed = true;
    }     

    //draw framerate on display image 
    if(fnum >= 9){      
      t1 = cvGetTickCount();
      fps = 10.0/((double(t1-t0)/cvGetTickFrequency())/1e+6); 
      t0 = t1; fnum = 0;
    }else fnum += 1;
    if(show){
      sprintf(sss,"%d frames/sec",(int)floor(fps)); text = sss;
      cv::putText(im,text,cv::Point(10,20),
      CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
    }
  }
};
