#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\video\background_segm.hpp"
#include "UtilityFunction.h"
#include "VisionAlgorithm.h"
#include "auxfunc.h"
#include "opencv2\video\tracking.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include "WinStopWatch.h"
#include "VisionDataStruct.h"
#include "TestSub.h"
#include <memory>
#include <iostream>
using namespace std;
using namespace cv;

int TestNotImage()
{
    //display the original image
    IplImage* img = cvLoadImage("./MyPic.jpg");

    //cvSetImageROI ()
    cvNamedWindow("MyWindow");
    cvShowImage("MyWindow", img);

    //invert and display the inverted image
    cvNot(img, img);
    cvNamedWindow("Inverted");
    cvShowImage("Inverted", img);

    cvWaitKey(0);

    //cleaning up
    cvDestroyWindow("MyWindow");
    cvDestroyWindow("Inverted");
    cvReleaseImage(&img);

    return 0;
}

int TestAdjustIntensityImage()
{
    IplImage* img = cvLoadImage("./MyPic.jpg");
    Mat mat1 = cvarrToMat(img);

    IplImage* im1g = new IplImage( mat1 );
    cvShowImage ("im1g", im1g );

    waitKey(0);


    //cvCreate

    Mat mat = imread("MyPic.jpg", CV_LOAD_IMAGE_COLOR);

    int intensityToChange = 40;
    //invert and display the inverted image
    Mat matH = mat + Scalar( intensityToChange, intensityToChange, intensityToChange );
    Mat matL = mat - Scalar( intensityToChange, intensityToChange, intensityToChange );

    namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    namedWindow("High Brightness", CV_WINDOW_AUTOSIZE);
    namedWindow("Low Brightness", CV_WINDOW_AUTOSIZE);

    imshow("Original Image", mat);
    imshow("High Brightness", matH);
    imshow("Low Brightness", matL);

    waitKey(0);

    destroyAllWindows(); //destroy all open windows

    //cvReleaseImage(&img);

    return 0;
}

int TestAdjustContrastImage()
{
    //IplImage* img = cvLoadImage("./MyPic.jpg");
    //Mat mat = cvarrToMat(img);

    cv::Mat mat = cv::imread("MyPic.jpg", CV_LOAD_IMAGE_COLOR);

    int intensityToChange = 40;
    //invert and display the inverted image
    cv::Mat matH;
    mat.convertTo ( matH, -1, 2, 0 );

    cv::Mat matL;
    mat.convertTo ( matL, -1, 0.5, 0 );

    cv::namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("High Contrast", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Low Contrast", CV_WINDOW_AUTOSIZE);

    cv::imshow("Original Image", mat);
    cv::imshow("High Contrast", matH);
    cv::imshow("Low Contrast", matL);

    cv::waitKey(0);

    cv::destroyAllWindows(); //destroy all open windows

    //cvReleaseImage(&img);

    return 0;
}

void TestCSDN()
{
    //Mat img_curr_gray, img_prev_gray, fgmask;
    //img_curr_gray = imread("1.jpg", 1);
    //if ( img_curr_gray.empty() )
    //    return;
    //img_prev_gray = imread("2.jpg", 1);
    //Ptr<BackgroundSubtractorMOG2> bg_model = createBackgroundSubtractorMOG2();
    //bg_model->apply(img_prev_gray, fgmask, 0.005);
    //bg_model->apply(img_curr_gray, fgmask, 0.005);
    uchar buf[1000];
    CvSize size ( 100, 100 );
    //cvMat *mat = cvCreateMat ( 100, 100, CV_16UC1/*IPL_DEPTH_16U*/ );
    //mat->data = ( uchar* )buf;
    cv::Mat mat( size, CV_16UC1, cv::Scalar(0,0,0));
    mat.data = (uchar*)buf;
    //IplImage *image = cvCreateMat ( 100, 100, CV_16UC1/*IPL_DEPTH_16U*/ );
}

int TestSaveImage()
{
    cv::Mat img(650, 600, CV_16UC3, cv::Scalar ( 0, 50000, 50000 ) );
    CvMat cvmat(img);
    CvMat *pcvMat = &cvmat;

    if (img.empty())  {
        cout << "Error: Image can not be loaded ...!";
        return -1;
    }
    vector<int> compression_params;
    compression_params.push_back ( CV_IMWRITE_JPEG_QUALITY );
    compression_params.push_back ( 98 );
    bool bSuccess = imwrite ("TestImage.png", img, compression_params );

    cv::Mat matGray;
    img.convertTo ( matGray, CV_8U );
    cv::Mat result;
    cv::Canny( matGray, result, 50, 150);

    bSuccess = imwrite ("CannyResult.png", result, compression_params );

    if( ! bSuccess )    {
        cout << "Error : failed to save image !" << endl;
        return -1;
    }

    cv::namedWindow ( "MyWindow", CV_WINDOW_AUTOSIZE );
    imshow ( "MyWindow", img );
    cv::waitKey ( 0 );

    cv::destroyWindow ( "MyWindows");
    return 0;
}

int TestCanny()
{
    //cv::Mat mat = cv::imread(".\\data\\ShapeTemplate.png");
    cv::Mat mat = cv::imread(".\\data\\PCB1.jpg");
    if ( mat.empty() )
        return -1;
    
    cv::Mat matGray;
    mat.convertTo ( matGray, CV_8UC1 );

    cv::Mat matKernal = Mat::ones( cv::Size(5,5), CV_32FC1 ) / 25;
    cv::filter2D(matGray, matGray, matGray.depth(), matKernal );
    imshow("filter2D result", matGray);
    waitKey(0);

    cv::Mat matThresholdResult;
    cv::threshold( matGray, matThresholdResult, 60, 255, THRESH_BINARY );
    imshow("threshold result", matThresholdResult);
    waitKey(0);

    cv::Mat result;
    Canny( matGray, result, 50, 150);

    imshow("Canny Result", result);
    waitKey(0);
    //bool bSuccess = cv::imwrite (".\\data\\TemplateCannyResult.png", result );

    return 0;
}

int TestFindContours()
{
    char * src_filename = ".\\data\\TemplateCannyResult.png";
    //读入原图
    IplImage *SrcImage = cvLoadImage(src_filename, CV_LOAD_IMAGE_ANYDEPTH);
    //显示原图
    cvNamedWindow("src", CV_WINDOW_AUTOSIZE);
    cvShowImage("src", SrcImage);
 
    //二值化
    cvThreshold(SrcImage, SrcImage, 200, 255, THRESH_BINARY_INV);
    CvMemStorage *src_storage = cvCreateMemStorage(0);  //开辟内存空间 
    CvSeq*      src_contour = NULL;
    cvFindContours(SrcImage, src_storage, &src_contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);//这函数可选参数还有不少  
     
     
    //显示轮廓
    cvNamedWindow("src_img", CV_WINDOW_AUTOSIZE);
    cvShowImage("src_img", SrcImage);
    cvWaitKey(0);
    cvReleaseImage(&SrcImage);
    cvDestroyWindow("src_img");
    cvDestroyWindow("src");
    return 0;
}

int TestHassianCorner()
{
    Mat mat = imread ("BlackWhiteChecker.png");
    if ( mat.empty() )
        return -1;

    Mat gray;
    mat.convertTo ( gray, CV_8U );
    //cvCvtColor ( &mat, &gray, CV_BGR2GRAY );

    Mat response;
    cvCornerHarris ( &gray, &response, 2, 3, 0.04 );

    imshow("Response", response );

    waitKey ( 0 );

    return 0;
}

int TestHassianCorner1()
{
    IplImage *img = cvLoadImage ("BlackWhiteChecker.png");
    if ( NULL == img )
        return -1;

    cvShowImage("Original", img );

    IplImage *pGray;
    pGray = cvCreateImage ( cvGetSize ( img ), 8, 1 );
    cvCvtColor ( img, pGray, CV_BGR2GRAY );

    cvShowImage("Gray", pGray );

    IplImage *pResult;
    pResult = cvCreateImage ( cvGetSize ( img ), 32, 1 );
    cvCornerHarris ( pGray, pResult, 3, 5, 0.07 );    

    cvDilate ( pResult, pResult );

    cvShowImage("Result", pResult );

    waitKey ( 0 );

    return 0;
}

int TestDisplayVideo()
{
    VideoCapture cap(0); // open the video camera no. 0
    //cap.open()

    if (!cap.isOpened())  // if not success, exit program
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }

    double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
    double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

    cout << "Frame size : " << dWidth << " x " << dHeight << endl;

    bool bResult = cap.set(CV_CAP_PROP_FPS, 10);
    namedWindow("MyVideo"); //create a window called "MyVideo"
    
    while (1)
    {
        Mat frame;

        bool bSuccess = cap.read(frame); // read a new frame from video

        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }

        rectangle(frame, cv::Rect(100, 0, 400, frame.rows - 1), cv::Scalar(0, 0, 255));

        imshow("MyVideo", frame); //show the frame in "MyVideo" window

        cvResizeWindow ("MyVideo", 320, 160 );

        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl;
            break;
        }
    }
    destroyWindow("MyVideo");
    return 0;
}

int TestMat()
{
    Mat A = Mat::eye(10, 10, CV_32S);
    // extracts A columns, 1 (inclusive) to 3 (exclusive).
    Mat B = A(Range::all(), Range(1, 3));
    // extracts B rows, 5 (inclusive) to 9 (exclusive).
    // that is, C \~ A(Range(5, 9), Range(1, 3))
    Mat C = B(Range(5, 9), Range::all());
    Size size; Point ofs;
    C.locateROI(size, ofs);

    //cv::filter2D()
    Mat D(A,Rect(0,0,5,5));
    // size will be (width=10,height=10) and the ofs will be (x=1, y=5)
    return 0;
}

void TestIngensityDistribute()
{
     Mat mat = imread("./MyPic.jpg");

     if ( mat.empty() )
        return;

    Mat gray;
    mat.convertTo ( gray, CV_8U );

    //interpolate<unsigned short>(mat, 0, 0 );
    vector<float> vecOutput;
    GetIntensityOnLine<unsigned char> ( mat, Point ( 0, 0 ), Point ( mat.cols - 1 , mat.rows - 1), vecOutput );

    for ( vector<float>::iterator it = vecOutput.begin(); it != vecOutput.end(); ++ it )
    {
        cout << (float)*it << endl;
    }
}

const int nImageCount = 4;
const int nTotalCols = 16;
const int nTotalRows = 6;
const int nOverlapX = 104;
const int nOverlapY = 173;
const int nImageSize = 2032;
const int nImageInCol = 61;

template<typename T>
double MatSquareSum(const cv::Mat &mat)
{
    double dSum = 0;
    for ( int row = 0; row < mat.rows; ++ row )
    for ( int col = 0; col < mat.cols; ++ col)
    {
        float fValue = mat.at<T>(row, col);
        dSum += fValue * fValue;
    }
    return dSum;
}

int RefineSrchTemplate(const cv::Mat &mat, cv::Mat &matTmpl, cv::Point2f &ptResult)
{
    const float fRefineAccuracy = 0.1f;
    const int nRefineMaxStep = 20;

    cv::Mat matFloat, matTmplFloat;
    mat.convertTo(matFloat, CV_32FC1);
    matTmpl.convertTo(matTmplFloat, CV_32FC1 );

    cv::Point2f ptStartPoint = cv::Point2f(ptResult.x - 1 + fRefineAccuracy, ptResult.y - 1 + fRefineAccuracy);
    bool bFoundResult = false;
    cv::Mat matSubPixel_1, matSubPixel_2, matSubPixel_3, matSubPixel_4;
    cv::Mat matDiff_1, matDiff_2, matDiff_3, matDiff_4;
    cv::Point2f ptRight, ptDown, ptRightDown;
    double dDiff_1, dDiff_2, dDiff_3, dDiff_4;
    int nIteratorCount = 0;
    do
    {
        ptRight     = cv::Point2f( ptStartPoint.x + fRefineAccuracy, ptStartPoint.y );
        ptDown      = cv::Point2f( ptStartPoint.x,                   ptStartPoint.y + fRefineAccuracy );
        ptRightDown = cv::Point2f( ptStartPoint.x + fRefineAccuracy, ptStartPoint.y + fRefineAccuracy );

        cv::getRectSubPix ( matFloat, matTmpl.size(), ptStartPoint, matSubPixel_1 );  matDiff_1 = matTmplFloat - matSubPixel_1;  dDiff_1 = MatSquareSum<float> ( matDiff_1 );
        cv::getRectSubPix ( matFloat, matTmpl.size(), ptRight,      matSubPixel_2 );  matDiff_2 = matTmplFloat - matSubPixel_2;  dDiff_2 = MatSquareSum<float> ( matDiff_2 );
        cv::getRectSubPix ( matFloat, matTmpl.size(), ptDown,       matSubPixel_3 );  matDiff_3 = matTmplFloat - matSubPixel_3;  dDiff_3 = MatSquareSum<float> ( matDiff_3 );
        cv::getRectSubPix ( matFloat, matTmpl.size(), ptRightDown,  matSubPixel_4 );  matDiff_4 = matTmplFloat - matSubPixel_4;  dDiff_4 = MatSquareSum<float> ( matDiff_4 );

        if ( dDiff_1 <= dDiff_2 && dDiff_1 <= dDiff_3 && dDiff_1 <= dDiff_4 )
        {
            bFoundResult = true;
            ptResult = ptStartPoint;
        }else if ( dDiff_2 < dDiff_1 && dDiff_2 < dDiff_3 && dDiff_2 < dDiff_4 )
            ptStartPoint = ptRight;
        else if ( dDiff_3 < dDiff_1 && dDiff_3 < dDiff_2 && dDiff_3 < dDiff_4 )
            ptStartPoint = ptDown;
        else if ( dDiff_4 < dDiff_1 && dDiff_4 < dDiff_2 && dDiff_4 < dDiff_3 )
            ptStartPoint = ptRightDown;
        ++ nIteratorCount;
    }while ( ! bFoundResult && nIteratorCount < nRefineMaxStep );

    return 0;
}

int RefineSrchTemplateExt(const cv::Mat &mat, cv::Mat &matTmpl, cv::Point2f &ptResult)
{
    const float fRefineAccuracy = 0.1f;
    const int nRefineMaxStep = 10;
    cv::Mat matFloat, matTmplFloat;
    mat.convertTo(matFloat, CV_32FC1);
    matTmpl.convertTo(matTmplFloat, CV_32FC1 );
    cv::Mat matResult;
    matResult.create(nRefineMaxStep, nRefineMaxStep, CV_32FC1);
    for ( int row = 0; row < nRefineMaxStep; ++ row)
    for ( int col = 0; col < nRefineMaxStep; ++ col)
    {
        cv::Mat matSubPixel, matDiff;
        cv::Point2f ptCenter( ptResult.x - 0.5f + ( col + 1 ) * fRefineAccuracy, ptResult.y - 0.5f + ( row + 1 ) * fRefineAccuracy );
        cv::getRectSubPix ( matFloat, matTmpl.size(), ptCenter, matSubPixel );
        matDiff = matTmplFloat - matSubPixel;
        float fSquareSum = MatSquareSum<float> ( matDiff );
        fSquareSum /=  matTmpl.rows * matTmpl.cols;
        matResult.at<float>(row, col) = fSquareSum;
    }
    double minVal; double maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(matResult, &minVal, &maxVal, &minLoc, &maxLoc);
    cv::Point2f ptCenter( ptResult.x - 0.5f + ( minLoc.x + 1 ) * fRefineAccuracy, ptResult.y - 0.5f + ( minLoc.y + 1 ) * fRefineAccuracy );
    ptResult = ptCenter;
    cout << "Refined center " << ptResult.x << " " << ptResult.y << endl;
    return 0;
}

int SrchTemplate(const cv::Mat &mat, cv::Mat &matTmpl, cv::Point2f &ptResult)
{
  cv::Mat img_display, matResult;
  const int match_method = CV_TM_SQDIFF;

  mat.copyTo( img_display );

  /// Create the result matrix
  int result_cols = mat.cols - matTmpl.cols + 1;
  int result_rows = mat.rows - matTmpl.rows + 1;

  matResult.create( result_rows, result_cols, CV_32FC1 );

  /// Do the Matching and Normalize
  cv::matchTemplate( mat, matTmpl, matResult, match_method );
  cv::normalize( matResult, matResult, 0, 1, NORM_MINMAX, -1, Mat() );

  /// Localizing the best match with minMaxLoc
  double minVal; double maxVal; Point minLoc; Point maxLoc;
  Point matchLoc;

  minMaxLoc( matResult, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

  /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
  if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    { matchLoc = minLoc; }
  else
    { matchLoc = maxLoc; }

  /// Show me what you got
  rectangle( img_display, matchLoc, Point( matchLoc.x + matTmpl.cols , matchLoc.y + matTmpl.rows ), Scalar(245), 1, 8, 0 );
  rectangle( matResult, matchLoc, Point( matchLoc.x + matTmpl.cols , matchLoc.y + matTmpl.rows ), Scalar::all(0), 2, 8, 0 );

  ptResult.x = (float)matchLoc.x + (float)matTmpl.cols / 2;
  ptResult.y = (float)matchLoc.y + (float)matTmpl.rows / 2;

  RefineSrchTemplateExt ( mat, matTmpl, ptResult );
  imwrite("./data/Alignment/SrchTmplResult.png", img_display );
  //char* image_window = "Source Image";
  //char* result_window = "Result window";
  //imshow(image_window, img_display);
  //imshow(result_window, matResult);
  //cv::waitKey(0);
  return 0;
}

void TestSearchFiducialMark_3()
{
    cv::Mat mat = cv::imread("./data/CircleFiducialMark.png", cv::IMREAD_GRAYSCALE);
    if ( mat.empty() )  {
        std::cout << "Read input image fail" << std::endl;
        return;
    }

    const int TMPL_IMAGE_SIZE = 90;
    const int FIDUCIAL_MARK_SIZE = 64;
    const float BIG_IMAGE_HEIGHT = 11327.f;
    const int FICUCIAL_START_POS = (TMPL_IMAGE_SIZE - FIDUCIAL_MARK_SIZE) / 2;
    cv::Mat matTmpl = cv::Mat::zeros(TMPL_IMAGE_SIZE, TMPL_IMAGE_SIZE, CV_8UC1);
    auto margin = (TMPL_IMAGE_SIZE - FIDUCIAL_MARK_SIZE) / 2;
    auto radius = FIDUCIAL_MARK_SIZE / 2;
    cv::circle(matTmpl, cv::Point(margin + radius, margin + radius), radius, cv::Scalar::all(256), -1);

    cv::Point2f ptResult;
    SrchTemplate ( mat, matTmpl, ptResult );
}

cv::Mat ReadBayerImgCvtToGray(const std::string& strImagePath)
{
    cv::Mat mat = cv::imread(strImagePath, IMREAD_GRAYSCALE );
    if ( mat.empty() )
        return mat;

    cv::Mat matColor, matGray;
    cv::cvtColor (mat, matColor, CV_BayerGR2BGR );
    cv::cvtColor ( matColor, matGray, CV_BGR2GRAY );
    return matGray;
}

int DoAlignment()
{
    const int TMPL_IMAGE_SIZE = 80;
    const int FIDUCIAL_MARK_SIZE = 54;
    const float BIG_IMAGE_HEIGHT = 11327.f;
    const int FICUCIAL_START_POS = (TMPL_IMAGE_SIZE - FIDUCIAL_MARK_SIZE) / 2;
    cv::Mat matTmpl = cv::Mat::zeros(TMPL_IMAGE_SIZE, TMPL_IMAGE_SIZE, CV_8UC1);
    cv::rectangle(matTmpl,
        cv::Rect(FICUCIAL_START_POS, FICUCIAL_START_POS, FIDUCIAL_MARK_SIZE, FIDUCIAL_MARK_SIZE),
        cv::Scalar::all(256),
        -1);
    cv::imshow("Template", matTmpl);
    cv::waitKey(0);

    cv::Mat mat;
    cv::Point2f ptResult1, ptResult2, ptResult3;
    mat = ReadBayerImgCvtToGray ( "./data/Alignment1/F6-313-1.bmp"); //original F1-8-1.bmp
    if(mat.empty())
        return -1;
    
    SrchTemplate ( mat, matTmpl, ptResult1 );

    mat = ReadBayerImgCvtToGray("./data/Alignment1/F6-357-1.bmp");
    if(mat.empty())
        return -1;
    SrchTemplate ( mat, matTmpl, ptResult2 );

    mat = ReadBayerImgCvtToGray("./data/Alignment1/F1-8-1.bmp"); //Original F6-313-1.bmp
    if(mat.empty())
        return -1;
    SrchTemplate ( mat, matTmpl, ptResult3 );

    Point2f ptSrcArray[3] = {{4750.f, 3310.f }, {-18050.f, 3310.f}, {4750.f, -5270.f}};
    Point2f ptDstArray[3];
    //ptDstArray[0].x = 13 * ( nImageSize - nOverlapX ) + ptResult1.x;
    //ptDstArray[0].y =  0 * ( nImageSize - nOverlapY ) + ptResult1.y;

    //ptDstArray[1].x =  2 * ( nImageSize - nOverlapX ) + ptResult2.x;
    //ptDstArray[1].y =  5 * ( nImageSize - nOverlapY ) + ptResult2.y;

    //ptDstArray[2].x = 13 * ( nImageSize - nOverlapX ) + ptResult3.x;
    //ptDstArray[2].y = 5  * ( nImageSize - nOverlapY ) + ptResult3.y;

    ptDstArray[0].x = 13 * ( nImageSize - nOverlapX ) + ptResult1.x;
    ptDstArray[0].y =  5 * ( nImageSize - nOverlapY ) + ptResult1.y;

    ptDstArray[1].x =  2 * ( nImageSize - nOverlapX ) + ptResult2.x;
    ptDstArray[1].y =  5 * ( nImageSize - nOverlapY ) + ptResult2.y;

    ptDstArray[2].x = 13 * ( nImageSize - nOverlapX ) + ptResult3.x;
    ptDstArray[2].y =  0 * ( nImageSize - nOverlapY ) + ptResult3.y;

    CStopWatch stopWatch;
    cv::Mat matResult = cv::getAffineTransform(ptSrcArray, ptDstArray);
    int nNs = stopWatch.NowInMicro();
    std::cout << "getAffineTransform takes: " << nNs << " ns" << std::endl;
    printfMat<double>(matResult);

    return 0;
}

void TestFindLines()
{
    cv::Mat mat = cv::imread("./data/FindLineCase_1.png", cv::IMREAD_COLOR);
    if ( mat.empty() )
        return;

    cv::Mat matGray, matCanny;
    cv::cvtColor(mat, matGray, CV_BGR2GRAY);
    vector<cv::Vec4i> lines;

    cv::imshow("Gray image", matGray);
    //cv::waitKey(0);

    //cv::Canny( matGray, matCanny, 50, 200, 3 );
    //cv::imshow("Canny image", matCanny);
    //cv::waitKey(0);

    int nThreshold = 200;
    int nMinLength = 100;
    HoughLinesP(matGray, lines, 1, CV_PI / 180, nThreshold, nMinLength, 20);

    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Point2f pt1(( float ) lines [ i ] [ 0 ], ( float ) lines [ i ] [ 1 ]);
        cv::Point2f pt2(( float ) lines [ i ] [ 2 ], ( float ) lines [ i ] [ 3 ]);
        cv::line(mat, pt1, pt2, cv::Scalar(0, 0, 255), 3, 8);
    }
    cv::imshow("Hough Result", mat);
    cv::waitKey(0);
}

void TestFindCirclesOld()
{
	cv::Mat mat = cv::imread("./data/TestHoughLines.png", cv::IMREAD_COLOR);
	if ( mat.empty() )
	    return;
	cv::Mat matGray, matCanny;
	cv::cvtColor(mat, matGray, CV_BGR2GRAY);

	cv::GaussianBlur(matGray, matGray, cv::Size(5, 5), 1);

	cv::imshow("Gray image", matGray);    
	cv::waitKey(0);

	cv::Canny( matGray, matCanny, 200, 100, 3 );
	cv::imshow("Canny image", matCanny);
	//cv::waitKey(0);
	int nThreshold = 50;
	int nMinLength = 50;
	vector<Vec3f> circles;
	int minDist = 2;
	HoughCircles(matGray, circles, HOUGH_GRADIENT, 2, minDist, 200, 100);
	for ( size_t i = 0; i < circles.size(); ++ i )
	{    
	    Point center(cvRound(circles [ i ] [ 0 ]), cvRound(circles [ i ] [ 1 ]));    
	    int radius = cvRound(circles [ i ] [ 2 ]);
	    
		// draw the circle center    
	    circle(mat, center, 3, Scalar(0, 255, 0), -1, 8, 0);   
	    
		//draw the circle outline    
	    circle(mat, center, radius, Scalar(0, 0, 255), 1, 8, 0);    
	}    
	cv::imshow("Hough Result", mat);    
	cv::waitKey(0);
}

void ThinSubiteration1(CvMat *pSrc, CvMat *pDst) {
	int rows = pSrc->rows;
	int cols = pSrc->cols;
	cvCopy(pSrc, pDst);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (CV_MAT_ELEM(*pSrc, float, i, j) == 1.0f) {
				/// get 8 neighbors 
				/// calculate C(p) 
				int neighbor0 = (int)CV_MAT_ELEM(*pSrc, float, i - 1, j - 1);
				int neighbor1 = (int)CV_MAT_ELEM(*pSrc, float, i - 1, j);
				int neighbor2 = (int)CV_MAT_ELEM(*pSrc, float, i - 1, j + 1);
				int neighbor3 = (int)CV_MAT_ELEM(*pSrc, float, i, j + 1);
				int neighbor4 = (int)CV_MAT_ELEM(*pSrc, float, i + 1, j + 1);
				int neighbor5 = (int)CV_MAT_ELEM(*pSrc, float, i + 1, j);
				int neighbor6 = (int)CV_MAT_ELEM(*pSrc, float, i + 1, j - 1);
				int neighbor7 = (int)CV_MAT_ELEM(*pSrc, float, i, j - 1);
				int C = int(~neighbor1 & (neighbor2 | neighbor3)) +
					int(~neighbor3 & (neighbor4 | neighbor5)) +
					int(~neighbor5 & (neighbor6 | neighbor7)) +
					int(~neighbor7 & (neighbor0 | neighbor1));
				if (C == 1) {
					/// calculate N 
					int N1 = int(neighbor0 | neighbor1) +
						int(neighbor2 | neighbor3) +
						int(neighbor4 | neighbor5) +
						int(neighbor6 | neighbor7);
					int N2 = int(neighbor1 | neighbor2) +
						int(neighbor3 | neighbor4) +
						int(neighbor5 | neighbor6) +
						int(neighbor7 | neighbor0);
					int N = min(N1, N2);
					if ((N == 2) || (N == 3)) {
						/// calculate criteria 3 
						int c3 = (neighbor1 | neighbor2 | ~neighbor4) & neighbor3;
						if (c3 == 0) {
							CV_MAT_ELEM(*pDst, float, i, j) = 0.0f;
						}
					}
				}
			}
		}
	}
}

void ThinSubiteration2(CvMat *pSrc, CvMat *pDst) {
	int rows = pSrc->rows;
	int cols = pSrc->cols;
	cvCopy(pSrc, pDst);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (CV_MAT_ELEM(*pSrc, float, i, j) == 1.0f) {
				/// get 8 neighbors 
				/// calculate C(p) 
				int neighbor0 = (int)CV_MAT_ELEM(*pSrc, float, i - 1, j - 1);
				int neighbor1 = (int)CV_MAT_ELEM(*pSrc, float, i - 1, j);
				int neighbor2 = (int)CV_MAT_ELEM(*pSrc, float, i - 1, j + 1);
				int neighbor3 = (int)CV_MAT_ELEM(*pSrc, float, i, j + 1);
				int neighbor4 = (int)CV_MAT_ELEM(*pSrc, float, i + 1, j + 1);
				int neighbor5 = (int)CV_MAT_ELEM(*pSrc, float, i + 1, j);
				int neighbor6 = (int)CV_MAT_ELEM(*pSrc, float, i + 1, j - 1);
				int neighbor7 = (int)CV_MAT_ELEM(*pSrc, float, i, j - 1);
				int C = int(~neighbor1 & (neighbor2 | neighbor3)) +
					int(~neighbor3 & (neighbor4 | neighbor5)) +
					int(~neighbor5 & (neighbor6 | neighbor7)) +
					int(~neighbor7 & (neighbor0 | neighbor1));
				if (C == 1) {
					/// calculate N 
					int N1 = int(neighbor0 | neighbor1) +
						int(neighbor2 | neighbor3) +
						int(neighbor4 | neighbor5) +
						int(neighbor6 | neighbor7);
					int N2 = int(neighbor1 | neighbor2) +
						int(neighbor3 | neighbor4) +
						int(neighbor5 | neighbor6) +
						int(neighbor7 | neighbor0);
					int N = min(N1, N2);
					if ((N == 2) || (N == 3)) {
						int E = (neighbor5 | neighbor6 | ~neighbor0) & neighbor7;
						if (E == 0) {
							CV_MAT_ELEM(*pDst, float, i, j) = 0.0f;
						}
					}
				}
			}
		}
	}
}

void MorphologicalThinning(CvMat *pSrc, CvMat *pDst) {
	bool bDone = false;
	int rows = pSrc->rows;
	int cols = pSrc->cols;
	/// pad source 
	CvMat *p_enlarged_src = cvCreateMat(rows + 2, cols + 2, CV_32FC1);
	for (int i = 0; i < (rows + 2); i++) {
		CV_MAT_ELEM(*p_enlarged_src, float, i, 0) = 0.0f;
		CV_MAT_ELEM(*p_enlarged_src, float, i, cols + 1) = 0.0f;
	}
	for (int j = 0; j < (cols + 2); j++) {
		CV_MAT_ELEM(*p_enlarged_src, float, 0, j) = 0.0f;
		CV_MAT_ELEM(*p_enlarged_src, float, rows + 1, j) = 0.0f;
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (CV_MAT_ELEM(*pSrc, float, i, j) >= 0.5f) {
				CV_MAT_ELEM(*p_enlarged_src, float, i + 1, j + 1) = 1.0f;
			}
			else
				CV_MAT_ELEM(*p_enlarged_src, float, i + 1, j + 1) = 0.0f;
		}
	}
	/// start to thin 
	CvMat *p_thinMat1 = cvCreateMat(rows + 2, cols + 2, CV_32FC1);
	CvMat *p_thinMat2 = cvCreateMat(rows + 2, cols + 2, CV_32FC1);
	CvMat *p_cmp = cvCreateMat(rows + 2, cols + 2, CV_8UC1);
	while (bDone != true) {
		/// sub-iteration 1 
		ThinSubiteration1(p_enlarged_src, p_thinMat1);
		/// sub-iteration 2 
		ThinSubiteration2(p_thinMat1, p_thinMat2);
		/// compare 
		cvCmp(p_enlarged_src, p_thinMat2, p_cmp, CV_CMP_EQ);
		/// check 
		int num_non_zero = cvCountNonZero(p_cmp);
		if (num_non_zero == (rows + 2) * (cols + 2)) {
			bDone = true;
		}
		/// copy 
		cvCopy(p_thinMat2, p_enlarged_src);
	}
	/// copy result 
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			CV_MAT_ELEM(*pDst, float, i, j) = CV_MAT_ELEM(*p_enlarged_src, float, i + 1, j + 1);
		}
	}
	/// clean memory 
	cvReleaseMat(&p_enlarged_src);
	cvReleaseMat(&p_thinMat1);
	cvReleaseMat(&p_thinMat2);
	cvReleaseMat(&p_cmp);
}

void TestFindCircles()
{
	cv::Mat mat = cv::imread("./data/TestHoughLines.png", cv::IMREAD_COLOR);
	if (mat.empty())
		return;
	cv::Mat matGray, matCanny;
	cv::cvtColor(mat, matGray, CV_BGR2GRAY);
	cv::imshow("Gray image", matGray);
	cv::waitKey(0);
	cv::GaussianBlur(matGray, matGray, cv::Size(5, 5), 1);

	std::unique_ptr<CvMat> pSrc = std::make_unique<CvMat>(matGray);
	std::unique_ptr<CvMat> pDst = std::make_unique<CvMat>(matGray.clone());
	MorphologicalThinning(pSrc.get(), pDst.get());
	cv::Mat matThin = cvarrToMat(pDst.get());

	cv::imshow("Thinning image", matThin);
	cv::waitKey(0);

	cv::Canny(matGray, matCanny, 200, 100, 3);
	cv::imshow("Canny image", matCanny);
	//cv::waitKey(0);
	int nThreshold = 50;
	int nMinLength = 50;
	vector<Vec3f> circles;
	int minDist = 2;
	HoughCircles(matGray, circles, HOUGH_GRADIENT, 2, minDist, 200, 100);
	for (size_t i = 0; i < circles.size(); ++i)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		// draw the circle center    
		circle(mat, center, 3, Scalar(0, 255, 0), -1, 8, 0);

		//draw the circle outline    
		circle(mat, center, radius, Scalar(0, 0, 255), 1, 8, 0);
	}
	cv::imshow("Hough Result", mat);
	cv::waitKey(0);
}

void TestPhaseImage()
{
    cv::Mat mat = cv::imread("./data/Obj_image5.bmp", cv::IMREAD_GRAYSCALE);
    if (mat.empty())
        return;

    cv::Mat matResult = cv::Mat::ones(256, mat.cols, CV_8UC1) * 255;
    const int ROW = 420;
    for (int col = 1; col < mat.cols; ++col)
    {
        int grayScale1 = mat.at<unsigned char>(ROW, col - 1);
        int grayScale2 = mat.at<unsigned char>(ROW, col);
        cv::Point pt1(col - 1, grayScale1);
        cv::Point pt2(col,     grayScale2);
        cv::line(matResult, pt1, pt2, cv::Scalar(0), 1);
    }
    cv::imshow("Gray Scale", matResult);
    cv::waitKey(0);
}

void Test_connectedComponents()
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<uchar> Input = { 
        1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 1, 0, 0,
        1, 1, 1, 0, 0, 1, 0, 0,
        1, 1, 1, 0, 0, 0, 1, 0,
        1, 1, 1, 0, 0, 0, 1, 0,
        1, 1, 1, 0, 0, 0, 1, 0,
        1, 1, 1, 0, 0, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0
    };
    cv::Mat matInput(Input);
    matInput = matInput.reshape(1, 8);
    std::cout << "Input: " << std::endl;
    printfMatInt<uchar>(matInput);

    findContours(matInput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    for (int i = 0; i < contours.size(); i++)
    {
        auto area = contourArea ( contours[i] );
        std::cout << "area = " << area << std::endl;
    }

    cv::Mat matLabels;
    connectedComponents(matInput, matLabels, 8, CV_32S);
    std::cout << "Labels: " << std::endl;
    printfMatInt<int>(matLabels);

    //cv::Mat matStats, matCenter;
    //connectedComponentsWithStats(matInput, matLabels, matStats, matCenter);
    //printfMatInt<int>(matLabels);
}

int _refineSrchTemplate(const cv::Mat &mat, cv::Mat &matTmpl, cv::Point2f &ptResult)
{
    cv::Mat matWarp = cv::Mat::eye(2, 3, CV_32FC1);
    matWarp.at<float>(0,2) = ptResult.x;
    matWarp.at<float>(1,2) = ptResult.y;
    printfMat<float>(matWarp);
    int number_of_iterations = 30;
    double termination_eps = 0.001;

    cv::findTransformECC ( matTmpl, mat, matWarp, MOTION_TRANSLATION, TermCriteria (TermCriteria::COUNT+TermCriteria::EPS,
        number_of_iterations, termination_eps));
    ptResult.x = matWarp.at<float>(0,2);
    ptResult.y = matWarp.at<float>(1,2);
    return 0;
}

float GuassianValue2D(float ssq, float x, float y )
{
    return exp( -(x*x + y*y) / ( 2.0 *ssq ) ) / ( 2.0 * CV_PI * ssq );
}

template<typename _tp>
void meshgrid ( float xStart, float xInterval, float xEnd, float yStart, float yInterval, float yEnd, cv::Mat &matX, cv::Mat &matY )
{
    std::vector<_tp> vectorX, vectorY;
    _tp xValue = xStart;
    while ( xValue <= xEnd )    {
        vectorX.push_back(xValue);
        xValue += xInterval;
    }

    _tp yValue = yStart;
    while ( yValue <= yEnd )    {
        vectorY.push_back(yValue);
        yValue += yInterval;
    }
    cv::Mat matCol ( vectorX );
    matCol = matCol.reshape ( 1, 1 );

    cv::Mat matRow ( vectorY );
    matRow = matRow.reshape ( 1, vectorY.size() );
    matX = cv::repeat ( matCol, vectorY.size(), 1 );
    matY = cv::repeat ( matRow, 1, vectorX.size() );
}

void TestMeshGrid()
{
    cv::Mat matX, matY;
    meshgrid<int> ( -2,1,2, -2,1,2, matX, matY);
    printfMatInt<int> ( matX );
    std::cout << endl;
    printfMatInt<int> ( matY );
}

void filter2D_Conv(InputArray src, OutputArray dst, int ddepth,
                   InputArray kernel, Point anchor = Point(-1,-1),
                   double delta = 0, int borderType = BORDER_DEFAULT )
{
    cv::Mat newKernel;
    const int FLIP_H_Z = -1;
    cv::flip ( kernel, newKernel, FLIP_H_Z );
    cv::Point newAnchor = anchor;
    if ( anchor.x > 0 && anchor.y >= 0 )
        newAnchor = cv::Point ( newKernel.cols - anchor.x - 1, newKernel.rows - anchor.y - 1 );
    cv::filter2D ( src, dst, ddepth, newKernel, newAnchor, delta, borderType );
}

int _refineWithLMIterationChenLN( const cv::Mat &mat, cv::Mat &matTmpl, cv::Point2f &ptResult )
{
    cv::Mat matGuassian;
    int width = 2;
    float ssq = 1.;
    matGuassian.create(width * 2 + 1, width * 2 + 1, CV_32FC1 );
    cv::Mat matI, matT;
    mat.convertTo ( matI, CV_32FC1);
    matTmpl.convertTo ( matT, CV_32FC1 );

    cv::Mat matX, matY;
    meshgrid<float> ( -width, 1, width, -width, 1, width, matX, matY );
    for ( int row = 0; row < matX.rows; ++ row )
    for ( int col = 0; col < matX.cols; ++ col )
    {
        matGuassian.at<float>(row, col) = GuassianValue2D( ssq, matX.at<float>(row, col), matY.at<float>(row, col) );
    }
    matGuassian = matGuassian.mul(-matX);
    cv::Mat matTmp( matGuassian, Range::all(), cv::Range(0,2));
    float fSum = cv::sum(matTmp)[0];
    cv::Mat matGuassianKernalY = matGuassian / fSum;        //XSG question, the kernel is reversed?
    cv::Mat matGuassianKernalX;
    cv::transpose( matGuassianKernalY, matGuassianKernalX );

    std::cout << "matGuassianKernalX" << std::endl;
    printfMat<float>(matGuassianKernalX);
    std::cout << "matGuassianKernalY" << std::endl;
    printfMat<float>(matGuassianKernalY);

    {
        cv::Mat matGuassianKernalFlipX;
        const int FLIP_H_Z = -1;
        cv::flip ( matGuassianKernalX, matGuassianKernalFlipX, FLIP_H_Z );
        std::cout << "matGuassianKernalFlipX" << std::endl;
        printfMat<float>(matGuassianKernalFlipX);
    }

    /**************** Using LM Iteration ****************/
    int N = 0, v = 2;
    cv::Mat matD;// = cv::Mat::create( );
    matD.create( 2,1, CV_32FC1 );
    matD.at<float>(0, 0) = ptResult.x;
    matD.at<float>(1, 0) = ptResult.y;

    cv::Mat matDr = matD.clone();

    cv::Mat matInputNew;

    //cv::Point ptSourceCtr(ptResult.x + matTmpl.cols / 2, ptResult.y + matTmpl.rows / 2);    
    //cv::getRectSubPix ( mat, matTmpl.size(), ptSourceCtr, matInputNew, CV_32F );
    //cv::Mat matTmp1( matInputNew, cv::Rect (0, 0, 10, 10 ) );
    //printfMat<float> ( matTmp1 );

    auto interp2 = [matI, matT](cv::Mat &matOutput, const cv::Mat &matD) {
        cv::Mat map_x, map_y;
        map_x.create(matT.size(), CV_32FC1);
        map_y.create(matT.size(), CV_32FC1);
        cv::Point2f ptStart(matD.at<float>(1, 0), matD.at<float>(0, 0) );
        for (int row = 0; row < matT.rows; ++ row )
        for (int col = 0; col < matT.cols; ++ col )
        {
            map_x.at<float>(row, col) = ptStart.x + col;
            map_y.at<float>(row, col) = ptStart.y + row;
        }
        cv::remap ( matI, matOutput, map_x, map_y, cv::INTER_LINEAR );

        {
            //cv::Mat matForPrint( matOutput, cv::Rect(0, 0, 10, 10));
            //printfMat<float>(matForPrint);
        }
    };

    interp2 ( matInputNew, matD );   
    
    cv::Mat matR = matT - matInputNew;
    cv::Mat matRn = matR.clone();
    float fRSum = cv::sum ( matR.mul ( matR ) )[0];
    float fRSumN = fRSum;

    cv::Mat matDerivativeX, matDerivativeY;
    filter2D_Conv ( matInputNew, matDerivativeX, CV_32F, matGuassianKernalX, cv::Point(-1, -1 ), 0.0, BORDER_REPLICATE );    
    filter2D_Conv ( matInputNew, matDerivativeY, CV_32F, matGuassianKernalY, cv::Point(-1, -1 ), 0.0, BORDER_REPLICATE );
    
    cv::Mat matRt = matR.reshape ( 1, 1 );
    cv::Mat matRtTranspose;
    cv::transpose ( matRt, matRtTranspose );
    matDerivativeX = matDerivativeX.reshape ( 1, 1 );
    matDerivativeY = matDerivativeY.reshape ( 1, 1 );

    const float* p = matDerivativeX.ptr<float>(0);
    std::vector<float> vecDerivativeX(p, p + matDerivativeX.cols);    

    cv::Mat matJacobianT, matJacobian;
    matJacobianT.push_back ( matDerivativeX );
    matJacobianT.push_back ( matDerivativeY );
    cv::transpose ( matJacobianT, matJacobian );

    cv::Mat matE = cv::Mat::eye(2, 2, CV_32FC1);

    cv::Mat A = matJacobianT * matJacobian;
    cv::Mat g = - matJacobianT * matRtTranspose;    

    double min, max;
    cv::minMaxLoc(A, &min, &max);
    float mu = 1.f * max;
    float err1 = 1e-4, err2 = 1e-4;
    auto Nmax = 100;
    while ( cv::norm ( matDr ) > err2 && N < Nmax ) {
        ++ N;
        cv::solve ( A + mu * matE, -g, matDr );     // equal to matlab matDr = (A+mu*E)\(-g);
        {
            std::cout << "A" << std::endl;
            printfMat<float>(A);

            std::cout << "g" << std::endl;
            printfMat<float>(g);

            std::cout << "matDr" << std::endl;
            printfMat<float>(matDr);
        }
        
        cv::Mat matDn = matD + matDr;
        if ( cv::norm ( matDr ) < err2 )    {            
            interp2 ( matInputNew, matDn );
            matRn = matT - matInputNew;
            fRSumN = cv::sum ( matR.mul ( matR ) )[0];
            matD = matDn;
            break;
        }else {
            if (matDn.at<float> ( 0, 0 ) > matI.cols - matT.cols ||
                matDn.at<float> ( 0, 0 ) < 0 ||
                matDn.at<float> ( 1, 0 ) > matI.rows - matT.rows ||
                matDn.at<float> ( 1, 0 ) < 0 )  {
                mu *= v;
                v *= 2;
            }else  {
                interp2 ( matInputNew, matDn );
                matRn = matT - matInputNew;
                fRSumN = cv::sum ( matRn.mul ( matRn ) )[0];

                cv::Mat matDrTranspose;
                cv::transpose ( matDr, matDrTranspose );
                cv::Mat matL = ( matDrTranspose * ( mu * matDr - g ) );   // L(0) - L(hlm) = 0.5 * h' ( uh - g)
                auto L = matL.at<float>(0, 0);
                auto F = fRSum - fRSumN;
                float rho = F / L;

                if ( rho > 0 )  {
                    matD = matDn.clone();
                    matR = matRn.clone();
                    fRSum = fRSumN;

                    filter2D_Conv ( matInputNew, matDerivativeX, CV_32F, matGuassianKernalX, cv::Point(-1, -1 ), 0.0, BORDER_REPLICATE );
                    filter2D_Conv ( matInputNew, matDerivativeY, CV_32F, matGuassianKernalY, cv::Point(-1, -1 ), 0.0, BORDER_REPLICATE );
                    matRt = matR.reshape(1, 1);
                    cv::transpose ( matRt, matRtTranspose );

                    matDerivativeX = matDerivativeX.reshape(1, 1);
                    matDerivativeY = matDerivativeY.reshape(1, 1);

                    matJacobianT.release();
                    matJacobianT.push_back(matDerivativeX);
                    matJacobianT.push_back(matDerivativeY);
                    cv::transpose(matJacobianT, matJacobian);

                    A = matJacobianT * matJacobian;
                    g = - matJacobianT * matRtTranspose;

                    mu *= max ( 1.f/3.f, 1 - pow ( 2 * rho-1, 3 ) );
                }else {
                    mu *= v; v *= 2;
                }
            }
        }
    }

    ptResult.x = matD.at<float>(1, 0);
    ptResult.y = matD.at<float>(0, 0);
    return 0;
}

int _refineWithLMIteration( const cv::Mat &mat, cv::Mat &matTmpl, cv::Point2f &ptResult )
{
    cv::Mat matGuassian;
    int width = 2;
    float ssq = 1.;
    matGuassian.create(width * 2 + 1, width * 2 + 1, CV_32FC1 );
    cv::Mat matI, matT;
    mat.convertTo ( matI, CV_32FC1);
    matTmpl.convertTo ( matT, CV_32FC1 );

    cv::Mat matX, matY;
    meshgrid<float> ( -width, 1, width, -width, 1, width, matX, matY );
    for ( int row = 0; row < matX.rows; ++ row )
    for ( int col = 0; col < matX.cols; ++ col )
    {
        matGuassian.at<float>(row, col) = GuassianValue2D( ssq, matX.at<float>(row, col), matY.at<float>(row, col) );
    }
    matGuassian = matGuassian.mul(-matX);
    cv::Mat matTmp( matGuassian, Range::all(), cv::Range(0,2));
    float fSum = cv::sum(matTmp)[0];
    cv::Mat matGuassianKernalX, matGuassianKernalY;
    matGuassianKernalX = matGuassian / fSum;        //XSG question, the kernel is reversed?
    cv::transpose( matGuassianKernalX, matGuassianKernalY );

    /**************** Using LM Iteration ****************/
    int N = 0, v = 2;
    cv::Mat matD;
    matD.create( 2,1, CV_32FC1 );
    matD.at<float>(0, 0) = ptResult.x;
    matD.at<float>(1, 0) = ptResult.y;

    cv::Mat matDr = matD.clone();

    cv::Mat matInputNew;

    auto interp2 = [matI, matT](cv::Mat &matOutput, const cv::Mat &matD) {
        cv::Mat map_x, map_y;
        map_x.create(matT.size(), CV_32FC1);
        map_y.create(matT.size(), CV_32FC1);
        cv::Point2f ptStart(matD.at<float>(0, 0), matD.at<float>(1, 0) );
        for (int row = 0; row < matT.rows; ++ row )
        for (int col = 0; col < matT.cols; ++ col )
        {
            map_x.at<float>(row, col) = ptStart.x + col;
            map_y.at<float>(row, col) = ptStart.y + row;
        }
        cv::remap ( matI, matOutput, map_x, map_y, cv::INTER_LINEAR );
    };

    interp2 ( matInputNew, matD );   
    
    cv::Mat matR = matT - matInputNew;
    cv::Mat matRn = matR.clone();
    float fRSum = cv::sum ( matR.mul ( matR ) )[0];
    float fRSumN = fRSum;

    cv::Mat matDerivativeX, matDerivativeY;
    filter2D_Conv ( matInputNew, matDerivativeX, CV_32F, matGuassianKernalX, cv::Point(-1, -1 ), 0.0, BORDER_REPLICATE );    
    filter2D_Conv ( matInputNew, matDerivativeY, CV_32F, matGuassianKernalY, cv::Point(-1, -1 ), 0.0, BORDER_REPLICATE );
    
    cv::Mat matRt = matR.reshape ( 1, 1 );
    cv::Mat matRtTranspose;
    cv::transpose ( matRt, matRtTranspose );
    matDerivativeX = matDerivativeX.reshape ( 1, 1 );
    matDerivativeY = matDerivativeY.reshape ( 1, 1 );

    const float* p = matDerivativeX.ptr<float>(0);
    std::vector<float> vecDerivativeX(p, p + matDerivativeX.cols);

    cv::Mat matJacobianT, matJacobian;
    matJacobianT.push_back ( matDerivativeX );
    matJacobianT.push_back ( matDerivativeY );
    cv::transpose ( matJacobianT, matJacobian );

    cv::Mat matE = cv::Mat::eye(2, 2, CV_32FC1);

    cv::Mat A = matJacobianT * matJacobian;
    cv::Mat g = - matJacobianT * matRtTranspose;    

    double min, max;
    cv::minMaxLoc(A, &min, &max);
    float mu = 1.f * max;
    float err1 = 1e-4, err2 = 1e-4;
    auto Nmax = 100;
    while ( cv::norm ( matDr ) > err2 && N < Nmax ) {
        ++ N;
        cv::solve ( A + mu * matE, -g, matDr );     // equal to matlab matDr = (A+mu*E)\(-g);

        cv::Mat matDn = matD + matDr;
        if ( cv::norm ( matDr ) < err2 )    {            
            interp2 ( matInputNew, matDn );
            matRn = matT - matInputNew;
            fRSumN = cv::sum ( matR.mul ( matR ) )[0];
            matD = matDn;
            break;
        }else {
            if (matDn.at<float> ( 0, 0 ) > matI.cols - matT.cols ||
                matDn.at<float> ( 0, 0 ) < 0 ||
                matDn.at<float> ( 1, 0 ) > matI.rows - matT.rows ||
                matDn.at<float> ( 1, 0 ) < 0 )  {
                mu *= v;
                v *= 2;
            }else  {
                interp2 ( matInputNew, matDn );
                matRn = matT - matInputNew;
                fRSumN = cv::sum ( matRn.mul ( matRn ) )[0];

                cv::Mat matDrTranspose;
                cv::transpose ( matDr, matDrTranspose );
                cv::Mat matL = ( matDrTranspose * ( mu * matDr - g ) );   // L(0) - L(hlm) = 0.5 * h' ( uh - g)
                auto L = matL.at<float>(0, 0);
                auto F = fRSum - fRSumN;
                float rho = F / L;

                if ( rho > 0 )  {
                    matD = matDn.clone();
                    matR = matRn.clone();
                    fRSum = fRSumN;

                    filter2D_Conv ( matInputNew, matDerivativeX, CV_32F, matGuassianKernalX, cv::Point(-1, -1 ), 0.0, BORDER_REPLICATE );
                    filter2D_Conv ( matInputNew, matDerivativeY, CV_32F, matGuassianKernalY, cv::Point(-1, -1 ), 0.0, BORDER_REPLICATE );
                    matRt = matR.reshape(1, 1);
                    cv::transpose ( matRt, matRtTranspose );

                    matDerivativeX = matDerivativeX.reshape(1, 1);
                    matDerivativeY = matDerivativeY.reshape(1, 1);

                    matJacobianT.release();
                    matJacobianT.push_back(matDerivativeX);
                    matJacobianT.push_back(matDerivativeY);
                    cv::transpose(matJacobianT, matJacobian);

                    A = matJacobianT * matJacobian;
                    g = - matJacobianT * matRtTranspose;

                    mu *= max ( 1.f/3.f, 1 - pow ( 2 * rho-1, 3 ) );
                }else {
                    mu *= v; v *= 2;
                }
            }
        }
    }

    ptResult.x = matD.at<float>(0, 0);
    ptResult.y = matD.at<float>(1, 0);
    return 0;
}

int matchTemplate(const cv::Mat &mat, cv::Mat &matTmpl, cv::Point2f &ptResult)
{
    cv::Mat img_display, matResult;
    const int match_method = CV_TM_SQDIFF;

    mat.copyTo(img_display);

    /// Create the result matrix
    int result_cols = mat.cols - matTmpl.cols + 1;
    int result_rows = mat.rows - matTmpl.rows + 1;

    matResult.create(result_rows, result_cols, CV_32FC1);

    /// Do the Matching and Normalize
    cv::matchTemplate(mat, matTmpl, matResult, match_method);
    cv::normalize ( matResult, matResult, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    /// Localizing the best match with minMaxLoc
    double minVal; double maxVal;
    cv::Point minLoc, maxLoc, matchLoc;

    cv::minMaxLoc(matResult, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
        matchLoc = minLoc;
    else
        matchLoc = maxLoc;

    ptResult.x = (float)matchLoc.x;
    ptResult.y = (float)matchLoc.y;
    _refineWithLMIteration(mat, matTmpl, ptResult);

    ptResult.x += (float)( matTmpl.cols / 2 + 0.5 );
    ptResult.y += (float)( matTmpl.rows / 2 + 0.5 );
    return 0;
}

void TestTemplateFineTune()
{
    cv::Mat matInput = cv::imread("./data/ChessBoard.png", cv::IMREAD_GRAYSCALE);
    if ( matInput.empty())
        return;
    cv::Rect rectSrchROI(171, 1, 80, 80);
    cv::Mat matSrchROI(matInput, rectSrchROI);

    cv::Mat matTmpl = cv::Mat::zeros ( 20, 20, CV_8UC1);
    cv::Mat matTmplROI(matTmpl,cv::Rect(0,0,10,10));
    matTmplROI.setTo(cv::Scalar::all(255));
    cv::Mat matTmplROI1(matTmpl,cv::Rect(10,10,10,10));
    matTmplROI1.setTo(cv::Scalar::all(255));

    cv::imwrite("./data/ChessBoardTmpl.png", matTmpl );

    cv::Point2f ptResult;
    matchTemplate ( matSrchROI, matTmpl, ptResult);
    
    ptResult.x += rectSrchROI.x;
    ptResult.y += rectSrchROI.y;
    printf( "matchTemplate result ( %f, %f ) \n", ptResult.x, ptResult.y );
}

cv::Size findChessBoardBlockSize( const cv::Mat &matInput ) {
    cv::Size size;
    cv::Mat matROI ( matInput, cv::Rect ( matInput.cols / 2 - 100,  matInput.rows / 2 - 100, 200, 200 ) );  //Use the ROI 200x200 in the center of the image.
    cv::Mat matFilter;
    cv::blur ( matROI, matFilter, cv::Size(5, 5) );
    cv::Mat matThreshold;
    cv::threshold ( matFilter, matThreshold, 150, 255, cv::THRESH_BINARY_INV );
    cv::imshow("Threshold image", matThreshold);
    cv::waitKey(0);

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours ( matThreshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE );
    for ( const auto &contour : contours )  {
        auto area = cv::contourArea ( contour );
        if ( area > 1000 )  {
            cv::RotatedRect rotatedRect = cv::minAreaRect ( contour );
            size = rotatedRect.size; 
            break;
        }
    }
    if ( size.width != size.height )
        size.height = size.width = max ( size.width, size.height );    
    return size;
}

void Test_findChessBoardBlockSize()
{
    cv::Mat mat = cv::imread ("./data/ChessBoard.png", cv::IMREAD_GRAYSCALE );
    if ( mat.empty() )
        return;

    cv::Size size = findChessBoardBlockSize ( mat );
}

cv::Point findFirstCorner(const cv::Mat &matInput, int nBlockSize) {
    cv::Mat matROI ( matInput, cv::Rect ( 0, 0, 300, 300 ) );
    cv::Mat matFilter;
    cv::blur ( matROI, matFilter, cv::Size(5, 5) );
    cv::Mat matThreshold;
    cv::threshold ( matFilter, matThreshold, 100, 255, cv::THRESH_BINARY_INV );
    
    cv::imwrite("./data/UpperleftCornerThreshold.png", matThreshold);

    cv::Point ptBlackCenter;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours ( matThreshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE );
    std::vector<std::vector<cv::Point> > vecDrawContours;
    std::vector<cv::Point2f> vecPoints;
    for ( const auto &contour : contours )  {
        auto area = cv::contourArea ( contour );
        if ( area > 1000 )  {
            cv::RotatedRect rotatedRect = cv::minAreaRect ( contour );
            if ( fabs ( rotatedRect.size.width - rotatedRect.size.height ) / ( rotatedRect.size.width + rotatedRect.size.height ) < 0.1 )   {
                cv::Moments moment = cv::moments( contour );
                ptBlackCenter.x = static_cast< float >(moment.m10 / moment.m00);
                ptBlackCenter.y = static_cast< float >(moment.m01 / moment.m00);
                vecDrawContours.push_back ( contour );
                vecPoints.push_back ( rotatedRect.center );
            }
        }
    }

    int minDistanceToZero = 10000000;
    cv::Point ptUpperLeftBlockCenter;
    for ( const auto point : vecPoints )    {
        auto distanceToZero = point.x * point.x + point.y * point.y;
        if ( distanceToZero < minDistanceToZero )   {
            ptUpperLeftBlockCenter = point;
            minDistanceToZero = distanceToZero;
        }          
    }

    cv::Point ptInitialCenter( ptUpperLeftBlockCenter.x - nBlockSize / 2, ptUpperLeftBlockCenter.y - nBlockSize / 2 );

    cv::Mat matOutput;
    cv::cvtColor ( matROI, matOutput, CV_GRAY2BGR );
    for ( int i = 0; i < vecDrawContours.size(); ++ i )    {
        cv::drawContours ( matOutput, vecDrawContours, i, cv::Scalar(0, 0, 255) );
    }
    cv::imwrite("./data/UpperLeftCornerDraw.png", matOutput);
    return ptBlackCenter;
}

void Test_findFirstCorner()
{
    cv::Mat mat = cv::imread ("./data/ChessBoard.png", cv::IMREAD_GRAYSCALE );
    if ( mat.empty() )
        return;

    cv::Size size = findFirstCorner ( mat, 42 );
}

double CalcSimilarity(const cv::Mat &mat1, const cv::Mat &mat2) {
    //double errorL2 = norm( mat1, mat2, CV_L2 );
    // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
    //double similarity = 1 - errorL2 / (double)( mat1.rows * mat1.cols );
    int nThreshold1 = autoThreshold ( mat1 );
    int nThreshold2 = autoThreshold ( mat2 );
    cv::Mat matThreshold1, matThreshold2;
    cv::threshold ( mat1, matThreshold1, 195, 255, cv::THRESH_BINARY );
    cv::threshold ( mat2, matThreshold2, nThreshold2, 255, cv::THRESH_BINARY );

    //cv::imshow("matThreshold1", matThreshold1 );
    //cv::imshow("matThreshold2", matThreshold2 );
    //cv::waitKey(0);

    cv::Mat matDiff = matThreshold1 - matThreshold2;
    double dDiffPoint = cv::countNonZero ( matDiff );
    return 1. - dDiffPoint / mat1.rows * mat1.cols;
}

void Test_CalcSimilarity() {
    cv::Mat mat1 = cv::imread("./data/Compare.png", cv::IMREAD_GRAYSCALE );
    if ( mat1.empty() )
        return;

    cv::Mat mat2 = mat1 - cv::Scalar(50);
    //cv::imshow("mat1", mat1 );
    //cv::imshow("mat2", mat2 );
    //cv::waitKey(0);

    double similarity = CalcSimilarity ( mat1, mat2 );
}