// TestCV.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\video\background_segm.hpp"
#include "UtilityFunction.h"
#include "VisionAlgorithm.h"
#include "auxfunc.h"
#include "opencv2\video\tracking.hpp"

#include <iostream>
using namespace std;
using namespace cv;
using namespace AOI;
using namespace Vision;

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

    Mat mat = imread("MyPic.jpg", CV_LOAD_IMAGE_COLOR);

    int intensityToChange = 40;
    //invert and display the inverted image
    Mat matH;
    mat.convertTo ( matH, -1, 2, 0 );

    Mat matL;
    mat.convertTo ( matL, -1, 0.5, 0 );

    namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    namedWindow("High Contrast", CV_WINDOW_AUTOSIZE);
    namedWindow("Low Contrast", CV_WINDOW_AUTOSIZE);

    imshow("Original Image", mat);
    imshow("High Contrast", matH);
    imshow("Low Contrast", matL);

    waitKey(0);

    destroyAllWindows(); //destroy all open windows

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
   // CvMat *mat = cvCreateMat ( 100, 100, CV_16UC1/*IPL_DEPTH_16U*/ );
    //mat->data = ( uchar* )buf;
    Mat mat( size, CV_16UC1,Scalar(0,0,0));
    mat.data = (uchar*)buf;
    //IplImage *image = cvCreateMat ( 100, 100, CV_16UC1/*IPL_DEPTH_16U*/ );
}

int TestSaveImage()
{
    Mat img(650, 600, CV_16UC3, Scalar ( 0, 50000, 50000 ) );
    if (img.empty())  {
        cout << "Error: Image can not be loaded ...!";
        return -1;
    }
    vector<int> compression_params;
    compression_params.push_back ( CV_IMWRITE_JPEG_QUALITY );
    compression_params.push_back ( 98 );
    bool bSuccess = imwrite ("TestImage.png", img, compression_params );

    Mat matGray;
    img.convertTo ( matGray, CV_8U );
    Mat result;
    Canny( matGray, result, 50, 150);

    bSuccess = imwrite ("CannyResult.png", result, compression_params );

    if( ! bSuccess )    {
        cout << "Error : failed to save image !" << endl;
        return -1;
    }

    namedWindow ( "MyWindow", CV_WINDOW_AUTOSIZE );
    imshow ( "MyWindow", img );
    waitKey ( 0 );

    destroyWindow ( "MyWindows");
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

template<class T>
void printfMat ( const Mat &mat)
{
    for ( short row = 0; row < mat.rows; ++ row )
    {
        for ( short col = 0; col < mat.cols; ++ col )
        {
            printf ("%f ", mat.at<T>(row, col) );
        }
        printf("\n");
    }
}

int TestVisionAlgorithm()
{
    AOI::Vision::VisionAlgorithm visionAlgorithm;

    float WZ = 0, TX = 0, TY = 0;
    printf("Please enter WZ, TX and TY, separated by space.\n");
    printf("Example: -0.01 5 -3\n");
    printf(">");
    scanf_s("%f %f %f", &WZ, &TX, &TY);

    // Here we will store our images.
    IplImage* pColorPhoto = 0; // Photo of a butterfly on a flower.
    IplImage* pGrayPhoto = 0;  // Grayscale copy of the photo.
    IplImage* pImgT = 0;	   // Template T.
    IplImage* pImgI = 0;	   // Image I.

    // Here we will store our warp matrix.
    CvMat* W = 0;  // Warp W(p)

    // Create two simple windows. 
    cvNamedWindow("template"); // Here we will display T(x).
    cvNamedWindow("image"); // Here we will display I(x).

    // Load photo.
    pColorPhoto = cvLoadImage(".\\data\\photo.jpg");
    if ( NULL == pColorPhoto )
        return -1;

    // Create other images.
    CvSize photo_size = cvSize(pColorPhoto->width, pColorPhoto->height);
    pGrayPhoto = cvCreateImage(photo_size, IPL_DEPTH_8U, 1);
    pImgT = cvCreateImage(photo_size, IPL_DEPTH_8U, 1);
    pImgI = cvCreateImage(photo_size, IPL_DEPTH_8U, 1);

    // Convert photo to grayscale, because we need intensity values instead of RGB.	
    cvCvtColor(pColorPhoto, pGrayPhoto, CV_RGB2GRAY);

    // Create warp matrix, we will use it to create image I.
    W = cvCreateMat(3, 3, CV_32F);

    // Create template T
    // Set region of interest to butterfly's bounding rectangle.
    cvCopy(pGrayPhoto, pImgT);
    CvRect omega = cvRect(110, 100, 200, 150);

    // Create I by warping photo with sub-pixel accuracy. 
    init_warp(W, WZ, TX, TY);
    warp_image(pGrayPhoto, pImgI, W);

    // Draw the initial template position approximation rectangle.
    //cvSetIdentity(W);
    draw_warped_rect(pImgI, omega, W);

    // Show T and I and wait for key press.
    cvSetImageROI(pImgT, omega);
    cvShowImage("template", pImgT);
    cvShowImage("image", pImgI);
   // printf("Press any key to start Lucas-Kanade algorithm.\n");
    cvWaitKey(0);
    cvResetImageROI(pImgT);

    // Print what parameters were entered by user.
    printf("==========================================================\n");
    printf("Ground truth:  WZ=%f TX=%f TY=%f\n", WZ, TX, TY);
    printf("==========================================================\n");

    //init_warp(W, WZ, TX, TY);

    // Restore image I
    //warp_image(pGrayPhoto, pImgI, W);
    short nAlgorithm = ALIGN_ALGORITHM_SURF;
    PR_LearnTmplCmd stLrnTmplCmd;
    PR_LearnTmplRpy stLrnTmplRpy;
    stLrnTmplCmd.mat = cvarrToMat ( pImgT, true );
    stLrnTmplCmd.rectLrn = omega;
    stLrnTmplCmd.nAlgorithm = nAlgorithm;
    visionAlgorithm.learn ( &stLrnTmplCmd, &stLrnTmplRpy );

    PR_FindObjCmd stFindObjCmd;
    PR_FindObjRpy stFindObjRpy;
    stFindObjCmd.mat = cvarrToMat ( pImgI, true );
    stFindObjCmd.rectLrn = omega;
    stFindObjCmd.vecModelKeyPoint = stLrnTmplRpy.vecKeyPoint;
    stFindObjCmd.matModelDescritor = stLrnTmplRpy.matDescritor;
    stFindObjCmd.ptExpectedPos = stLrnTmplRpy.ptCenter;
    stFindObjCmd.rectSrchWindow = Rect ( 0, 0, stFindObjCmd.mat.cols, stFindObjCmd.mat.rows );
    stFindObjCmd.nAlgorithm = nAlgorithm;

    visionAlgorithm.findObject ( &stFindObjCmd, &stFindObjRpy );
    printf("SURF result \n");
    printfMat<double>(stFindObjRpy.matHomography);


    PR_AlignCmd stAlignCmd;
    PR_AlignRpy stAlignRpy;
    stAlignCmd.matInput = cvarrToMat ( pImgI, true );
    stAlignCmd.rectSrchWindow.x = stFindObjRpy.ptObjPos.x - omega.width / 2.0f;
    stAlignCmd.rectSrchWindow.y = stFindObjRpy.ptObjPos.y - omega.height / 2.0f;
    stAlignCmd.rectSrchWindow.width = omega.width;
    stAlignCmd.rectSrchWindow.height = omega.height;
    Mat matT = cvarrToMat ( pImgT );
    stAlignCmd.matTmpl = Mat ( matT, omega );
    visionAlgorithm.align( &stAlignCmd, &stAlignRpy );

    printf("SURF and findTransformECC result \n");
    printfMat<float>(stAlignRpy.matAffine);

    
    /* Lucas-Kanade */
    //align_image_forwards_additive(pImgT, omega, pImgI);

    //Mat matT = cvarrToMat ( pImgT );
    Mat matLearn ( matT, omega );
    Mat matI = cvarrToMat ( pImgI );
    Mat resultW ( 2, 3, CV_32F);
    resultW.setTo(Scalar(0));
    resultW.at<float>(0,0) = 1.0;
    resultW.at<float>(1,1) = 1.0;
    //resultW.at<float>(2,2) = 1.0;
    findTransformECC ( matT , matI, resultW, MOTION_AFFINE );

    cv::Mat matOriginalPos ( 3, 1, CV_32F );
    matOriginalPos.at<float>(0, 0) = omega.x + omega.width / 2.0f;
    matOriginalPos.at<float>(1, 0) = omega.y + omega.height / 2.0f;
    matOriginalPos.at<float>(2, 0) = 1.0;

    Mat matDestPos ( 3, 1, CV_32F );
    matDestPos = resultW * matOriginalPos;

    Point2d pt2dDest;
    pt2dDest.x = (float) matDestPos.at<float>(0, 0);
    pt2dDest.y = (float) matDestPos.at<float>(1, 0);

    printf("findTransformECC result \n");
    printfMat<float>(resultW);

    Mat resultW1 = estimateRigidTransform(matT, matI, false);
    printf("estimateRigidTransform resultW \n");
    printfMat<double>(resultW1);

    // Restore image I   
    
    
    //for ( short row = 0; row < stAlignRpy.matHomography.rows; ++ row )
    //{
    //    for ( short col = 0; col < stAlignRpy.matHomography.cols; ++ col )
    //    {
    //        resultW.at<float>(row, col) = stAlignRpy.matHomography.at<double>(row, col);
    //    }
    //    printf("\n");
    //}
    
    
    //CvMat n = resultW;
    //warp_image ( pGrayPhoto, pImgI, &n );

    //printf("Press any key to start Baker-Dellaert-Matthews algorithm.\n");
    cvWaitKey();
    return 0;
}

int _tmain(int argc, _TCHAR* argv[])
{
    //TestDisplayVideo();
    //TestSaveImage();
    //TestNotImage();    
    //TestAdjustIntensityImage();
    //TestAdjustContrastImage();

    //TestHassianCorner1();

    //CvMat* X = cvCreateMat(3, 1, CV_32F);
    //TestMat();

    //TestCSDN();
    //VisionAlgorithm visionAlgorithm;

    //VisionAlgorithm v1(visionAlgorithm);

    //TestIngensityDistribute();
    TestVisionAlgorithm();

    getchar();
    return 0;
}

