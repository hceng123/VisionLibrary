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
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include "StopWatch.h"

#include <iostream>
using namespace std;
//using namespace cv;
using namespace AOI;
using namespace Vision;


template<class T>
void printfMat ( const cv::Mat &mat)
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
	//scanf_s("%f %f %f", &WZ, &TX, &TY);

	// Here we will store our images.
	IplImage* pColorPhoto = 0; // Photo of a butterfly on a flower.
	IplImage* pGrayPhoto = 0;  // Grayscale copy of the photo.
	IplImage* pImgT = 0;	   // Template T.
	IplImage* pImgI = 0;	   // Image I.

	// Here we will store our warp matrix.
	CvMat* W = 0;  // Warp W(p)

	// Create two simple windows. 
	cvNamedWindow("template"); // Here we will display T(x).
	//cvNamedWindow("image"); // Here we will display I(x).

	// Load photo.
	pColorPhoto = cvLoadImage(".\\data\\photo.jpg");
	if (NULL == pColorPhoto)
		return -1;

	// Create other images.
	CvSize photo_size = cvSize(pColorPhoto->width, pColorPhoto->height);
	pGrayPhoto = cvCreateImage(photo_size, IPL_DEPTH_8U, 1);
	pImgT = cvCreateImage(photo_size, IPL_DEPTH_8U, 1);
	pImgI = cvCreateImage(photo_size, IPL_DEPTH_8U, 1);

	// Convert photo to grayscale, because we need intensity values instead of RGB.	
	cvCvtColor(pColorPhoto, pGrayPhoto, CV_RGB2GRAY);

	// Create warp matrix, we will use it to create image I.
	//W = cvCreateMat(3, 3, CV_32F);
	cv::Mat matRotation = cv::getRotationMatrix2D(cv::Point(210, 175), 3, 1);
	printf("Rotation Mat \n");
	printfMat<double> ( matRotation );
	cv::Mat matRotated;
	cv::Mat src = cv::cvarrToMat ( pGrayPhoto );
	warpAffine(src, matRotated, matRotation, src.size());
	cv::line(matRotated, cv::Point(210, 175), cv::Point(260, 225), cv::Scalar(255,255, 0), 2 );		//Manually add defect.
	circle(matRotated, cv::Point(240, 190), 15, cv::Scalar(255,255, 0), CV_FILLED);

    circle(matRotated, cv::Point(200, 160), 15, cv::Scalar(0,0, 0), CV_FILLED);
	imshow("Rotated", matRotated);

	// Create template T
	// Set region of interest to butterfly's bounding rectangle.
	cvCopy(pGrayPhoto, pImgT);
	CvRect omega = cvRect(110, 100, 200, 150);

	// Create I by warping photo with sub-pixel accuracy. 
	//init_warp(W, WZ, TX, TY);
	//warp_image(pGrayPhoto, pImgI, W);

	// Draw the initial template position approximation rectangle.
	//cvSetIdentity(W);
	//draw_warped_rect(pImgI, omega, W);

	// Show T and I and wait for key press.
	cvSetImageROI(pImgT, omega);
	cvShowImage("template", pImgT);
	//cvShowImage("image", pImgI);
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
	PR_ALIGN_ALGORITHM enAlgorithm = PR_ALIGN_ALGORITHM::SURF;
	PR_LearnTmplCmd stLrnTmplCmd;
	PR_LearnTmplRpy stLrnTmplRpy;
	stLrnTmplCmd.mat = cv::cvarrToMat(pImgT, true);
	stLrnTmplCmd.rectLrn = omega;
	stLrnTmplCmd.enAlgorithm = enAlgorithm;
	visionAlgorithm.learn(&stLrnTmplCmd, &stLrnTmplRpy);

	PR_FindObjCmd stFindObjCmd;
	PR_FindObjRpy stFindObjRpy;
	stFindObjCmd.mat = matRotated;
	stFindObjCmd.rectLrn = omega;
	stFindObjCmd.vecModelKeyPoint = stLrnTmplRpy.vecKeyPoint;
	stFindObjCmd.matModelDescritor = stLrnTmplRpy.matDescritor;
	stFindObjCmd.ptExpectedPos = stLrnTmplRpy.ptCenter;
	cv::Rect rectSrchWindow ( cv::Point(100, 80), cv::Point ( stFindObjCmd.mat.cols - 70, stFindObjCmd.mat.rows - 60 ) );
	stFindObjCmd.rectSrchWindow = rectSrchWindow;
	stFindObjCmd.enAlgorithm = enAlgorithm;

	CStopWatch stopWatch;
	visionAlgorithm.findObject(&stFindObjCmd, &stFindObjRpy);
	printf("SURF result, time consume %d us \n", stopWatch.NowInMicro() );
    printf("Match score %.2f \n", stFindObjRpy.fMatchScore );
	printfMat<double>(stFindObjRpy.matHomography);

	PR_InspCmd stInspCmd;
	PR_InspRpy stInspRpy;
	stInspCmd.matTmpl = cv::cvarrToMat(pImgT, true);
	stInspCmd.rectLrn = omega;
	stInspCmd.ptObjPos = stFindObjRpy.ptObjPos;
	stInspCmd.fRotation = stFindObjRpy.fRotation;
	stInspCmd.matInsp = matRotated;
	stInspCmd.u16NumOfDefectCriteria = 2;
	stInspCmd.astDefectCriteria[0].fArea = 50;
	stInspCmd.astDefectCriteria[0].fLength = 0.f;
	stInspCmd.astDefectCriteria[0].ubContrast = 20;
	stInspCmd.astDefectCriteria[0].enAttribute = PR_DEFECT_ATTRIBUTE::BOTH;

	stInspCmd.astDefectCriteria[1].fLength = 50.0;
	stInspCmd.astDefectCriteria[1].fArea = 0.f;
	stInspCmd.astDefectCriteria[1].ubContrast = 20;
	visionAlgorithm.inspect ( &stInspCmd, &stInspRpy );

	PR_AlignCmd stAlignCmd;
	PR_AlignRpy stAlignRpy;
	stAlignCmd.matInput = matRotated;
	stAlignCmd.rectLrn = omega;
	stAlignCmd.rectSrchWindow.x = stFindObjRpy.ptObjPos.x - ( omega.width + 30.f ) / 2.0f;
	stAlignCmd.rectSrchWindow.y = stFindObjRpy.ptObjPos.y - ( omega.height + 30.f) / 2.0f;
	stAlignCmd.rectSrchWindow.width = (omega.width + 30.f);
	stAlignCmd.rectSrchWindow.height = (omega.height + 30.f);
	stAlignCmd.ptExpectedPos = stLrnTmplRpy.ptCenter;
	cv::Mat matT = cv::cvarrToMat(pImgT);
	stAlignCmd.matTmpl = cv::Mat(matT, omega);
	visionAlgorithm.align(&stAlignCmd, &stAlignRpy);

	printf("SURF and findTransformECC result \n");
	printfMat<float>(stAlignRpy.matAffine);

	/* Lucas-Kanade */
	//align_image_forwards_additive(pImgT, omega, pImgI);

	//Mat matT = cvarrToMat ( pImgT );
	cv::Mat matLearn(matT, omega);
	//Mat matI = cvarrToMat(pImgI);
	cv::Mat resultW(2, 3, CV_32F);
	resultW.setTo(cv::Scalar(0));
	resultW.at<float>(0, 0) = 1.0;
	resultW.at<float>(1, 1) = 1.0;
	//resultW.at<float>(2,2) = 1.0;
	findTransformECC(matT, matRotated, resultW, cv::MOTION_AFFINE);

	cv::Mat matOriginalPos(3, 1, CV_32F);
	matOriginalPos.at<float>(0, 0) = omega.x + omega.width / 2.0f;
	matOriginalPos.at<float>(1, 0) = omega.y + omega.height / 2.0f;
	matOriginalPos.at<float>(2, 0) = 1.0;

	cv::Mat matDestPos(3, 1, CV_32F);
	matDestPos = resultW * matOriginalPos;

	cv::Point2d pt2dDest;
	pt2dDest.x = (float)matDestPos.at<float>(0, 0);
	pt2dDest.y = (float)matDestPos.at<float>(1, 0);

	printf("findTransformECC result \n");
	printfMat<float>(resultW);

	cv::Mat resultW1 = estimateRigidTransform(matT, matRotated, false);
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
    visionAlgorithm.dumpTimeLog("PRTime.log");
	cvWaitKey();
	return 0;
}

int TestVisionAlgorithmShape()
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
    pColorPhoto = cvLoadImage(".\\data\\TestShape.png");
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
    CvRect omega = cvRect(70, 70, 200, 200);

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
    PR_ALIGN_ALGORITHM enAlgorithm = PR_ALIGN_ALGORITHM::SURF;
    PR_LearnTmplCmd stLrnTmplCmd;
    PR_LearnTmplRpy stLrnTmplRpy;
    stLrnTmplCmd.mat = cv::cvarrToMat ( pImgT, true );
    stLrnTmplCmd.rectLrn = omega;
    stLrnTmplCmd.enAlgorithm = enAlgorithm;
    visionAlgorithm.learn ( &stLrnTmplCmd, &stLrnTmplRpy );

    PR_FindObjCmd stFindObjCmd;
    PR_FindObjRpy stFindObjRpy;
    stFindObjCmd.mat = cv::cvarrToMat ( pImgI, true );
    stFindObjCmd.rectLrn = omega;
    stFindObjCmd.vecModelKeyPoint = stLrnTmplRpy.vecKeyPoint;
    stFindObjCmd.matModelDescritor = stLrnTmplRpy.matDescritor;
    stFindObjCmd.ptExpectedPos = stLrnTmplRpy.ptCenter;
    stFindObjCmd.rectSrchWindow = cv::Rect ( 0, 0, stFindObjCmd.mat.cols, stFindObjCmd.mat.rows );
    stFindObjCmd.enAlgorithm = enAlgorithm;

    visionAlgorithm.findObject ( &stFindObjCmd, &stFindObjRpy );
    printf("SURF result \n");
    printfMat<double>(stFindObjRpy.matHomography);

    return 0;
}

extern void runGHT(char c);

void TestGHT()
{
    runGHT('t');
    runGHT('r');
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
    //TestVisionAlgorithmShape();

    //TestCanny();
    //TestFindContours();
    //TestGHT();

    //getchar();
    return 0;
}

