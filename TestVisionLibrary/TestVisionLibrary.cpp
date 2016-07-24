// TestVisionLibrary.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace AOI::Vision;

template<class T>
void printfMat(const Mat &mat)
{
	for (short row = 0; row < mat.rows; ++row)
	{
		for (short col = 0; col < mat.cols; ++col)
		{
			printf("%f ", mat.at<T>(row, col));
		}
		printf("\n");
	}
}

int TestVisionAlgorithm()
{
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
	PR_LEARN_TMPL_CMD stLrnTmplCmd;
	PR_LEARN_TMPL_RPY stLrnTmplRpy;
	stLrnTmplCmd.mat = cv::cvarrToMat(pImgT, true);
	stLrnTmplCmd.rectLrn = omega;
	stLrnTmplCmd.enAlgorithm = enAlgorithm;
	PR_LearnTmpl(&stLrnTmplCmd, &stLrnTmplRpy);

	PR_SRCH_TMPL_CMD stSrchTmplCmd;
	PR_SRCH_TMPL_RPY stSrchTmplRpy;
	stSrchTmplCmd.mat = matRotated;
	stSrchTmplCmd.rectLrn = omega;
	stSrchTmplCmd.nRecordID = stLrnTmplRpy.nRecordID;
	stSrchTmplCmd.ptExpectedPos = stLrnTmplRpy.ptCenter;
	cv::Rect rectSrchWindow ( cv::Point(100, 80), cv::Point ( stSrchTmplCmd.mat.cols - 70, stSrchTmplCmd.mat.rows - 60 ) );
	stSrchTmplCmd.rectSrchWindow = rectSrchWindow;
	stSrchTmplCmd.enAlgorithm = enAlgorithm;

	PR_SrchTmpl(&stSrchTmplCmd, &stSrchTmplRpy);

    printf("Match score %.2f \n", stSrchTmplRpy.fMatchScore );
	printfMat<double>(stSrchTmplRpy.matHomography);

	//PR_InspCmd stInspCmd;
	//PR_InspRpy stInspRpy;
	//stInspCmd.matTmpl = cv::cvarrToMat(pImgT, true);
	//stInspCmd.rectLrn = omega;
	//stInspCmd.ptObjPos = stSrchTmplRpy.ptObjPos;
	//stInspCmd.fRotation = stSrchTmplRpy.fRotation;
	//stInspCmd.matInsp = matRotated;
	//stInspCmd.u16NumOfDefectCriteria = 2;
	//stInspCmd.astDefectCriteria[0].fArea = 50;
	//stInspCmd.astDefectCriteria[0].fLength = 0.f;
	//stInspCmd.astDefectCriteria[0].ubContrast = 20;
	//stInspCmd.astDefectCriteria[0].enAttribute = PR_DEFECT_ATTRIBUTE::BOTH;

	//stInspCmd.astDefectCriteria[1].fLength = 50.0;
	//stInspCmd.astDefectCriteria[1].fArea = 0.f;
	//stInspCmd.astDefectCriteria[1].ubContrast = 20;
	//visionAlgorithm.inspect ( &stInspCmd, &stInspRpy );

	//PR_AlignCmd stAlignCmd;
	//PR_AlignRpy stAlignRpy;
	//stAlignCmd.matInput = matRotated;
	//stAlignCmd.rectLrn = omega;
	//stAlignCmd.rectSrchWindow.x = stFindObjRpy.ptObjPos.x - ( omega.width + 30.f ) / 2.0f;
	//stAlignCmd.rectSrchWindow.y = stFindObjRpy.ptObjPos.y - ( omega.height + 30.f) / 2.0f;
	//stAlignCmd.rectSrchWindow.width = (omega.width + 30.f);
	//stAlignCmd.rectSrchWindow.height = (omega.height + 30.f);
	//stAlignCmd.ptExpectedPos = stLrnTmplRpy.ptCenter;
	//cv::Mat matT = cv::cvarrToMat(pImgT);
	//stAlignCmd.matTmpl = cv::Mat(matT, omega);
	//visionAlgorithm.align(&stAlignCmd, &stAlignRpy);

	//printf("SURF and findTransformECC result \n");
	//printfMat<float>(stAlignRpy.matAffine);

	/* Lucas-Kanade */
	//align_image_forwards_additive(pImgT, omega, pImgI);

	//printf("Press any key to start Baker-Dellaert-Matthews algorithm.\n");
    PR_DumpTimeLog("PRTime.log");
	cvWaitKey();
	return 0;
}

int _tmain(int argc, _TCHAR* argv[])
{
    TestVisionAlgorithm();
	return 0;
}

