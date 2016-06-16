// TestOpenCVConsole.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core.hpp"
#include "VisionAlgorithm.h"
#include "auxfunc.h"
#include "opencv2\video\tracking.hpp"
#include "StopWatch.h"

using namespace cv;
using namespace std;
using namespace AOI::Vision;

int TestErode()
{
	IplImage* img = cvLoadImage("MyPic.jpg");
	if (NULL == img)
		return -1;
	cvNamedWindow("MyWindow");
	cvShowImage("MyWindow", img);

	//erode and display the eroded image
	IplConvKernel stConvKernel;
	memset(&stConvKernel, 0, sizeof(IplConvKernel));
	stConvKernel.anchorX = 2;
	stConvKernel.anchorY = 2;
	stConvKernel.nCols = 3;
	stConvKernel.nRows = 3;
	stConvKernel.nShiftR = 1000;
	int size = stConvKernel.nCols * stConvKernel.nRows;

	stConvKernel.values = new int[ size ];
	cvDilate(img, img, &stConvKernel, 1);
	cvErode(img, img, &stConvKernel, 1);
	cvNamedWindow("Eroded");
	cvShowImage("Eroded", img);

	cvWaitKey(0);

	//cleaning up
	cvDestroyWindow("MyWindow");
	cvDestroyWindow("Eroded");
	cvReleaseImage(&img);

	return 0;
}

int TestHistogramEqualizeGray()
{
	Mat img = imread("MyPic.jpg", CV_LOAD_IMAGE_COLOR);
	if (img.empty()) {
		cout << "Failed to load image" << endl;
		return -1;
	}

	cvtColor ( img, img, CV_BGR2GRAY );

	Mat img_hist_equalized;
	equalizeHist ( img, img_hist_equalized);

	//create windows
	namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
	namedWindow("Histogram Equalized", CV_WINDOW_AUTOSIZE);

	//show the image
	imshow("Original Image", img);
	imshow("Histogram Equalized", img_hist_equalized);

	waitKey(0); //wait for key press

	destroyAllWindows(); //destroy all open windows

	return 0;
}

int TestHistogramEqualizeColor()
{
	Mat img = imread("MyPic.jpg", CV_LOAD_IMAGE_COLOR);
	if (img.empty()) {
		cout << "Failed to load image" << endl;
		return -1;
	}

	Mat img_hist_equalized;
	cvtColor ( img, img_hist_equalized, CV_BGR2YCrCb );

	vector<Mat> channels;
	split (img_hist_equalized, channels );
	equalizeHist ( channels[0], channels[0] );
	merge (channels, img_hist_equalized );

	cvtColor ( img_hist_equalized, img_hist_equalized, CV_YCrCb2BGR );

	//create windows
	namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
	namedWindow("Histogram Equalized", CV_WINDOW_AUTOSIZE);

	//show the image
	imshow("Original Image", img);
	imshow("Histogram Equalized", img_hist_equalized);

	waitKey(0); //wait for key press

	destroyAllWindows(); //destroy all open windows

	return 0;
}

int TestImageBlur()
{
	//create 2 empty windows
	namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
	namedWindow("Smoothed Image", CV_WINDOW_AUTOSIZE);

	Mat src = imread ( "MyPic.jpg", IMREAD_COLOR);

	imshow("Original Image", src );

	Mat dst;
	char szBuffer[35];

	for (int i = 1; i <= 31; i += 2) {
		_snprintf_s ( szBuffer, 35,  "Kernal size %d x %d", i, i );

		//blur ( src, dst, Size ( i, i ) );
		//GaussianBlur ( src, dst, Size ( i , i ), 5, 5 );
		//medianBlur ( src, dst, i );
		bilateralFilter ( src, dst, i, i ,i );

		putText ( dst, szBuffer, Point ( src.cols / 4, src.rows / 8 ), CV_FONT_HERSHEY_COMPLEX, 1, Scalar ( 255, 255, 255 ) );

		//show the blurred image with the text
		imshow("Smoothed Image", dst);

		//wait for 2 seconds
		int c = waitKey(2000);

		//if the "esc" key is pressed during the wait, return
		if (c == 27)
		{
			return 0;
		}

		dst = Mat::zeros ( src.size(), src.type() );
	}

	//copy the text to the "zBuffer"
	_snprintf_s(szBuffer, 35, "Press Any Key to Exit");
	
	//put the text in the "zBuffer" to the "dst" image
	putText(dst, szBuffer, Point(src.cols / 4, src.rows / 2), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));

	//show the black image with the text
	imshow("Smoothed Image", dst);

	//wait for a key press infinitely
	waitKey(0);

	return 0;
}

int TestConvertScale()
{
	//create 2 empty windows
	namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
	namedWindow("Scaled Image", CV_WINDOW_AUTOSIZE);

	IplImage *src = cvLoadImage ( "MyPic.jpg", IMREAD_COLOR );

	cvShowImage("Original Image", src );
	CvSize imageSize = cvSize ( src->width, src->height );
	Size size;
	size.width = imageSize.width;
	size.height = imageSize.height;

	IplImage *dst = cvCreateImage ( imageSize, src->depth, src->nChannels );

	cvConvertScale ( src, dst, 0.5 );
	//show the black image with the text
	cvShowImage("Scaled Image", dst);

	//wait for a key press infinitely
	waitKey(0);

	return 0;
}

int TestTrackBar()
{
	Mat src = imread("MyPic.jpg", CV_LOAD_IMAGE_COLOR);
	if (src.empty()) {
		cout << "Failed to load image" << endl;
		return -1;
	}

	namedWindow ( "My Window", 1);
	
	int iSlideValue1 = 50;
	createTrackbar ("Brightness", "My Window", &iSlideValue1, 100 );

	//Create trackbar to change contrast
	int iSliderValue2 = 50;
	createTrackbar("Contrast", "My Window", &iSliderValue2, 100);

	while (true) {
		Mat dst;
		int iBrightness = iSlideValue1 - 50;
		double dContrast = iSliderValue2 / 2;

		src.convertTo ( dst, -1, iBrightness, dContrast );

		//show the brightness and contrast adjusted image
		imshow("My Window", dst);

		// Wait until user press some key for 50ms
		int iKey = waitKey(50);

		//if user press 'ESC' key
		if (iKey == 27)
		{
			break;
		}
	}

	return 0;
}

Mat gSrc;

void AdjustImage( int iValueForBrightness, int iValueForContrast)
{
	Mat dst;
	int iBrightness = iValueForBrightness - 50;
	double dContrast = iValueForContrast / 50.0;

	//Calculated contrast and brightness value
	cout << "MyCallbackForContrast : Contrast=" << dContrast << ", Brightness=" << iBrightness << endl;

	//adjust the brightness and contrast
	gSrc.convertTo(dst, -1, dContrast, iBrightness);

	//show the brightness and contrast adjusted image
	imshow("My Window", dst);
}

void MyCallbackForBrightness(int iValueForBrightness, void *userData)
{
	int iValueForConstrast = * ( static_cast<int *>(userData) );
	AdjustImage ( iValueForBrightness, iValueForConstrast );
}

void MyCallbackForContrast(int iValueForConstrast, void *userData)
{
	int iValueforBrightness = *(static_cast<int *>(userData));
	AdjustImage ( iValueforBrightness, iValueForConstrast );
}

void CallBackFunc(int iEvent, int x, int y, int flags, void *userdata) {
	if (EVENT_LBUTTONDOWN == iEvent) {
		cout << "Left button of mouse clicked, poition (" << x << "," << y << ")" <<endl;
	}
	else if (EVENT_RBUTTONDOWN == iEvent) {
		cout << "Right button of mouse clicked, poition (" << x << "," << y << ")" << endl;
	}
	else if (EVENT_MBUTTONDOWN == iEvent) {
		cout << "Middle button of mouse clicked, poition (" << x << "," << y << ")" << endl;
	}
	else if (EVENT_MOUSEMOVE == iEvent) {
		cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
	}
}

int TestTrackBarCallBack()
{
	gSrc = imread("MyPic.jpg", CV_LOAD_IMAGE_COLOR);
	if (gSrc.empty()) {
		cout << "Failed to load image" << endl;
		return -1;
	}

	namedWindow("My Window", 1);

	int iValueForBrightness = 50;
	int iValueForContrast = 50;
	createTrackbar("Brightness", "My Window", &iValueForBrightness, 100, MyCallbackForBrightness, &iValueForBrightness );

	createTrackbar("Contrast", "My Window", &iValueForContrast, 100, MyCallbackForContrast, &iValueForBrightness );

	setMouseCallback ( "My Window", CallBackFunc, NULL );
	imshow("My Window", gSrc );

	waitKey ( 0 );

	return 0;
}

int TestRotateImage()
{
	Mat src = imread ( "MyPic.jpg", CV_LOAD_IMAGE_COLOR );

	const char *pzOriginalImage = "Original Image";
	namedWindow ( pzOriginalImage, CV_WINDOW_AUTOSIZE );
	imshow ( pzOriginalImage, src );

	const char *pzRotatedImage = "Rotated Image";
	namedWindow ( pzRotatedImage, CV_WINDOW_AUTOSIZE );

	int iAngle = 180;
	createTrackbar ( "Angle", pzRotatedImage, &iAngle, 360 );

	int iImageHeight = src.rows / 2;
	int iImageWidth = src.cols / 2;

	while (true) {
		Mat matRotation = getRotationMatrix2D ( Point( iImageWidth, iImageHeight), iAngle - 180, 1 );
		Mat imRotated;
		warpAffine ( src, imRotated, matRotation, src.size() );
		imshow ( pzRotatedImage, imRotated );

		int iRet = waitKey ( 30 );
		if ( 27 == iRet )
			break;
	}
	return 0;
}

int TestDetectObject()
{
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "Cannot open webcam" << endl;
		return -1;
	}

	namedWindow ( "Control", CV_WINDOW_AUTOSIZE );
	int iLowH = 0;
	int iHighH = 179;

	int iLowS = 0;
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 255;

	cvCreateTrackbar ( "LowH", "Control", &iLowH, 179 );
	cvCreateTrackbar ( "HighH", "Control", &iHighH, 179 );

	cvCreateTrackbar ( "LowS", "Control", &iLowS, 255 );
	cvCreateTrackbar ( "HighS", "Control", &iHighS, 255 );

	cvCreateTrackbar ( "LowV", "Control", &iLowV, 255 );
	cvCreateTrackbar ( "HighV", "Control", &iHighV, 255 );

	while (true) {
		Mat imgOriginal;
		bool bSuccss = cap.read ( imgOriginal );
		if (!bSuccss) {
			cout << "Cannot read a frame from video camera" << endl;
			break;
		}

		Mat imgHSV;
		cvtColor ( imgOriginal, imgHSV, COLOR_BGR2HSV );

		Mat imgThresholded;
		inRange ( imgHSV, Scalar ( iLowH, iLowS, iLowV), Scalar ( iHighH, iHighS, iHighV ), imgThresholded );
		erode ( imgThresholded, imgThresholded, getStructuringElement ( MORPH_ELLIPSE, Size ( 5, 5 ) ) );
		dilate (imgThresholded, imgThresholded, getStructuringElement ( MORPH_ELLIPSE, Size ( 5, 5 ) ) );

		dilate (imgThresholded, imgThresholded, getStructuringElement ( MORPH_ELLIPSE, Size ( 5, 5 ) ) );
		erode ( imgThresholded, imgThresholded, getStructuringElement ( MORPH_ELLIPSE, Size ( 5, 5 ) ) );
		
		imshow ("Thresholded Image", imgThresholded );
		imshow ("Original", imgOriginal );

		if ( waitKey ( 30 ) == 27 )	{
			break;
		}
	}

	return 0;
}

int TestDetectRedObject()
{
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "Cannot open webcam" << endl;
		return -1;
	}

	namedWindow ( "Control", CV_WINDOW_AUTOSIZE );
	int iLowH = 170;
	int iHighH = 179;

	int iLowS = 150;
	int iHighS = 255;

	int iLowV = 60;
	int iHighV = 255;

	cvCreateTrackbar ( "LowH", "Control", &iLowH, 179 );
	cvCreateTrackbar ( "HighH", "Control", &iHighH, 179 );

	cvCreateTrackbar ( "LowS", "Control", &iLowS, 255 );
	cvCreateTrackbar ( "HighS", "Control", &iHighS, 255 );

	cvCreateTrackbar ( "LowV", "Control", &iLowV, 255 );
	cvCreateTrackbar ( "HighV", "Control", &iHighV, 255 );

	int iLastX = -1, iLastY = -1;
	Mat imgTmp;
	cap.read ( imgTmp );

	Mat imgLines = Mat::zeros ( imgTmp.size(), CV_8UC3 );

	while (true) {
		Mat imgOriginal;
		bool bSuccss = cap.read ( imgOriginal );
		if (!bSuccss) {
			cout << "Cannot read a frame from video camera" << endl;
			break;
		}

		Mat imgHSV;
		cvtColor ( imgOriginal, imgHSV, COLOR_BGR2HSV );

		Mat imgThresholded;
		inRange ( imgHSV, Scalar ( iLowH, iLowS, iLowV), Scalar ( iHighH, iHighS, iHighV ), imgThresholded );
		erode ( imgThresholded, imgThresholded, getStructuringElement ( MORPH_ELLIPSE, Size ( 5, 5 ) ) );
		dilate (imgThresholded, imgThresholded, getStructuringElement ( MORPH_ELLIPSE, Size ( 5, 5 ) ) );

		dilate (imgThresholded, imgThresholded, getStructuringElement ( MORPH_ELLIPSE, Size ( 5, 5 ) ) );
		erode ( imgThresholded, imgThresholded, getStructuringElement ( MORPH_ELLIPSE, Size ( 5, 5 ) ) );
		
		Moments oMoments = moments ( imgThresholded );
		double dM01 = oMoments.m01;
		double dM10 = oMoments.m10;
		double dArea = oMoments.m00;

		if ( dArea > 10000 )	{
			int posX = static_cast<int>(dM10 / dArea);
			int posY = static_cast<int>(dM01 / dArea);

			if ( iLastX >= 0 && iLastY >=0 && posX >=0 && posY >= 0 )
				line ( imgLines, Point ( posX, posY ), Point ( iLastX, iLastY ), Scalar ( 0, 0, 255 ), 2 );

			iLastX = posX;
			iLastY = posY;
		}
		
		imshow ("Thresholded Image", imgThresholded );

		imgOriginal += imgLines;
		imshow ("Original", imgOriginal );
		
		if ( waitKey ( 30 ) == 27 )	{
			break;
		}
	}

	return 0;
}

int TestFindContour()
{
	IplImage *img = cvLoadImage("FindContours.png", CV_LOAD_IMAGE_COLOR );
	if ( NULL == img )
		return -1;

	cvNamedWindow("Raw");
	cvShowImage("Raw", img );

	IplImage *imgGrayScale = cvCreateImage ( cvGetSize ( img ), 8, 1 );
	cvCvtColor ( img, imgGrayScale, CV_BGR2GRAY );

	cvThreshold ( imgGrayScale, imgGrayScale, 128, 255, CV_THRESH_BINARY );

	CvSeq *contours;
	CvSeq *result;

	CvMemStorage *storage = cvCreateMemStorage(0);

	cvFindContours ( imgGrayScale, storage, &contours, sizeof ( CvContour ), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint ( 0 , 0 ) );

	while ( contours )	{
		result = cvApproxPoly ( contours, sizeof ( CvContour ), storage, CV_POLY_APPROX_DP, cvContourPerimeter ( contours ) * 0.02, 0 );

		//if there are 3  vertices  in the contour(It should be a triangle)
		if (result->total == 3)
		{
			//iterating through each point
			CvPoint *pt[3];
			for (int i = 0; i < 3; i++) {
				pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			}

			//drawing lines around the triangle
			cvLine(img, *pt[0], *pt[1], cvScalar(255, 0, 0), 4);
			cvLine(img, *pt[1], *pt[2], cvScalar(255, 0, 0), 4);
			cvLine(img, *pt[2], *pt[0], cvScalar(255, 0, 0), 4);
		}

		//if there are 4 vertices in the contour(It should be a quadrilateral)
		else if (result->total == 4)
		{
			//iterating through each point
			CvPoint *pt[4];
			for (int i = 0; i < 4; i++) {
				pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			}

			//drawing lines around the quadrilateral
			cvLine(img, *pt[0], *pt[1], cvScalar(0, 255, 0), 4);
			cvLine(img, *pt[1], *pt[2], cvScalar(0, 255, 0), 4);
			cvLine(img, *pt[2], *pt[3], cvScalar(0, 255, 0), 4);
			cvLine(img, *pt[3], *pt[0], cvScalar(0, 255, 0), 4);
		}

		//if there are 7  vertices  in the contour(It should be a heptagon)
		else if (result->total == 7)
		{
			//iterating through each point
			CvPoint *pt[7];
			for (int i = 0; i < 7; i++) {
				pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			}

			//drawing lines around the heptagon
			cvLine(img, *pt[0], *pt[1], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[1], *pt[2], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[2], *pt[3], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[3], *pt[4], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[4], *pt[5], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[5], *pt[6], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[6], *pt[0], cvScalar(0, 0, 255), 4);
		}
		contours = contours->h_next;
	}

	//show the image in which identified shapes are marked   
	cvNamedWindow("Tracked");
	cvShowImage("Tracked", img);

	cvWaitKey(0); //wait for a key press

	 //cleaning up
	cvDestroyAllWindows();
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&img);
	cvReleaseImage(&imgGrayScale);
	return 0;
}

int TestFindTriangle()
{
	IplImage *img = cvLoadImage("DetectingContours.jpg", CV_LOAD_IMAGE_COLOR );
	if ( NULL == img )
		return -1;

	cvNamedWindow("Original");
	cvShowImage("Original", img );

	//smooth the original image
	cvSmooth ( img, img, CV_GAUSSIAN, 3, 3 );

	IplImage *imgGrayScale = cvCreateImage ( cvGetSize ( img ), 8, 1 );
	cvCvtColor ( img, imgGrayScale, CV_BGR2GRAY );

	cvNamedWindow ( "GrayScale Image" );
	cvShowImage ( "GrayScale Image", imgGrayScale );

	cvThreshold ( imgGrayScale, imgGrayScale, 100, 255, CV_THRESH_BINARY_INV );

	cvNamedWindow ( "Thresholded Image" );
	cvShowImage ( "Thresholded Image", imgGrayScale );

	CvSeq *contours;
	CvSeq *result;

	CvMemStorage *storage = cvCreateMemStorage(0);

	cvFindContours ( imgGrayScale, storage, &contours, sizeof ( CvContour ), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint ( 0 , 0 ) );
	cvPoint ( 0, 0 );

	while ( contours )	{
		result = cvApproxPoly ( contours, sizeof ( CvContour ), storage, CV_POLY_APPROX_DP, cvContourPerimeter ( contours ) * 0.02, 0 );

		//if there are 3  vertices  in the contour(It should be a triangle)
		if (result->total == 3 && fabs ( cvContourArea ( result, CV_WHOLE_SEQ ) ) > 100 )
		{
			//iterating through each point
			CvPoint *pt[3];
			for (int i = 0; i < 3; i++) {
				pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			}

			//drawing lines around the triangle
			cvLine(img, *pt[0], *pt[1], cvScalar(255, 0, 0), 4);
			cvLine(img, *pt[1], *pt[2], cvScalar(255, 0, 0), 4);
			cvLine(img, *pt[2], *pt[0], cvScalar(255, 0, 0), 4);
		}
		
		contours = contours->h_next;
	}

	//show the image in which identified shapes are marked   
	cvNamedWindow("Tracked");
	cvShowImage("Tracked", img);

	cvWaitKey(0); //wait for a key press

	 //cleaning up
	cvDestroyAllWindows();
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&img);
	cvReleaseImage(&imgGrayScale);
	return 0;
}

IplImage* imgTracking = NULL;
IplImage* imgShapeBorder = NULL;

int lastX1 = -1;
int lastY1 = -1;

int lastX2 = -1;
int lastY2 = -1;

void trackObject ( IplImage *imgThresh )	{
	CvSeq *contour;
	CvSeq *result;
	CvMemStorage *storage = cvCreateMemStorage ( 0 );
	cvZero ( imgShapeBorder );

	cvFindContours ( imgThresh, storage, &contour, sizeof ( CvContour ), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint ( 0, 0 ) );
	while ( contour )	{
		result = cvApproxPoly ( contour, sizeof (CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter (contour) * 0.02, 0 );
		if ( result->total == 3 && fabs ( cvContourArea ( contour ) ) > 100 )	{
			CvPoint *pt[3];
			for ( int i = 0; i < 3; ++ i )
				pt[i] = (CvPoint *)cvGetSeqElem ( result, i );

			cvLine ( imgShapeBorder, *pt[0], *pt[1], Scalar ( 0, 255, 0 ), 4 );
			cvLine ( imgShapeBorder, *pt[1], *pt[2], Scalar ( 0, 255, 0 ), 4 );
			cvLine ( imgShapeBorder, *pt[2], *pt[0], Scalar ( 0, 255, 0 ), 4 );

			int posX = ( pt[0]->x + pt[1]->x + pt[2]->x ) / 3;
			int posY = ( pt[0]->y + pt[1]->y + pt[2]->y ) / 3;

			if ( posX > 360 )	{
				if ( lastX1 >=0 && lastY1 >=0 && posX >= 0 && posY >= 0 )
					cvLine ( imgTracking, cvPoint ( posX, posY ), cvPoint ( lastX1, lastY1 ), cvScalar ( 0, 0, 255 ), 4 );

				lastX1 = posX;
				lastY1 = posY;
			}
			else
			{
				if ( lastX2 >= 0 && lastY2 >= 0 && posX >= 0 && posY >= 0 )
					cvLine ( imgTracking, cvPoint ( posX, posY ), cvPoint ( lastX2, lastY2 ), cvScalar ( 255, 0 , 0 ), 4 );
				lastX2 = posX;
				lastY2 = posY;
			}
		}

		contour = contour->h_next;
	}

	cvReleaseMemStorage ( &storage );
}

int TrackOnVideo ()
{
	VideoCapture cap(0);
	if ( ! cap.isOpened() )	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	//CvCapture * capture = cvCaptureFromCAM ( 0 );
	//if ( ! capture )	{
	//	cout << "Capture failure" <<endl;
	//	return -1;
	//}
	cvNamedWindow ( "Original Image" );

	Mat mat;
	bool bReadResult = cap.read ( mat );
	IplImage iplImage(mat);
	IplImage *frame = &iplImage;
	//frame = cvQueryFrame ( capture );
	if ( ! frame )
		return -1;

	cvShowImage ( "Original Image", frame );

	imgTracking = cvCreateImage ( cvGetSize ( frame ), IPL_DEPTH_8U, 3 );
	cvZero ( imgTracking );

	imgShapeBorder = cvCreateImage ( cvGetSize ( frame ), IPL_DEPTH_8U, 3 );
	cvZero ( imgShapeBorder );

	cvNamedWindow ( "Video" );
	
	while ( true )	{
		//frame = cvQueryFrame ( capture );
		//if ( ! frame ) break;

		bReadResult = cap.read(mat);
		IplImage iplImage(mat);
		IplImage *frame = &iplImage;

		cvShowImage ( "Original Image", frame );

		frame = cvCloneImage ( frame );

		cvSmooth ( frame, frame, CV_GAUSSIAN, 3, 3 );

		IplImage *imgGrayScale = cvCreateImage ( cvGetSize ( frame ), 8, 1 );
		cvCvtColor ( frame, imgGrayScale, CV_BGR2GRAY );
		cvThreshold ( imgGrayScale, imgGrayScale, 70, 255, CV_THRESH_BINARY_INV );

		cvShowImage ("Gray Scale", imgGrayScale );

		trackObject ( imgGrayScale );

		cvAdd ( frame, imgTracking, frame );
		cvAdd ( frame, imgShapeBorder, frame );
		cvShowImage ("Video", frame );

		cvReleaseImage ( &imgGrayScale );
		cvReleaseImage ( &frame );

		int c = cvWaitKey ( 30 );
		if ( (char)c == 27 ) break;
	}

	cvDestroyAllWindows();
	cvReleaseImage ( &imgTracking );
	//cvReleaseCapture ( &capture );

	return 0;
}

int TestCaptureImageFromCam()
{
	VideoCapture cap(0); // open the video camera no. 0

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	cout << "Frame size : " << dWidth << " x " << dHeight << endl;

	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

	while (1)
	{
		Mat frame;

		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		imshow("MyVideo", frame); //show the frame in "MyVideo" window

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
	return 0;
}

int TestCSDNCode()
{
	cvNamedWindow("a", 1);
	IplImage* img_8uc1 = cvLoadImage("1000.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage* img_edge = cvCreateImage(cvGetSize(img_8uc1), 8, 1);
	IplImage* img_8uc3 = cvCreateImage(cvGetSize(img_8uc1), 8, 3);	

	cvThreshold(img_8uc1, img_edge, 128, 255, CV_THRESH_BINARY);	

	CvMemStorage* storage = cvCreateMemStorage();//采用默认大小，即：0.

	CvChain* first_contour = NULL;

	int Nc = cvFindContours(img_edge, storage, (CvSeq **)&first_contour, sizeof(CvChain),
		CV_RETR_LIST,
		CV_CHAIN_CODE///*这个是关键参数*/
		);

	for (CvChain* c = first_contour; c != NULL; c = (CvChain *)c->h_next)
	{
		//CvSeqReader reader;
		CvChainPtReader reader;
		int i, total = c->total;
		//cvStartReadSeq((CvSeq*)c, &reader, 0);
		cvStartReadChainPoints ( c, &reader );

		for (i = 0; i < total; i++)
		{
			char code;
			CV_READ_SEQ_ELEM(code, reader);    printf(" %d: \n", code);       //得到轮廓的Freeman链码序列

			CvPoint point;

			point = cvReadChainPoint( &reader );//不能将参数 1 从“CvSeqReader”转换为“CvChainPtReader *”

			printf(" x=%d ;y=%d: \n", point.x, point.y);//得到链码坐标点。
		}
	}
	printf("Finished all contours.\n");

	cvCvtColor(img_8uc1, img_8uc3, CV_GRAY2BGR);
	cvShowImage("a", img_8uc3 );
	cvWaitKey(0);
	cvReleaseMemStorage(&storage);//释放
	cvDestroyWindow("a");
	cvReleaseImage(&img_8uc1);
	cvReleaseImage(&img_edge);
	cvReleaseImage(&img_8uc3);
	return 0;
}

int GenerateTransparentMask()
{
	const Mat mat = imread("DetectingContours.jpg");
	if (mat.empty())
		return -1;

	imshow("origianl", mat);

	Size size = mat.size();
	cv::Mat copy ( size, mat.type() );
	mat.copyTo ( copy );

	//cv::Mat mask(mat);	//Can not work, becuase mask will be just a copy of mat, draw on mask, means draw on mat too

	cv::Mat mask(size, CV_8U);	//mask must
	mask.setTo(Scalar(0));
	Rect rect(200, 100, 200, 200);
	rectangle(mask, rect, Scalar(255, 255, 255), CV_FILLED, CV_AA, 0);
	rectangle(mask, Rect(280, 120, 40, 40), Scalar(0, 0, 0), CV_FILLED);
	circle(mask, Point(300, 220), 30, Scalar(0), CV_FILLED);

	copy.setTo(Scalar(0, 0, 255), mask);
	//imshow("copy", copy);

	double alpha = 0.5;
	double beta = 1.0 - alpha;

	cv::Mat result(size, mat.type());
	cv::addWeighted(copy, alpha, mat, beta, 0.0, result);

	Point *pPoint = new Point(5);
	pPoint[0].x = 10;
	pPoint[0].y = 10;

	pPoint[1].x = 50;
	pPoint[1].y = 50;

	pPoint[2].x = 400;
	pPoint[2].y = 50;

	int nContours = 1;

	cv::polylines(result, &pPoint, &nContours, 1, false, Scalar(0, 255, 0), 2);

	cvNamedWindow("Result");
	imshow("Result", result);

	cvWaitKey(0); //wait for a key press
	return 0;
}

int TestImageZoom()
{
	/// General instructions
	printf("\n Zoom In-Out demo  \n ");
	printf("------------------ \n");
	printf(" * [u] -> Zoom in  \n");
	printf(" * [d] -> Zoom out \n");
	printf(" * [ESC] -> Close program \n \n");

	Mat src, tmp, dst;
	char* window_name = "Pyramids Demo";

	/// Test image - Make sure it s divisible by 2^{n}
	src = imread("./Pyramids_Tutorial_Original_Image.jpg");
	if (!src.data)
	{
		printf(" No data! -- Exiting the program \n");
		return -1;
	}

	tmp = src;
	dst = tmp;

	/// Create window
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	imshow(window_name, dst);

	/// Loop
	while (true)
	{
		int c;
		c = waitKey(10);

		if ((char)c == 27)
		{
			break;
		}
		if ((char)c == 'u')
		{
			pyrUp(tmp, dst, Size(tmp.cols * 2, tmp.rows * 2));
			printf("** Zoom In: Image x 2 \n");
		}
		else if ((char)c == 'd')
		{
			pyrDown(tmp, dst, Size(tmp.cols / 2, tmp.rows / 2));
			printf("** Zoom Out: Image / 2 \n");
		}

		imshow(window_name, dst);
		tmp = dst;
	}
	return 0;
}

int TestPolyline()
{
	// create a RGB colour image (set it to a black background)

	Mat img = Mat::zeros(400, 400, CV_8UC3);

	// define a polygon (as a vector of points)

	vector<Point> contour;
	contour.push_back(Point(50, 50));
	contour.push_back(Point(300, 50));
	contour.push_back(Point(350, 200));
	contour.push_back(Point(300, 150));
	contour.push_back(Point(150, 350));
	contour.push_back(Point(100, 100));

	// create a pointer to the data as an array of points (via a conversion to 
	// a Mat() object)

	const cv::Point *pts = (const cv::Point*) Mat(contour).data;
	int npts = Mat(contour).rows;

	std::cout << "Number of polygon vertices: " << npts << std::endl;

	// draw the polygon 

	polylines(img, &pts, &npts, 1,
		false, 			// draw closed contour (i.e. joint end to start) 
		Scalar(0, 255, 0),// colour RGB ordering (here = green) 
		3, 		        // line thickness
		CV_AA, 0);

	//cv::fillPoly(img, &pts, &npts, 1, Scalar(0, 255, 0));


	// do point in polygon test (by conversion/cast to a Mat() object)
	// define and test point one (draw it in red)

	Point2f test_pt;
	test_pt.x = 150;
	test_pt.y = 75;

	rectangle(img, test_pt, test_pt, Scalar(0, 0, 255), 3, 8, 0); // RED point

	if (pointPolygonTest(Mat(contour), test_pt, true) > 0){
		std::cout << "RED {" << test_pt.x << "," << test_pt.y
			<< "} is in the polygon (dist. "
			<< pointPolygonTest(Mat(contour), test_pt, 1) << ")"
			<< std::endl;
	}

	// define and test point two (draw it in blue)

	test_pt.x = 50;
	test_pt.y = 350;

	rectangle(img, test_pt, test_pt, Scalar(255, 0, 0), 3, 8, 0); // BLUE point

	if (pointPolygonTest(Mat(contour), test_pt, true) < 0){
		std::cout << "BLUE {" << test_pt.x << "," << test_pt.y
			<< "} is NOT in the polygon (dist. "
			<< pointPolygonTest(Mat(contour), test_pt, 1) << ")"
			<< std::endl;
	}

	// pointPolygonTest :- 
	// "The function determines whether the point is inside a contour, outside, 
	// or lies on an edge (or coincides with a vertex). It returns positive 
	// (inside), negative (outside) or zero (on an edge) value, correspondingly. 
	// When measureDist=false , the return value is +1, -1 and 0, respectively. 
	// Otherwise, the return value it is a signed distance between the point 
	// and the nearest contour edge." - OpenCV Manual version 2.1

	// create an image and display the image

	namedWindow("Polygon Test", 0);
	imshow("Polygon Test", img);
	waitKey(0);

	return 0;
}

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
	Mat matRotation = getRotationMatrix2D(Point(210, 175), 3, 1);
	printf("Rotation Mat \n");
	printfMat<double> ( matRotation );
	Mat matRotated;
	Mat src = cvarrToMat ( pGrayPhoto );
	warpAffine(src, matRotated, matRotation, src.size());
	cv::line(matRotated, Point(210, 175), Point(260, 225), Scalar(255,255, 0), 2 );		//Manually add defect.
	//cv::ellipse(matRotated, cv::RotatedRect(Point2f(150,150), Size2f(20,30),30 ), Scalar(255,255, 0), 0, FILLED );
	circle(matRotated, Point(240, 190), 15, Scalar(255,255, 0), CV_FILLED);

    circle(matRotated, Point(200, 160), 15, Scalar(0,0, 0), CV_FILLED);
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
	stLrnTmplCmd.mat = cvarrToMat(pImgT, true);
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
	Rect rectSrchWindow ( 30, 30, stFindObjCmd.mat.cols - 30, stFindObjCmd.mat.rows - 30 );
	stFindObjCmd.rectSrchWindow = rectSrchWindow;
	stFindObjCmd.enAlgorithm = enAlgorithm;

	CStopWatch stopWatch;
	visionAlgorithm.findObject(&stFindObjCmd, &stFindObjRpy);
	printf("SURF result, time consume %d us \n", stopWatch.NowInMicro() );
	printfMat<double>(stFindObjRpy.matHomography);

	PR_InspCmd stInspCmd;
	PR_InspRpy stInspRpy;
	stInspCmd.matTmpl = cvarrToMat(pImgT, true);
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
	Mat matT = cvarrToMat(pImgT);
	stAlignCmd.matTmpl = Mat(matT, omega);
	visionAlgorithm.align(&stAlignCmd, &stAlignRpy);

	printf("SURF and findTransformECC result \n");
	printfMat<float>(stAlignRpy.matAffine);

	/* Lucas-Kanade */
	//align_image_forwards_additive(pImgT, omega, pImgI);

	//Mat matT = cvarrToMat ( pImgT );
	Mat matLearn(matT, omega);
	//Mat matI = cvarrToMat(pImgI);
	Mat resultW(2, 3, CV_32F);
	resultW.setTo(Scalar(0));
	resultW.at<float>(0, 0) = 1.0;
	resultW.at<float>(1, 1) = 1.0;
	//resultW.at<float>(2,2) = 1.0;
	findTransformECC(matT, matRotated, resultW, MOTION_AFFINE);

	cv::Mat matOriginalPos(3, 1, CV_32F);
	matOriginalPos.at<float>(0, 0) = omega.x + omega.width / 2.0f;
	matOriginalPos.at<float>(1, 0) = omega.y + omega.height / 2.0f;
	matOriginalPos.at<float>(2, 0) = 1.0;

	Mat matDestPos(3, 1, CV_32F);
	matDestPos = resultW * matOriginalPos;

	Point2d pt2dDest;
	pt2dDest.x = (float)matDestPos.at<float>(0, 0);
	pt2dDest.y = (float)matDestPos.at<float>(1, 0);

	printf("findTransformECC result \n");
	printfMat<float>(resultW);

	Mat resultW1 = estimateRigidTransform(matT, matRotated, false);
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

int TestHoughTransform()
{
    Mat src, dst, color_dst;

	src = imread(".\\data\\building.jpg", 0 );
    if( src.empty() )
        return -1;

    Canny( src, dst, 50, 200, 3 );
    cvtColor( dst, color_dst, CV_GRAY2BGR );

#if 0
    vector<Vec2f> lines;
    HoughLines( dst, lines, 1, CV_PI/180, 100 );

    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0];
        float theta = lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        Point pt1(cvRound(x0 + 1000*(-b)),
                  cvRound(y0 + 1000*(a)));
        Point pt2(cvRound(x0 - 1000*(-b)),
                  cvRound(y0 - 1000*(a)));
        line( color_dst, pt1, pt2, Scalar(0,0,255), 3, 8 );
    }
#else
    vector<Vec4i> lines;
    HoughLinesP( dst, lines, 1, CV_PI/180, 80, 30, 10 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        line( color_dst, Point(lines[i][0], lines[i][1]),
            Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
    }
#endif
    namedWindow( "Source", 1 );
    imshow( "Source", src );

    namedWindow( "Detected Lines", 1 );
    imshow( "Detected Lines", color_dst );

    waitKey(0);
    return 0;
}

int TestFindBlob()
{
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.blobColor = 0;

	params.minThreshold = 0;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 50;

	// Filter by Circularity
	//params.filterByCircularity = true;
	//params.minCircularity = 0.1;

	// Filter by Convexity
	//params.filterByConvexity = true;
	//params.minConvexity = 0.87;

	// Filter by Inertia
	//params.filterByInertia = true;
	//params.minInertiaRatio = 0.01;

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params); 

    //Mat matDefect = cv::imread( ".\\data\\Defect.jpg");
    //if ( matDefect.empty() )
    //     return -1;
	IplImage *pImage = cvLoadImage ( ".\\data\\Defect.jpg");
	if ( NULL == pImage )
		return -1;

	IplImage *pImageGray = cvCreateImage ( cvGetSize ( pImage ), 8, 1 );
	cvCvtColor( pImage, pImageGray, CV_BGR2GRAY);

	Mat matDefect = cvarrToMat( pImageGray );
	//
	//matDefect.convertTo( matDefect, CV_BGR2GRAY );

    cv::imshow("matDefect", matDefect);

	vector<Mat> vecInput;
	vecInput.push_back(matDefect);

	VectorOfVectorKeyPoint keyPoints;
	detector->detect( vecInput, keyPoints);

	Mat im_with_keypoints;
	drawKeypoints( matDefect, keyPoints[0], im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

	cv::imshow("im_with_keypoints", im_with_keypoints);
    cvWaitKey(0);

	cv::Mat matCannyResult, color_dst;
	Canny( matDefect, matCannyResult, 0, 50, 3 );

	imshow( "Canny Result", matCannyResult );
    cvtColor( matCannyResult, color_dst, CV_GRAY2BGR );

#if 0
    vector<Vec2f> lines;
    HoughLines( dst, lines, 1, CV_PI/180, 100 );

    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0];
        float theta = lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        Point pt1(cvRound(x0 + 1000*(-b)),
                  cvRound(y0 + 1000*(a)));
        Point pt2(cvRound(x0 - 1000*(-b)),
                  cvRound(y0 - 1000*(a)));
        line( color_dst, pt1, pt2, Scalar(0,0,255), 3, 8 );
    }
#else
    vector<Vec4i> lines;
    HoughLinesP( matCannyResult, lines, 1, CV_PI/180, 0, 30, 10 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        line( color_dst, Point(lines[i][0], lines[i][1]),
            Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
    }
#endif
    imshow( "Detected Lines", color_dst );
	waitKey(0);

	return 0;
}

int TestFindLine()
{
	IplImage *img = cvLoadImage(".\\data\\Defect.jpg", CV_LOAD_IMAGE_COLOR);
	if (NULL == img)
		return -1;

	cvShowImage("Origianl image", img);

	IplImage *imgGrayScale = cvCreateImage(cvGetSize(img), 8, 1);
	cvCvtColor(img, imgGrayScale, CV_BGR2GRAY);

	cvThreshold(imgGrayScale, imgGrayScale, 128, 255, CV_THRESH_BINARY);
	cvShowImage("Threshold image", imgGrayScale);

	Mat matGray = cvarrToMat(imgGrayScale);

	Mat matSobelX, matSobelY;
	Sobel(matGray, matSobelX, matGray.depth(), 1, 0);
	Sobel(matGray, matSobelY, matGray.depth(), 0, 1);

	short depth = matSobelX.depth();

	Mat matSobelResult(matGray.size(), CV_8U);
	for (short row = 0; row < matGray.rows; ++row)
	{
		for (short col = 0; col < matGray.cols; ++col)
		{
			unsigned char x = (matSobelX.at<unsigned char>(row, col));
			unsigned char y = (matSobelY.at<unsigned char>(row, col));
			matSobelResult.at<unsigned char>(row, col) = x + y;
		}
	}

	imshow("Sobel Result", matSobelResult);
	cv::Mat color_dst;
	cvtColor(matSobelResult, color_dst, CV_GRAY2BGR);

	vector<Vec4i> lines;
	HoughLinesP(matSobelResult, lines, 1, CV_PI / 180, 30, 30, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		line(color_dst, Point(lines[i][0], lines[i][1]),
			Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
	}
	imshow("Detected Lines", color_dst);
	waitKey(0);
	return 0;
}

int main()
{
	//MaCvCapture* capture;
	//Mat frame;

	//capture = cvCaptureFromCAM(0);
	//frame = cvQueryFrame(capture);

	//imshow("Video", frame);

	//VideoCapture cap(0); // open the default camera
	//if (!cap.isOpened())  // check if we succeeded
	//	return -1;

	//Mat edges;
	//namedWindow("edges", 1);
	//for (;;)
	//{
	//	Mat frame;
	//	cap >> frame; // get a new frame from camera
	//	cvtColor(frame, edges, COLOR_BGR2GRAY);
	//	GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
	//	Canny(edges, edges, 0, 30, 3);
	//	imshow("edges", edges);
	//	if (waitKey(30) >= 0) break;
	//}
	//TestErode();

	//TestHistogramEqualizeGray();
	//TestHistogramEqualizeColor();
	//TestImageBlur();

	//TestTrackBar();

	//TestTrackBarCallBack();
	//TestRotateImage();

	//TestDetectObject();
	//TestDetectRedObject();
	//TestFindContour();

	//TestFindTriangle();

	//TestCaptureImageFromCam();

	//TrackOnVideo ();
	//TestCSDNCode();

	//TestConvertScale();
	//GenerateTransparentMask();

	//TestPolyline();

	//TestImageZoom();

	TestVisionAlgorithm();

	//TestFindBlob();
	//TestFindLine();

	//TestHoughTransform();

    return 0;
}