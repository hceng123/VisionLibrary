// Module: auxfunc.cpp
// Brief:  Contains implementation of auxilary functions.
// Author: Oleg A. Krivtsov
// Email:  olegkrivtsov@mail.ru
// Date:  March 2008

#include "auxfunc.h"

// Our warp matrix looks like this one:
//
//  ! 1  -wz  tx !
//  ! wz  1   ty !
//  ! 0   0   1  !
//

void init_warp(CvMat* W, float wz, float tx, float ty)
{
	CV_MAT_ELEM(*W, float, 0, 0) = 1;
	CV_MAT_ELEM(*W, float, 1, 0) = wz;
	CV_MAT_ELEM(*W, float, 2, 0) = 0;

	CV_MAT_ELEM(*W, float, 0, 1) = -wz;
	CV_MAT_ELEM(*W, float, 1, 1) = 1;
	CV_MAT_ELEM(*W, float, 2, 1) = 0;

	CV_MAT_ELEM(*W, float, 0, 2) = tx;
	CV_MAT_ELEM(*W, float, 1, 2) = ty;
	CV_MAT_ELEM(*W, float, 2, 2) = 1;
}



void warp_image(IplImage* pSrcFrame, IplImage* pDstFrame, CvMat* W)
{
	cvSet(pDstFrame, cvScalar(0));

	CvMat* X = cvCreateMat(3, 1, CV_32F);
	CvMat* Z = cvCreateMat(3, 1, CV_32F);
	int x, y;

	for(x=0;x<pSrcFrame->width; x++)
	{
		for(y=0;y<pSrcFrame->height; y++)
		{
			SET_VECTOR(X, x, y);

			cvGEMM(W, X, 1, 0, 0, Z);

			int x2, y2;
			GET_INT_VECTOR(Z, x2, y2);

			if(x2>=0 && x2<pDstFrame->width &&
				y2>=0 && y2<pDstFrame->height)
			{
				CV_IMAGE_ELEM(pDstFrame, uchar, y2, x2) = 
					CV_IMAGE_ELEM(pSrcFrame, uchar, y, x);
			}
		}
	}

	//cvSmooth(pDstFrame, pDstFrame);

	cvReleaseMat(&X);
	cvReleaseMat(&Z);
}

void draw_warped_rect(IplImage* pImage, CvRect rect, CvMat* W)
{
	CvPoint lt, lb, rt, rb;
	
	CvMat* X = cvCreateMat(3, 1, CV_32F);
	CvMat* Z = cvCreateMat(3, 1, CV_32F);

	// left-top point
	SET_VECTOR(X, rect.x, rect.y);
	cvGEMM(W, X, 1, 0, 0, Z);
	GET_INT_VECTOR(Z, lt.x, lt.y);

	// left-bottom point
	SET_VECTOR(X, rect.x, rect.y+rect.height);
	cvGEMM(W, X, 1, 0, 0, Z);
	GET_INT_VECTOR(Z, lb.x, lb.y);

	// right-top point
	SET_VECTOR(X, rect.x+rect.width, rect.y);
	cvGEMM(W, X, 1, 0, 0, Z);
	GET_INT_VECTOR(Z, rt.x, rt.y);

	// right-bottom point
	SET_VECTOR(X, rect.x+rect.width, rect.y+rect.height);
	cvGEMM(W, X, 1, 0, 0, Z);
	GET_INT_VECTOR(Z, rb.x, rb.y);

	// draw rectangle
	cvLine(pImage, lt, rt, cvScalar(255));
	cvLine(pImage, rt, rb, cvScalar(255));
	cvLine(pImage, rb, lb, cvScalar(255));
	cvLine(pImage, lb, lt, cvScalar(255));

	// release resources and exit
	cvReleaseMat(&X);
	cvReleaseMat(&Z);
}

