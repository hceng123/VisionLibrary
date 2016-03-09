// Module: auxfunc.h
// Brief:  
// Author: Oleg A. Krivtsov
// Email:  olegkrivtsov@mail.ru
// Date:  March 2008

#ifndef __AUXFUNC_H__
#define __AUXFUNC_H__

#include "opencv\cv.h"

void init_warp(CvMat* W, float wz, float tx, float ty);
void warp_image(IplImage* pSrcFrame, IplImage* pDstFrame, CvMat* W);
void draw_warped_rect(IplImage* pImage, CvRect rect, CvMat* W);

// Interpolates pixel intensity with subpixel accuracy.
// Abount bilinear interpolation in Wikipedia:
// http://en.wikipedia.org/wiki/Bilinear_interpolation

template <class T>
float interpolate(IplImage* pImage, float x, float y)
{
	// Get the nearest integer pixel coords (xi;yi).
	int xi = cvFloor(x);
	int yi = cvFloor(y);

	float k1 = x-xi; // Coefficients for interpolation formula.
	float k2 = y-yi;

	int f1 = xi<pImage->width-1;  // Check that pixels to the right  
	int f2 = yi<pImage->height-1; // and to down direction exist.

	T* row1 = &CV_IMAGE_ELEM(pImage, T, yi, xi);
	T* row2 = &CV_IMAGE_ELEM(pImage, T, yi+1, xi);
				
	// Interpolate pixel intensity.
	float interpolated_value = (1.0f-k1)*(1.0f-k2)*(float)row1[0] +
				(f1 ? ( k1*(1.0f-k2)*(float)row1[1] ):0) +
				(f2 ? ( (1.0f-k1)*k2*(float)row2[0] ):0) +						
				((f1 && f2) ? ( k1*k2*(float)row2[1] ):0) ;

	return interpolated_value;

}

#define SET_VECTOR(X, u, v)\
	CV_MAT_ELEM(*(X), float, 0, 0) = (float)(u);\
	CV_MAT_ELEM(*(X), float, 1, 0) = (float)(v);\
	CV_MAT_ELEM(*(X), float, 2, 0) = 1.0f;

#define GET_VECTOR(X, u, v)\
	(u) = CV_MAT_ELEM(*(X), float, 0, 0);\
	(v) = CV_MAT_ELEM(*(X), float, 1, 0);

#define GET_INT_VECTOR(X, u, v)\
	(u) = (int)CV_MAT_ELEM(*(X), float, 0, 0);\
	(v) = (int)CV_MAT_ELEM(*(X), float, 1, 0);

	
#endif //__AUXFUNC_H__