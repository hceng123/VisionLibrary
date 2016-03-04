#include "stdafx.h"
#include "UtilityFunction.h"

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

  pImage->imageData

	T* row1 = &CV_IMAGE_ELEM(pImage, T, yi, xi);
	T* row2 = &CV_IMAGE_ELEM(pImage, T, yi+1, xi);
				
	// Interpolate pixel intensity.
	float interpolated_value = (1.0f-k1)*(1.0f-k2)*(float)row1[0] +
				(f1 ? ( k1*(1.0f-k2)*(float)row1[1] ):0) +
				(f2 ? ( (1.0f-k1)*k2*(float)row2[0] ):0) +						
				((f1 && f2) ? ( k1*k2*(float)row2[1] ):0) ;

	return interpolated_value;
}

// This is taken from item 6 (pp. 39) of "Effective C++, 3rd Ed." by
// Scott Meyers. Declaring it as a private base class does the trick.
// If you need a virtual destructor, use "boost::noncopyable" instead.
class Uncopyable
{
    Uncopyable(Uncopyable const &);
    Uncopyable &operator=(Uncopyable const &);

protected:
    Uncopyable() {}
    ~Uncopyable() {} /* Intentionally not virtual. */
};