#ifndef _UTILITY_FUNCTION_H_
#define _UTILITY_FUNCTION_H_

#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2\core\mat.hpp"

using namespace cv;
using namespace std;

template <class T>
float interpolate(IplImage* pImage, float x, float y);

// Interpolates pixel intensity with subpixel accuracy.
// Abount bilinear interpolation in Wikipedia:
// http://en.wikipedia.org/wiki/Bilinear_interpolation
template <class T>
float interpolate(const Mat &mat, float x, float y)
{
    // Get the nearest integer pixel coords (xi;yi).
    int xi = cvFloor(x);
    int yi = cvFloor(y);

    float k1 = x - xi; // Coefficients for interpolation formula.
    float k2 = y - yi;

    bool b1 = xi < mat.cols - 1;  // Check that pixels to the right  
    bool b2 = yi < mat.rows - 1; // and to down direction exist.

    float UL = mat.at<T>(Point(xi, yi));
    float UR = b1 ? mat.at<T>( Point (xi + 1, yi ) ) : 0.f;
    float LL = b2 ? mat.at<T>( Point ( xi, yi + 1) ) : 0.f;
    float LR = b1 & b2 ? mat.at <T>( Point ( xi + 1, yi + 1 ) ) : 0.f;

    // Interpolate pixel intensity.
    float interpolated_value = (1.0f - k1) * (1.0f - k2) * UL + k1 * (1.0f - k2) * UR +
        (1.0f - k1) * k2 * LL + k1 * k2 * LR;

    return interpolated_value;
}

//Get the intensity along an input line
template <class T>
int GetIntensityOnLine ( const Mat &mat, const Point &start, const Point &end, vector<float> &vecOutput )
{
    if ( start.x >= mat.cols || start.y >= mat.rows )
        return -1;
    if ( end.x >= mat.cols || end.y >= mat.rows )
        return -1;

    float fLineLen = (float)sqrt ( ( end.x - start.x ) * ( end.x - start.x ) + ( end.y - start.y ) * ( end.y - start.y ) );
    if ( fLineLen < 1 )
        return -1;

    float fCos = ( end.x - start.x ) / fLineLen;
    float fSin = ( end.y - start.y ) / fLineLen;

    float fCurrentLen = 0.f;
    while ( fCurrentLen < fLineLen )    {
        float fX = start.x + fCos * fCurrentLen;
        float fY = start.y + fSin * fCurrentLen;
        float fIntensity = interpolate<T> ( mat, fX, fY );
        vecOutput.push_back ( fIntensity );
        
        ++ fCurrentLen;
    }

    return 0;
}

#endif