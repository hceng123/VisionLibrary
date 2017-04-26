#ifndef _VISION_REGRESSION_TEST_UTILITY_FUNCTION_H_
#define _VISION_REGRESSION_TEST_UTILITY_FUNCTION_H_

#include "../VisionLibrary/VisionAPI.h"

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

#endif /*_VISION_REGRESSION_TEST_UTILITY_FUNCTION_H_*/