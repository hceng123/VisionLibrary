#ifndef _AUXLIARY_H_
#define _AUXLIARY_H_

#include <math.h>
#include "CalcUtils.hpp"

namespace AOI
{
namespace Vision
{

struct HOUGH_LINE {
    float rho;
    float theta;
    bool operator==(const HOUGH_LINE &houghLine) {
        bool bAngleMatch = fabs(theta - houghLine.theta) < 0.1 ||
            fabs(theta - houghLine.theta - CV_PI) < 0.1 ||
            fabs(theta - houghLine.theta + CV_PI) < 0.1;
        return fabs(fabs(rho) - fabs(houghLine.rho)) < 8 && bAngleMatch;
    }
};

template<class T>
void printfMat(const cv::Mat &mat, int nPrecision = 2)
{
    if (mat.empty()) {
        std::cout << "The matrix to print is empty" << std::endl;
        return;
    }

    std::cout << std::fixed << std::setprecision(nPrecision);
    for (short row = 0; row < mat.rows; ++row)
    {
        for (short col = 0; col < mat.cols; ++col)
        {
            std::cout << mat.at<T>(row, col) << " ";
        }
        std::cout << std::endl;
    }
}

}
}
#endif


