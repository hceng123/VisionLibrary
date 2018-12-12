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

}
}
#endif


