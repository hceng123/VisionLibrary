#include "CalcUtils.h"

namespace AOI
{
namespace Vision
{

/*static*/ double CalcUtils::radian2Degree( double fRadian )
{
    return ( fRadian * 180.f / CV_PI );
}

/*static*/ double CalcUtils::degree2Radian( double fDegree )
{
    return ( fDegree * CV_PI / 180.f );
}

}
}