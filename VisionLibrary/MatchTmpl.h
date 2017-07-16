#ifndef _VISION_MATCH_TMPL_H_
#define _VISION_MATCH_TMPL_H_

#include "VisionHeader.h"

namespace AOI
{
namespace Vision
{

class MatchTmpl
{
public:
    MatchTmpl ();
    ~MatchTmpl ();
    static cv::Point matchByRecursionEdge(const cv::Mat &matInput, const cv::Mat &matTmpl, const cv::Mat &matMaskOfTmpl );
};

}
}

#endif
