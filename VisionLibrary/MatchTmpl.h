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
    static VisionStatus refineSrchTemplate(const cv::Mat &mat, const cv::Mat &matTmpl, PR_OBJECT_MOTION enMotion, cv::Point2f &ptResult, float &fRotation, float &fCorrelation);
    static VisionStatus matchTemplate(const cv::Mat &mat, const cv::Mat &matTmpl, bool bSubPixelRefine, PR_OBJECT_MOTION enMotion, cv::Point2f &ptResult, float &fRotation, float &fCorrelation);
    static cv::Point myMatchTemplate(const cv::Mat &mat, const cv::Mat &matTmpl);
    static cv::Point matchTemplateRecursive(const cv::Mat &matInput, const cv::Mat &matTmpl);
};

}
}

#endif
