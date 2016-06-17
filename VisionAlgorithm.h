#ifndef _AOI_VISION_ALGORITHM_H
#define _AOI_VISION_ALGORITHM_H

#include "BaseType.h"
#include "opencv2/core/core.hpp"
#include "VisionDataStruct.h"
#include "StopWatch.h"

#include <list>
#include <vector>
#include <mutex>
#include <thread>
using namespace std;

//using namespace cv;

namespace AOI
{
namespace Vision
{

typedef std::vector<cv::DMatch> VectorOfDMatch;
typedef std::vector<VectorOfDMatch> VectorOfVectorOfDMatch;
typedef std::vector<cv::KeyPoint> VectorOfKeyPoint;
typedef std::vector<VectorOfKeyPoint> VectorOfVectorKeyPoint;

#define RADIAN_2_DEGREE(RADIAN)	( RADIAN * 180.f / CV_PI )
#define DEGREE_2_RADIAN(DEGREE) ( DEGREE * CV_PI / 180.f )

class VisionAlgorithm : private Uncopyable
{
public:
    explicit VisionAlgorithm(const cv::Mat &mat);
    explicit VisionAlgorithm();
    int learn(const cv::Rect &rect, const cv::Mat &mask, int const nAlgorithm);
    int learn(PR_LearnTmplCmd * const pLearnTmplCmd, PR_LearnTmplRpy *pLearnTmplRpy);
    int findObject(PR_FindObjCmd *const pFindObjCmd, PR_FindObjRpy *pFindObjRpy);
    int align(PR_AlignCmd *const pAlignCmd, PR_AlignRpy *pAlignRpy);
	int inspect(PR_InspCmd *const pInspCmd, PR_InspRpy *pInspRpy);
	float distanceOf2Point(const cv::Point &pt1, const cv::Point &pt2);
	float distanceOf2Point(const cv::Point2f &pt1, const cv::Point2f &pt2);
    void dumpTimeLog(const std::string &strFilePath);

protected:
	int _findBlob(const cv::Mat &mat, const cv::Mat &matRevs, PR_InspCmd *const pInspCmd, PR_InspRpy *pInspRpy );
	int _findLine(const cv::Mat &mat, PR_InspCmd *const pInspCmd, PR_InspRpy *pInspRpy );
	int _mergeLines(const vector<PR_Line2f> &vecLines, vector<PR_Line2f> &vecResultLines);
	int _merge2Line(const PR_Line2f &line1, const PR_Line2f &line2, PR_Line2f &lineResult);
	int _findLineCrossPoint(const PR_Line2f &line1, const PR_Line2f &line2, cv::Point2f ptResult);
    void _addTimeLog(const std::string &strMsg);
    void _addTimeLog(const std::string &strMsg, __int64 nTimeSpan);
protected:
    const int       _constMinHessian    =       300;
    const int       _constOctave        =       4;
    const int       _constOctaveLayer   =       3;
    const UInt32    _constLeastFeatures =       3;
    cv::Mat         _mat;
    PR_DEBUG_MODE   _enDebugMode = PR_DEBUG_MODE::DISABLED;
    std::mutex      _mutexTimeLog;
    StringVector    _vecStringTimeLog;
    CStopWatch      _stopWatch;
};

}
}
#endif