#ifndef _AOI_VISION_ALGORITHM_H
#define _AOI_VISION_ALGORITHM_H

#include "BaseType.h"
#include "opencv2/core/core.hpp"
#include "VisionHeader.h"
#include "StopWatch.h"
#include <list>
#include <memory>

using namespace std;

namespace AOI
{
namespace Vision
{

class VisionAlgorithm : private Uncopyable
{
public:
    explicit VisionAlgorithm();
    static shared_ptr<VisionAlgorithm> create();
    VisionStatus lrnTmpl(PR_LRN_TMPL_CMD * const pLrnTmplCmd, PR_LRN_TMPL_RPY *pLrnTmplRpy, bool bReplay = false);
    VisionStatus srchTmpl(PR_SRCH_TMPL_CMD *const pFindObjCmd, PR_SRCH_TMPL_RPY *pFindObjRpy);
    VisionStatus lrnDevice(PR_LRN_DEVICE_CMD *pstLrnDeviceCmd, PR_LRN_DEVICE_RPY *pstLrnDeivceRpy);
    VisionStatus inspDevice(PR_INSP_DEVICE_CMD *pstInspDeviceCmd, PR_INSP_DEVICE_RPY *pstInspDeivceRpy);
    int align(PR_AlignCmd *const pAlignCmd, PR_AlignRpy *pAlignRpy);
	VisionStatus inspSurface(PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy);
    void showImage(String windowName, const cv::Mat &mat);
    VisionStatus runLogCase(const std::string &strPath);
    VisionStatus matchTemplate(const cv::Mat &mat, cv::Mat &matTmpl, cv::Point2f &ptResult);
    VisionStatus srchFiducialMark(PR_SRCH_FIDUCIAL_MARK_CMD *pstCmd, PR_SRCH_FIDUCIAL_MARK_RPY *pstRpy);
    VisionStatus fitLine(PR_FIT_LINE_CMD *pstCmd, PR_FIT_LINE_RPY *pstRpy);
protected:
	int _findBlob(const cv::Mat &mat, const cv::Mat &matRevs, PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy );
	int _findLine(const cv::Mat &mat, PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy );
	int _mergeLines(const vector<PR_Line2f> &vecLines, vector<PR_Line2f> &vecResultLines);
	int _merge2Line(const PR_Line2f &line1, const PR_Line2f &line2, PR_Line2f &lineResult);
	int _findLineCrossPoint(const PR_Line2f &line1, const PR_Line2f &line2, cv::Point2f ptResult);
    int _autoThreshold(const cv::Mat &mat);
    VisionStatus _findDeviceElectrode(const cv::Mat &matDeviceROI, VectorOfPoint &vecElectrodePos);
    VisionStatus _findDeviceEdge(const cv::Mat &matDeviceROI, const cv::Size2f &size, cv::Rect &rectDeviceResult);
    VisionStatus _writeLrnTmplRecord(PR_LRN_TMPL_RPY *pLrnTmplRpy);
    VisionStatus _writeDeviceRecord(PR_LRN_DEVICE_RPY *pLrnDeviceRpy);
    VisionStatus _refineSrchTemplate(const cv::Mat &mat, cv::Mat &matTmpl, cv::Point2f &ptResult);
    static VectorOfPoint _findPointInRegionOverThreshold(const cv::Mat &mat, const cv::Rect &rect, int nThreshold);
    static std::vector<size_t> _findPointOverTol(const VectorOfPoint      &vecPoint,
                                                 const float               fSlope,
                                                 const float               fIntercept,
                                                 PR_RM_FIT_NOISE_METHOD    method,
                                                 float                     tolerance);
protected:
    const int       _constMinHessian        =       300;
    const int       _constOctave            =       4;
    const int       _constOctaveLayer       =       3;
    const UInt32    _constLeastFeatures     =       3;
    const String    _strRecordLogPrefix     =       "tmplDir.";
    const float     _constExpSmoothRatio    =       0.3f;
    CStopWatch      _stopWatch;
};

}
}
#endif