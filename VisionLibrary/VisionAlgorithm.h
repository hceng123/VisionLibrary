#ifndef _AOI_VISION_ALGORITHM_H
#define _AOI_VISION_ALGORITHM_H

#include "BaseType.h"
#include "opencv2/core.hpp"
#include "opencv2/text.hpp"
#include "VisionHeader.h"
#include "StopWatch.h"
#include <list>
#include <memory>

using namespace std;

namespace AOI
{
namespace Vision
{
class VisionAlgorithm;

using VisionAlgorithmPtr = std::unique_ptr<VisionAlgorithm>;
using OcrTesseractPtr = cv::Ptr<cv::text::OCRTesseract>;

class VisionAlgorithm : private Uncopyable
{
public:
    enum CalibPattern { NOT_EXISTING, CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

    explicit VisionAlgorithm();
    static unique_ptr<VisionAlgorithm> create();
    VisionStatus lrnTmpl(PR_LRN_OBJ_CMD * const pLrnTmplCmd, PR_LRN_TMPL_RPY *pLrnTmplRpy, bool bReplay = false);
    static VisionStatus srchObj(PR_SRCH_OBJ_CMD *const pFindObjCmd, PR_SRCH_OBJ_RPY *pFindObjRpy);
    VisionStatus lrnDevice(PR_LRN_DEVICE_CMD *pstLrnDeviceCmd, PR_LRN_DEVICE_RPY *pstLrnDeivceRpy);
    VisionStatus inspDevice(PR_INSP_DEVICE_CMD *pstInspDeviceCmd, PR_INSP_DEVICE_RPY *pstInspDeivceRpy);
    static VisionStatus matchTemplate(PR_MATCH_TEMPLATE_CMD *const pstCmd, PR_MATCH_TEMPLATE_RPY *pstRpy, bool bReplay = false);
	VisionStatus inspSurface(PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy);
    static void showImage(String windowName, const cv::Mat &mat);
    static VisionStatus runLogCase(const std::string &strPath);    
    VisionStatus srchFiducialMark(PR_SRCH_FIDUCIAL_MARK_CMD *pstCmd, PR_SRCH_FIDUCIAL_MARK_RPY *pstRpy, bool bReplay = false);
    static VisionStatus fitLine(PR_FIT_LINE_CMD *pstCmd, PR_FIT_LINE_RPY *pstRpy, bool bReplay = false);
    static VisionStatus detectLine(PR_DETECT_LINE_CMD *pstCmd, PR_DETECT_LINE_RPY *pstRpy, bool bReplay = false);
    VisionStatus fitParallelLine(PR_FIT_PARALLEL_LINE_CMD *pstCmd, PR_FIT_PARALLEL_LINE_RPY *pstRpy, bool bReplay = false);
    VisionStatus fitRect(PR_FIT_RECT_CMD *pstCmd, PR_FIT_RECT_RPY *pstRpy, bool bReplay = false);
    VisionStatus findEdge(PR_FIND_EDGE_CMD *pstCmd, PR_FIND_EDGE_RPY *pstRpy);
    static VisionStatus fitCircle(PR_FIT_CIRCLE_CMD *pstCmd, PR_FIT_CIRCLE_RPY *pstRpy, bool bReplay = false);
    static VisionStatus ocr(PR_OCR_CMD *pstCmd, PR_OCR_RPY *pstRpy, bool bReplay = false);
    static VisionStatus colorToGray(PR_COLOR_TO_GRAY_CMD *pstCmd, PR_COLOR_TO_GRAY_RPY *pstRpy);
    static VisionStatus filter(PR_FILTER_CMD *pstCmd, PR_FILTER_RPY *pstRpy);
    static VisionStatus autoThreshold(PR_AUTO_THRESHOLD_CMD *pstCmd, PR_AUTO_THRESHOLD_RPY *pstRpy, bool bReplay = false);
    static VisionStatus removeCC(PR_REMOVE_CC_CMD *pstCmd, PR_REMOVE_CC_RPY *pstRpy, bool bReplay = false);
    static VisionStatus detectEdge(PR_DETECT_EDGE_CMD *pstCmd, PR_DETECT_EDGE_RPY *pstRpy, bool bReplay = false);
    static VisionStatus circleRoundness(PR_CIRCLE_ROUNDNESS_CMD *pstCmd, PR_CIRCLE_ROUNDNESS_RPY *pstRpy, bool bReplay = false);
    static VisionStatus fillHole(PR_FILL_HOLE_CMD *pstCmd, PR_FILL_HOLE_RPY *pstRpy, bool bReplay = false);
    static VisionStatus pickColor(PR_PICK_COLOR_CMD *pstCmd, PR_PICK_COLOR_RPY *pstRpy, bool bReplay = false);
    static VisionStatus calibrateCamera(const PR_CALIBRATE_CAMERA_CMD *const pstCmd, PR_CALIBRATE_CAMERA_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus restoreImage(const PR_RESTORE_IMG_CMD *const pstCmd, PR_RESTORE_IMG_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calcUndistortRectifyMap(const PR_CALC_UNDISTORT_RECTIFY_MAP_CMD *const pstCmd, PR_CALC_UNDISTORT_RECTIFY_MAP_RPY *const pstRpy);
    static VisionStatus autoLocateLead(const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd, PR_AUTO_LOCATE_LEAD_RPY *const pstRpy, bool bReplay = false );
    static VisionStatus inspBridge(const PR_INSP_BRIDGE_CMD *const pstCmd, PR_INSP_BRIDGE_RPY *const pstRpy, bool bReplay = false );
protected:
	int _findBlob(const cv::Mat &mat, const cv::Mat &matRevs, PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy );
	int _findLine(const cv::Mat &mat, PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy );
	int _mergeLines(const vector<PR_Line2f> &vecLines, vector<PR_Line2f> &vecResultLines);
	int _merge2Line(const PR_Line2f &line1, const PR_Line2f &line2, PR_Line2f &lineResult);
	int _findLineCrossPoint(const PR_Line2f &line1, const PR_Line2f &line2, cv::Point2f ptResult);
    static std::vector<Int16> _autoMultiLevelThreshold(const cv::Mat &matInput, const cv::Mat &matMask, int N);
    static int _autoThreshold(const cv::Mat &mat, const cv::Mat &matMask = cv::Mat() );
    VisionStatus _findDeviceElectrode(const cv::Mat &matDeviceROI, VectorOfPoint &vecElectrodePos);
    VisionStatus _findDeviceEdge(const cv::Mat &matDeviceROI, const cv::Size2f &size, cv::Rect &rectDeviceResult);
    VisionStatus _writeLrnTmplRecord(PR_LRN_TMPL_RPY *pLrnTmplRpy);
    VisionStatus _writeDeviceRecord(PR_LRN_DEVICE_RPY *pLrnDeviceRpy);
    static VisionStatus _matchTemplate     (const cv::Mat &mat, const cv::Mat &matTmpl, PR_OBJECT_MOTION enMotion, cv::Point2f &ptResult, float &fRotation);
    static VisionStatus _refineSrchTemplate(const cv::Mat &mat, const cv::Mat &matTmpl, PR_OBJECT_MOTION enMotion, cv::Point2f &ptResult, float &fRotation);
    static VectorOfPoint _findPointInRegionOverThreshold(const cv::Mat &mat, const cv::Rect &rect, int nThreshold);
    static ListOfPoint _findPointsInRegionByThreshold(const cv::Mat &mat, const cv::Rect &rect, int nThreshold, PR_OBJECT_ATTRIBUTE enAttribute = PR_OBJECT_ATTRIBUTE::BRIGHT);
    static ListOfPoint _findPointsInRegion(const cv::Mat &mat, const cv::Rect &rect);
    static std::vector<size_t> _findPointOverLineTol(const VectorOfPoint   &vecPoint,
                                                     bool                   bReversedFit,
                                                     const float            fSlope,
                                                     const float            fIntercept,
                                                     PR_RM_FIT_NOISE_METHOD method,
                                                     float                  tolerance);
    static std::vector<ListOfPoint::const_iterator> _findPointOverLineTol(const ListOfPoint   &vecPoint,
                                                     bool                   bReversedFit,
                                                     const float            fSlope,
                                                     const float            fIntercept,
                                                     PR_RM_FIT_NOISE_METHOD method,
                                                     float                  tolerance);
    static VectorOfPoint _randomSelectPoints(const VectorOfPoint &vecPoints, int numOfPtToSelect);
    static cv::RotatedRect _fitCircleRansac(const VectorOfPoint &vecPoints, float tolerance, int maxRansacTime, size_t nFinishThreshold);
    static VectorOfPoint _findPointsInCircleTol( const VectorOfPoint &vecPoints, const cv::RotatedRect &rotatedRect, float tolerance );
    static std::vector<ListOfPoint::const_iterator> _findPointsOverCircleTol( const ListOfPoint &listPoint, const cv::RotatedRect &rotatedRect, PR_RM_FIT_NOISE_METHOD enMethod, float tolerance );
    static cv::RotatedRect _fitCircleIterate(const VectorOfPoint &vecPoints, PR_RM_FIT_NOISE_METHOD method, float tolerance);
    static VisionStatus _fillHoleByContour(const cv::Mat &matInput, cv::Mat &matOutput, PR_OBJECT_ATTRIBUTE enAttribute);
    static VisionStatus _fillHoleByMorph(const cv::Mat      &matInput,
                                         cv::Mat            &matOutput,
                                         PR_OBJECT_ATTRIBUTE enAttribute,
                                         cv::MorphShapes     enMorphShape,
                                         cv::Size            szMorphKernel,
                                         Int16               nMorphIteration);
    static VisionStatus _findChessBoardCorners (const cv::Mat   &mat,
                                                const cv::Mat   &matTmpl,
                                                cv::Mat         &matCornerPointsImg,
                                                const cv::Point &startPoint,
                                                float            fStepSize,
                                                int              nSrchSize,
                                                const cv::Size  &szBoardPattern,
                                                VectorOfPoint2f &dvecVecCorners);
    static float _findChessBoardBlockSize( const cv::Mat &matInput, const cv::Size szBoardPattern );
    static void _calcBoardCornerPositions(cv::Size boardSize, float squareSize, std::vector<cv::Point3f> &corners, CalibPattern patternType = CHESSBOARD );
    static cv::Point2f _findFirstChessBoardCorner(const cv::Mat &matInput, float fBlockSize);
    static VisionStatus _inspBridgeItem(const cv::Mat &matGray, cv::Mat &matResult, const PR_INSP_BRIDGE_CMD::INSP_ITEM &inspItem, PR_INSP_BRIDGE_RPY::ITEM_RESULT &inspResult);
protected:
    static const int       _constMinHessian        = 300;
    static const int       _constOctave            = 4;
    static const int       _constOctaveLayer       = 3;
    static const UInt32    _constLeastFeatures     = 3;
    static const String    _strRecordLogPrefix;
    static const float     _constExpSmoothRatio;

    static const cv::Scalar _constRedScalar;
    static const cv::Scalar _constBlueScalar;
    static const cv::Scalar _constGreenScalar;
    static OcrTesseractPtr  _ptrOcrTesseract;
private:
    VisionAlgorithmPtr _pInstance = nullptr;
};

}
}
#endif