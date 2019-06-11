#ifndef _AOI_VISION_ALGORITHM_H_
#define _AOI_VISION_ALGORITHM_H_

#include "BaseType.h"
#include "opencv2/core.hpp"
#include "opencv2/text.hpp"
#include "VisionHeader.h"
#include "StopWatch.h"
#include "LogCase.h"
#include "Record.h"
#include <list>
#include <memory>

namespace AOI
{
namespace Vision
{
class VisionAlgorithm;

using VisionAlgorithmPtr = std::unique_ptr<VisionAlgorithm>;
using OcrTesseractPtr = cv::Ptr<cv::text::OCRTesseract>;

const float ConstToPercentage = 100.f;

class VisionAlgorithm : private Uncopyable
{
public:
    enum CalibPattern { NOT_EXISTING, CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };
    enum BGR_CHANNEL {
        BLUE,
        GREEN,
        RED,
    };

    static const cv::Scalar RED_SCALAR;
    static const cv::Scalar BLUE_SCALAR;
    static const cv::Scalar CYAN_SCALAR;
    static const cv::Scalar GREEN_SCALAR;
    static const cv::Scalar YELLOW_SCALAR;

    explicit VisionAlgorithm();
    static std::unique_ptr<VisionAlgorithm> create();
    static bool isAutoMode();
    static void EnableAutoMode(bool bEnableAutoMode);
    static VisionStatus lrnObj(const PR_LRN_OBJ_CMD *const pstCmd, PR_LRN_OBJ_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus srchObj(const PR_SRCH_OBJ_CMD *const pstCmd, PR_SRCH_OBJ_RPY *const pstRpy, bool bReplay = false);
    VisionStatus lrnDevice(PR_LRN_DEVICE_CMD *pstLrnDeviceCmd, PR_LRN_DEVICE_RPY *pstLrnDeivceRpy);
    VisionStatus inspDevice(PR_INSP_DEVICE_CMD *pstInspDeviceCmd, PR_INSP_DEVICE_RPY *pstInspDeivceRpy);
    static VisionStatus lrnTemplate(const PR_LRN_TEMPLATE_CMD *const pstCmd, PR_LRN_TEMPLATE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus matchTemplate(const PR_MATCH_TEMPLATE_CMD *const pstCmd, PR_MATCH_TEMPLATE_RPY * const pstRpy, bool bReplay = false);
    VisionStatus inspSurface(PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy);
    static void showImage(String windowName, const cv::Mat &mat);
    static VisionStatus runLogCase(const String &strFilePath);   
    static VisionStatus srchFiducialMark(PR_SRCH_FIDUCIAL_MARK_CMD *pstCmd, PR_SRCH_FIDUCIAL_MARK_RPY *pstRpy, bool bReplay = false);
    static VisionStatus fitLine(const PR_FIT_LINE_CMD *const pstCmd, PR_FIT_LINE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus fitLineByPoint(const PR_FIT_LINE_BY_POINT_CMD *const pstCmd, PR_FIT_LINE_BY_POINT_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus findLine(const PR_FIND_LINE_CMD *const pstCmd, PR_FIND_LINE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus fitParallelLine(const PR_FIT_PARALLEL_LINE_CMD *const pstCmd, PR_FIT_PARALLEL_LINE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus fitRect(const PR_FIT_RECT_CMD *const pstCmd, PR_FIT_RECT_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus findEdge(const PR_FIND_EDGE_CMD *const pstCmd, PR_FIND_EDGE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus fitCircle(const PR_FIT_CIRCLE_CMD *const pstCmd, PR_FIT_CIRCLE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus fitCircleByPoint(const PR_FIT_CIRCLE_BY_POINT_CMD *const pstCmd, PR_FIT_CIRCLE_BY_POINT_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus findCircle(const PR_FIND_CIRCLE_CMD *const pstCmd, PR_FIND_CIRCLE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus inspCircle(const PR_INSP_CIRCLE_CMD *const pstCmd, PR_INSP_CIRCLE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus ocr(PR_OCR_CMD *pstCmd, PR_OCR_RPY *pstRpy, bool bReplay = false);
    static VisionStatus colorToGray(PR_COLOR_TO_GRAY_CMD *pstCmd, PR_COLOR_TO_GRAY_RPY *pstRpy);
    static VisionStatus filter(PR_FILTER_CMD *pstCmd, PR_FILTER_RPY *pstRpy);
    static VisionStatus autoThreshold(PR_AUTO_THRESHOLD_CMD *pstCmd, PR_AUTO_THRESHOLD_RPY *pstRpy, bool bReplay = false);
    static VisionStatus removeCC(PR_REMOVE_CC_CMD *pstCmd, PR_REMOVE_CC_RPY *pstRpy, bool bReplay = false);
    static VisionStatus detectEdge(PR_DETECT_EDGE_CMD *pstCmd, PR_DETECT_EDGE_RPY *pstRpy, bool bReplay = false);
    static VisionStatus fillHole(PR_FILL_HOLE_CMD *pstCmd, PR_FILL_HOLE_RPY *pstRpy, bool bReplay = false);
    static VisionStatus pickColor(const PR_PICK_COLOR_CMD *const pstCmd, PR_PICK_COLOR_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calibrateCamera(const PR_CALIBRATE_CAMERA_CMD *const pstCmd, PR_CALIBRATE_CAMERA_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus restoreImage(const PR_RESTORE_IMG_CMD *const pstCmd, PR_RESTORE_IMG_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calcUndistortRectifyMap(const PR_CALC_UNDISTORT_RECTIFY_MAP_CMD *const pstCmd, PR_CALC_UNDISTORT_RECTIFY_MAP_RPY *const pstRpy);
    static VisionStatus autoLocateLead(const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd, PR_AUTO_LOCATE_LEAD_RPY *const pstRpy, bool bReplay = false );
    static VisionStatus inspBridge(const PR_INSP_BRIDGE_CMD *const pstCmd, PR_INSP_BRIDGE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus lrnChip( const PR_LRN_CHIP_CMD *const pstCmd, PR_LRN_CHIP_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus inspChip( const PR_INSP_CHIP_CMD *const pstCmd, PR_INSP_CHIP_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus lrnContour(const PR_LRN_CONTOUR_CMD *const pstCmd, PR_LRN_CONTOUR_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus inspContour(const PR_INSP_CONTOUR_CMD *const pstCmd, PR_INSP_CONTOUR_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus inspHole(const PR_INSP_HOLE_CMD *const pstCmd, PR_INSP_HOLE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus inspLead(const PR_INSP_LEAD_CMD *const pstCmd, PR_INSP_LEAD_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus inspLeadTmpl(const PR_INSP_LEAD_TMPL_CMD *const pstCmd, PR_INSP_LEAD_TMPL_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus inspPolarity(const PR_INSP_POLARITY_CMD *const pstCmd, PR_INSP_POLARITY_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus gridAvgGrayScale(const PR_GRID_AVG_GRAY_SCALE_CMD *const pstCmd, PR_GRID_AVG_GRAY_SCALE_RPY *const pstRpy);
    static VisionStatus calib3DBase(const PR_CALIB_3D_BASE_CMD *const pstCmd, PR_CALIB_3D_BASE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calc3DBase(const PR_CALC_3D_BASE_CMD *const pstCmd, PR_CALC_3D_BASE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calib3DHeight(const PR_CALIB_3D_HEIGHT_CMD *const pstCmd, PR_CALIB_3D_HEIGHT_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus integrate3DCalib(const PR_INTEGRATE_3D_CALIB_CMD *const pstCmd, PR_INTEGRATE_3D_CALIB_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus motorCalib3D(const PR_MOTOR_CALIB_3D_CMD *const pstCmd, PR_MOTOR_CALIB_3D_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus motorCalib3DNew(const PR_MOTOR_CALIB_3D_CMD *const pstCmd, PR_MOTOR_CALIB_3D_NEW_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calibDlpOffset(const PR_CALIB_DLP_OFFSET_CMD *const pstCmd, PR_CALIB_DLP_OFFSET_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calc3DHeight(const PR_CALC_3D_HEIGHT_CMD *const pstCmd, PR_CALC_3D_HEIGHT_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calc3DHeightNew(const PR_CALC_3D_HEIGHT_NEW_CMD *const pstCmd, PR_CALC_3D_HEIGHT_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calc3DHeightGpu(const PR_CALC_3D_HEIGHT_GPU_CMD *const pstCmd, PR_CALC_3D_HEIGHT_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calcMerge4DlpHeight(const PR_CALC_MERGE_4_DLP_HEIGHT_CMD *const pstCmd, PR_CALC_MERGE_4_DLP_HEIGHT_RPY *const pstRpy);
    static VisionStatus merge3DHeight(const PR_MERGE_3D_HEIGHT_CMD *const pstCmd, PR_MERGE_3D_HEIGHT_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calc3DHeightDiff(const PR_CALC_3D_HEIGHT_DIFF_CMD *const pstCmd, PR_CALC_3D_HEIGHT_DIFF_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus insp3DSolder(const PR_INSP_3D_SOLDER_CMD *pstCmd, PR_INSP_3D_SOLDER_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calcDlpOffset(const PR_CALC_DLP_OFFSET_CMD *const pstCmd, PR_CALC_DLP_OFFSET_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calcFrameValue(const PR_CALC_FRAME_VALUE_CMD *const pstCmd, PR_CALC_FRAME_VALUE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calcCameraMTF(const PR_CALC_CAMERA_MTF_CMD *const pstCmd, PR_CALC_CAMERA_MTF_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calcMTF(const PR_CALC_MTF_CMD *const pstCmd, PR_CALC_MTF_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calcPD(const PR_CALC_PD_CMD *const pstCmd, PR_CALC_PD_RPY *const pstRpy, bool bReplay = false);
    static std::vector<Int16> autoMultiLevelThreshold(const cv::Mat &matInputImg, const cv::Mat &matMask, int N);
    static VisionStatus parallelLineDist(const PR_PARALLEL_LINE_DIST_CMD *const pstCmd, PR_PARALLEL_LINE_DIST_RPY *const pstRpy);
    static VisionStatus measureDist(const PR_MEASURE_DIST_CMD* const pstCmd, PR_MEASURE_DIST_RPY* const pstRpy);
    static VisionStatus crossSectionArea(const PR_CROSS_SECTION_AREA_CMD *const pstCmd, PR_CROSS_SECTION_AREA_RPY *const pstRpy);
    static VisionStatus combineImg(const PR_COMBINE_IMG_CMD *const pstCmd, PR_COMBINE_IMG_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus combineImgNew(const PR_COMBINE_IMG_NEW_CMD *const pstCmd, PR_COMBINE_IMG_NEW_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus threshold(const PR_THRESHOLD_CMD *const pstCmd, PR_THRESHOLD_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus heightToGray(const PR_HEIGHT_TO_GRAY_CMD *const pstCmd, PR_HEIGHT_TO_GRAY_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus lrnOcv(const PR_LRN_OCV_CMD *const pstCmd, PR_LRN_OCV_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus ocv(const PR_OCV_CMD *const pstCmd, PR_OCV_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus read2DCode(const PR_READ_2DCODE_CMD *const pstCmd, PR_READ_2DCODE_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus lrnSimilarity(const PR_LRN_SIMILARITY_CMD *const pstCmd, PR_LRN_SIMILARITY_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus inspSimilarity(const PR_INSP_SIMILARITY_CMD *const pstCmd, PR_INSP_SIMILARITY_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus tableMapping(const PR_TABLE_MAPPING_CMD *const pstCmd, PR_TABLE_MAPPING_RPY *const pstRpy, bool bReplay = false);
    static VisionStatus calcRestoreIdx(const PR_CALC_RESTORE_IDX_CMD* const pstCmd, PR_CALC_RESTORE_IDX_RPY* const pstRpy, bool bReplay = false);

protected:
    static VisionStatus _checkInputROI(const cv::Rect &rect, const cv::Mat &matInputImg, const char *filename, int line);
    static VisionStatus _checkMask(const cv::Mat &matInput, const cv::Mat &matMask, const char *filename, int line);
    static LogCasePtr _createLogCaseInstance(const String &strFolderPrefix, const String &strLocalPath);
    int _findBlob(const cv::Mat &mat, const cv::Mat &matRevs, PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy );
    int _findLine(const cv::Mat &mat, PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy );
    static int _mergeLines(const std::vector<PR_Line2f> &vecLines, std::vector<PR_Line2f> &vecResultLines);
    static int _merge2Line(const PR_Line2f &line1, const PR_Line2f &line2, PR_Line2f &lineResult); 
    static int _autoThreshold(const cv::Mat &mat, const cv::Mat &matMask = cv::Mat());
    static bool _isContourRectShape(const VectorOfPoint &vecPoint);
    static bool _isContourRectShapeProjection(const VectorOfPoint &vecPoint);
    static inline float getContourRectRatio( const VectorOfPoint &vecPoint);
    static VisionStatus _findDeviceElectrode(const cv::Mat &matDeviceROI, VectorOfPoint &vecElectrodePos, VectorOfSize2f &vecElectrodeSize, VectorOfVectorOfPoint &vevVecContour );
    static VisionStatus _findDeviceEdge(const cv::Mat &matDeviceROI, const cv::Size2f &size, cv::Rect &rectDeviceResult);
    static VisionStatus _writeLrnObjRecord(PR_LRN_OBJ_RPY *const pstRpy);
    VisionStatus _writeDeviceRecord(PR_LRN_DEVICE_RPY *pLrnDeviceRpy);
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
    static VectorOfPoint _findPointInLineTol(const VectorOfPoint   &vecPoint,
                                             bool                   bReversedFit,
                                             const float            fSlope,
                                             const float            fIntercept,
                                             float                  fTolerance);

    static void _fitLineRansac(const VectorOfPoint &vecPoints,
                               float                fTolerance,
                               int                  nMaxRansacTime,
                               size_t               nFinishThreshold,
                               bool                 bReversedFit,
                               float               &fSlope,
                               float               &fIntercept,
                               PR_Line2f           &stLine);

    static VisionStatus _fitLineRefine(const VectorOfPoint     &vecPoints,
                                       PR_RM_FIT_NOISE_METHOD   enRmNoiseMethod,
                                       float                    fTolerance,
                                       bool                     bReversedFit,
                                       float                   &fSlope,
                                       float                   &fIntercept,
                                       PR_Line2f               &stLine);
    static VectorOfPoint _findPointsOverCircleTol( const VectorOfPoint &vecPoints, const cv::RotatedRect &rotatedRect, float tolerance );
    static VisionStatus _fillHoleByContour(const cv::Mat &matInputImg, cv::Mat &matOutput, PR_OBJECT_ATTRIBUTE enAttribute);
    static VisionStatus _fillHoleByMorph(const cv::Mat      &matInputImg,
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
                                                float            fMinTmplMatchScore,
                                                const cv::Size  &szBoardPattern,
                                                VectorOfPoint2f &vecCorners,
                                                VectorOfPoint   &vecFailedRowCol);
    static float _findChessBoardBlockSize( const cv::Mat &matInputImg, const cv::Size szBoardPattern );
    static void _calcBoardCornerPositions(const cv::Size            &szBoardSize, 
                                          float                     squareSize,
                                          const VectorOfPoint       &vecFailedRowCol,
                                          std::vector<cv::Point3f>  &corners,
                                          CalibPattern               patternType = CHESSBOARD );
    static cv::Point2f _findFirstChessBoardCorner(const cv::Mat &matInputImg, float fBlockSize);
    static void _addBridgeIfNotExist(VectorOfRect& vecBridgeWindow, const cv::Rect& rectBridge);
    static void _inspBridgeItem(const PR_INSP_BRIDGE_CMD * const pstCmd, PR_INSP_BRIDGE_RPY * const pstRpy);
    static VisionStatus _lrnChipHeadMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_LRN_CHIP_RPY *const pstRpy);
    static VisionStatus _lrnChipCAEMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_LRN_CHIP_RPY *const pstRpy);
    static VisionStatus _lrnChipSquareMode(const cv::Mat &matThreshold, const cv::Rect &rectChip, const cv::Rect &rectROI, PR_LRN_CHIP_RPY *const pstRpy );
    static VisionStatus _writeChipRecord(const PR_LRN_CHIP_CMD *const pstCmd, PR_LRN_CHIP_RPY *const pstRpy, const cv::Mat &matTmpl);
    static VisionStatus _inspChipHeadMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy);
    static VisionStatus _inspChipBodyMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy);
    static VisionStatus _inspChipSquareMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy );
    static VisionStatus _inspChipCAEMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy);
    static VisionStatus _inspChipCircularMode(const cv::Mat &matThreshold, const cv::Rect &rectROI, PR_INSP_CHIP_RPY *const pstRpy);
    static bool _findDataUpEdge( const cv::Mat &matRow, int nStart, int nEnd, int &nEdgePos );
    static bool _findDataDownEdge( const cv::Mat &matRow, int nStart, int nEnd, int &nEdgePos );
    static VisionStatus _findLineByProjection(const cv::Mat &matGray, const cv::Mat &matROIMask, const cv::Rect &rectROI, PR_CALIPER_DIR enDetectDir, VectorOfPoint2f &vecFitPoint, PR_FIND_LINE_RPY *const pstRpy);
    static bool _findEdgePosInX( const cv::Mat &matInput, const cv::Mat &matGuassianDiffKernel, PR_CALIPER_DIR enDirection, PR_CALIPER_SELECT_EDGE enSelectEdge, int nEdgeThreshold, bool bFindPair, int &nPos1, int &nPos2);
    static bool _findEdgePosInY( const cv::Mat &matInput, const cv::Mat &matGuassianDiffKernel, PR_CALIPER_DIR enDirection, PR_CALIPER_SELECT_EDGE enSelectEdge, int nEdgeThreshold, bool bFindPair, int &nPos1, int &nPos2);
    static int _findMaxDiffPosInX( const cv::Mat &matInput, const cv::Mat &matGuassianDiffKernel, PR_CALIPER_DIR enDirection );
    static int _findMaxDiffPosInY( const cv::Mat &matInput, const cv::Mat &matGuassianDiffKernel, PR_CALIPER_DIR enDirection );
    static VisionStatus _findLineByCaliper(const cv::Mat &matInputImg, const cv::Mat &matROIMask, const cv::Rect &rectROI, const PR_FIND_LINE_CMD *const pstCmd,
                                           VectorOfPoint2f &vecFitPoint1, VectorOfPoint2f &vecFitPoint2, VectorOfRect &vecSubRects, PR_FIND_LINE_RPY *const pstRpy);
    static VisionStatus _extractRotatedROI(const cv::Mat &matInputImg, const cv::RotatedRect &rectRotatedROI, cv::Mat &matROI );
    static VisionStatus _calcCaliperDirection(const cv::Mat &matROIImg, bool bReverseFit, PR_CALIPER_DIR &enDirection );
    static VisionStatus _writeContourRecord(PR_LRN_CONTOUR_RPY *const pstRpy, const cv::Mat &matTmpl, const cv::Mat &matContour, const VectorOfVectorOfPoint &vecContours);
    static VectorOfPoint _findNearestContour(const cv::Point2f &ptInput, const VectorOfVectorOfPoint &vecContours);
    static void _getContourToContourDistance(const VectorOfPoint &contourInput,
                                             const VectorOfPoint &contourTarget,
                                             float               &fInnerFarthestDist,
                                             float               &fInnerNearestDist,
                                             float               &fOuterFarthestDist,
                                             float               &fOuterNearestDist);
    static cv::Mat _generateContourMask ( const cv::Size &size, const VectorOfVectorOfPoint &vecContours, float fInnerDepth, float fOuterDepth );
    static VisionStatus _segmentImgByGrayScaleRange(const cv::Mat &matInput, const PR_INSP_HOLE_CMD::GRAY_SCALE_RANGE &stGrayScaleRange, cv::Mat &matResult);
    static VisionStatus _setmentImgByColorRange(const cv::Mat &matInput, const PR_INSP_HOLE_CMD::COLOR_RANGE &stColorRange, cv::Mat &matResult);
    static VisionStatus _inspHoleByRatioMode(const cv::Mat &matInput, const cv::Mat &matMask, const PR_INSP_HOLE_CMD::RATIO_MODE_CRITERIA &stCriteria, PR_INSP_HOLE_RPY *const pstRpy);
    static VisionStatus _inspHoleByBlobMode(const cv::Mat &matInput, const cv::Mat &matMask, const PR_INSP_HOLE_CMD::BLOB_MODE_CRITERIA &stCriteria, PR_INSP_HOLE_RPY *const pstRpy);
    static VisionStatus _inspSingleLead(const cv::Mat                           &matInput,
                                        const PR_INSP_LEAD_CMD::LEAD_INPUT_INFO &stLeadInput,
                                        const PR_INSP_LEAD_CMD                  *pstCmd,
                                        PR_INSP_LEAD_RPY::LEAD_RESULT           &stLeadResult);
    static VisionStatus _calcMtfByFFT(const cv::Mat &matInput, bool bHorizontalStrip, VectorOfFloat &vecMtf);
    static VisionStatus _autoLocateLeadByTmpl(const cv::Mat &matGray, const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd, PR_AUTO_LOCATE_LEAD_RPY *const pstRpy);
    static VisionStatus _autoLocateLeadOneSide(const cv::Mat                       &matGray,
                                               const cv::Rect                      &rectSubRegion,
                                               const PR_DIRECTION                   enDir,
                                               const cv::Mat                       &matPad,
                                               const cv::Mat                       &matLead,
                                               Int32                                nPadRecordId,
                                               Int32                                nLeadRecordId,
                                               const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd,
                                               PR_AUTO_LOCATE_LEAD_RPY       *const pstRpy);
    static VisionStatus _autoLocateLeadByContour(const cv::Mat &matGray, const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd, PR_AUTO_LOCATE_LEAD_RPY *const pstRpy);
    static VisionStatus _imageSplitByMax(const cv::Mat &matInput, UInt16 objCount, std::vector<int> &vecIndex1, std::vector<int> &vecIndex2);
    static VisionStatus _imageSplitByMaxNew(const cv::Mat &matInput, UInt16 objCount, std::vector<int> &vecIndex1, std::vector<int> &vecIndex2);
    static cv::Mat _imageFilling(const cv::Mat &matInput, cv::Mat &matOuterEdge);
    static VectorOfPoint2f _lineIntersect4P(std::vector<int> &vecReverseFit, VectorOfFloat &vecSlope, VectorOfFloat &vecIntercept);
    static bool _lineScanRow(const cv::Mat &matInput, int &n, int &startRow);
    static bool _lineScanMatrix(const cv::Mat &matInput, int &m, int &n, int &startRow, int &startCol);
    static cv::Mat _pickColor(const std::vector<int> &vecValue, const cv::Mat &matColor, int nColorDiff, int nGrayDiff, UInt32 &nPointCountOut);
    static void _rotateOcvImage(PR_DIRECTION, cv::Mat &matInput);
    static void _rotateOcvImageBack(PR_DIRECTION, cv::Mat &matInput);
    static VisionStatus _checkOcvTmplSize(const cv::Mat &matInput, const std::vector<OcvRecordPtr>& vecRecordPtr);
    static VisionStatus _ocvOneDirection(PR_DIRECTION enDirection, const cv::Mat& matGray, const std::vector<OcvRecordPtr>& vecRecordPtr, const PR_OCV_CMD *const pstCmd, PR_OCV_RPY *const pstRpy);

protected:
    static const int       _constMinHessian        = 300;
    static const int       _constOctave            = 4;
    static const int       _constOctaveLayer       = 3;
    static const UInt32    _constLeastFeatures     = 3;
    static const String    _strRecordLogPrefix;
    static const float     _constExpSmoothRatio;
    
    static OcrTesseractPtr  _ptrOcrTesseract;
    static bool             _bAutoMode;

private:
    VisionAlgorithmPtr _pInstance = nullptr;
};

}
}
#endif /*_AOI_VISION_ALGORITHM_H_*/