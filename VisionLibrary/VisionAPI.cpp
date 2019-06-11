
#include <iostream>

#define DLLEXPORT
#include "VisionAPI.h"
#include "VisionAlgorithm.h"
#include "TimeLog.h"
#include "RecordManager.h"
#include "Config.h"
#include "Version.h"
#include "SubFunctions.h"
#include "Log.h"
#include "CalcUtils.hpp"
#include "Unwrap.h"
#include "TableMapping.h"
#include "CudaAlgorithm.h"

#define PR_FUNCTION_ENTRY   \
try \
{

#define PR_FUNCTION_EXIT    \
}catch(std::exception &e)   \
{   \
    WriteLog(e.what()); \
    pstRpy->enStatus = VisionStatus::OPENCV_EXCEPTION;  \
    return VisionStatus::OPENCV_EXCEPTION;  \
}

namespace AOI
{
namespace Vision
{

VisionAPI void PR_GetVersion(PR_VERSION_INFO *pstVersionInfo)
{
    if (nullptr != pstVersionInfo)
        _snprintf(pstVersionInfo->chArrVersion, sizeof (pstVersionInfo->chArrVersion), AOI_VISION_VERSION);
}

VisionAPI VisionStatus PR_Init()
{
    try {
        Unwrap::initAtan2Table();
        CudaAlgorithm::initCuda();
        return RecordManager::getInstance()->load();
    }catch (std::exception &e) {
        WriteLog(e.what());
        return VisionStatus::OPENCV_EXCEPTION;
    }
}

VisionAPI void PR_EnableAutoMode(bool bEnableAutoMode)
{
    VisionAlgorithm::EnableAutoMode(bEnableAutoMode);
}

VisionAPI void _PR_InternalTest()
{
    try {
        InternalTest();
    }catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}

VisionAPI VisionStatus PR_LrnObj(const PR_LRN_OBJ_CMD *const pstCmd, PR_LRN_OBJ_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::lrnObj(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_SrchObj(const PR_SRCH_OBJ_CMD *const pstCmd, PR_SRCH_OBJ_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
return VisionAlgorithm::srchObj(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_InspSurface(PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy)
{
    try
    {
        VisionAlgorithmPtr pVA = VisionAlgorithm::create();
        return pVA->inspSurface(pInspCmd, pInspRpy);
    }catch(std::exception &e)
    {
        WriteLog(e.what());
        return VisionStatus::OPENCV_EXCEPTION;
    }
}

VisionAPI void PR_DumpTimeLog(const std::string &strPath)
{
    TimeLog::GetInstance()->dumpTimeLog(strPath);
}

VisionAPI VisionStatus PR_FreeRecord(Int32 nRecordId)
{
    try {
        return RecordManagerInstance->free(nRecordId);
    }catch (std::exception &e) {
        WriteLog(e.what());
        return VisionStatus::OPENCV_EXCEPTION;
    }
}

VisionAPI VisionStatus PR_FreeAllRecord()
{
    try {
        return RecordManagerInstance->freeAllRecord();
    }catch (std::exception &e) {
        WriteLog(e.what());
        return VisionStatus::OPENCV_EXCEPTION;
    }
}

VisionAPI VisionStatus PR_GetRecordInfo(Int32 nRecordId, PR_GET_RECORD_INFO_RPY *const pstRpy)
{
    auto ptrRecord = RecordManagerInstance->get(nRecordId);
    if (nullptr == ptrRecord) {
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }
    pstRpy->enStatus = VisionStatus::OK;
    pstRpy->matImage = ptrRecord->getImage();
    return pstRpy->enStatus;
}

VisionAPI VisionStatus PR_LrnDevice(PR_LRN_DEVICE_CMD *pstLrnDeviceCmd, PR_LRN_DEVICE_RPY *pstLrnDeivceRpy)
{
    try
    {
        VisionAlgorithmPtr pVA = VisionAlgorithm::create();
        return pVA->lrnDevice(pstLrnDeviceCmd, pstLrnDeivceRpy);
    }catch(std::exception &e)
    {
        WriteLog(e.what());
        return VisionStatus::OPENCV_EXCEPTION;
    }
}

VisionAPI VisionStatus PR_InspDevice(PR_INSP_DEVICE_CMD *pstInspDeviceCmd, PR_INSP_DEVICE_RPY *pstInspDeivceRpy)
{
    try
    {
        VisionAlgorithmPtr pVA = VisionAlgorithm::create();
        return pVA->inspDevice(pstInspDeviceCmd, pstInspDeivceRpy);
    }catch(std::exception &e)
    {
        WriteLog(e.what());
        return VisionStatus::OPENCV_EXCEPTION;
    }
}

VisionAPI void PR_SetDebugMode(PR_DEBUG_MODE enDebugMode)
{
    Config::GetInstance()->setDebugMode(enDebugMode);
}

VisionAPI VisionStatus PR_RunLogCase(const std::string &strPath)
{
    try {
        return VisionAlgorithm::runLogCase(strPath);
    }catch(std::exception &e) {
        WriteLog(e.what());
        return VisionStatus::OPENCV_EXCEPTION;
    }
}

VisionAPI VisionStatus PR_SrchFiducialMark(PR_SRCH_FIDUCIAL_MARK_CMD *pstCmd, PR_SRCH_FIDUCIAL_MARK_RPY *pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::srchFiducialMark(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus  PR_FitLine(const PR_FIT_LINE_CMD *const pstCmd, PR_FIT_LINE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::fitLine(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_FitLineByPoint(const PR_FIT_LINE_BY_POINT_CMD *const pstCmd, PR_FIT_LINE_BY_POINT_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::fitLineByPoint(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus  PR_FindLine(const PR_FIND_LINE_CMD *const pstCmd, PR_FIND_LINE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::findLine(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus  PR_FitParallelLine(const PR_FIT_PARALLEL_LINE_CMD *const pstCmd, PR_FIT_PARALLEL_LINE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::fitParallelLine(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus  PR_FitRect(PR_FIT_RECT_CMD *pstCmd, PR_FIT_RECT_RPY *pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::fitRect(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus  PR_FindEdge(const PR_FIND_EDGE_CMD *const pstCmd, PR_FIND_EDGE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::findEdge(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionStatus  PR_FitCircle(const PR_FIT_CIRCLE_CMD *const pstCmd, PR_FIT_CIRCLE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::fitCircle(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_FitCircleByPoint(const PR_FIT_CIRCLE_BY_POINT_CMD *const pstCmd, PR_FIT_CIRCLE_BY_POINT_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::fitCircleByPoint(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_FindCircle(const PR_FIND_CIRCLE_CMD *const pstCmd, PR_FIND_CIRCLE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::findCircle(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_InspCircle(const PR_INSP_CIRCLE_CMD *const pstCmd, PR_INSP_CIRCLE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::inspCircle(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_GetErrorInfo(VisionStatus enStatus, PR_GET_ERROR_INFO_RPY *pstRpy)
{
    return GetErrorInfo(enStatus, pstRpy);
}

VisionAPI VisionStatus PR_Ocr(PR_OCR_CMD *pstCmd, PR_OCR_RPY *pstRpy)
{
    try
    {
        VisionAlgorithmPtr pVA = VisionAlgorithm::create();
        return pVA->ocr(pstCmd, pstRpy);
    }catch(std::exception &e)
    {
        WriteLog(e.what());
        return VisionStatus::OPENCV_EXCEPTION;
    }
}

VisionAPI VisionStatus PR_PointLineDistance(const PR_POINT_LINE_DISTANCE_CMD *const pstCmd, PR_POINT_LINE_DISTANCE_RPY *const pstRpy)
{
    pstRpy->fDistance = CalcUtils::ptDisToLine ( pstCmd->ptInput, pstCmd->bReversedFit, pstCmd->fSlope, pstCmd->fIntercept );
    pstRpy->fDistance = fabs(pstRpy->fDistance);
    return VisionStatus::OK;
}

VisionAPI VisionStatus PR_TwoLineAngle(const PR_TWO_LINE_ANGLE_CMD *const pstCmd, PR_TWO_LINE_ANGLE_RPY *const pstRpy)
{
    pstRpy->fAngle = CalcUtils::calc2LineAngle(pstCmd->line1, pstCmd->line2);
    return VisionStatus::OK;
}

VisionAPI VisionStatus PR_TwoLineIntersect(const PR_TWO_LINE_INTERSECT_CMD *const pstCmd, PR_TWO_LINE_INTERSECT_RPY *const pstRpy)
{
    int nResult = CalcUtils::twoLineIntersect(pstCmd->line1, pstCmd->line2, pstRpy->ptIntersect);
    pstRpy->enStatus = VisionStatus::OK;
    if (nResult != 0) {
        WriteLog("Cannot find intersect point for parallel lines");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
    }
    return pstRpy->enStatus;
}

VisionAPI VisionStatus PR_ParallelLineDist(const PR_PARALLEL_LINE_DIST_CMD *const pstCmd, PR_PARALLEL_LINE_DIST_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::parallelLineDist(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_MeasureDist(const PR_MEASURE_DIST_CMD* const pstCmd, PR_MEASURE_DIST_RPY* const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::measureDist(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_CrossSectionArea(const PR_CROSS_SECTION_AREA_CMD *const pstCmd, PR_CROSS_SECTION_AREA_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::crossSectionArea(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_ColorToGray(PR_COLOR_TO_GRAY_CMD *pstCmd, PR_COLOR_TO_GRAY_RPY *pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::colorToGray(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_Filter(PR_FILTER_CMD *pstCmd, PR_FILTER_RPY *pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::filter(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_AutoThreshold(PR_AUTO_THRESHOLD_CMD *pstCmd, PR_AUTO_THRESHOLD_RPY *pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::autoThreshold(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_RemoveCC(PR_REMOVE_CC_CMD *pstCmd, PR_REMOVE_CC_RPY *pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::removeCC(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_DetectEdge(PR_DETECT_EDGE_CMD *pstCmd, PR_DETECT_EDGE_RPY *pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::detectEdge(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_FillHole(PR_FILL_HOLE_CMD *const pstCmd, PR_FILL_HOLE_RPY *pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::fillHole(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_LrnTmpl(const PR_LRN_TEMPLATE_CMD *const pstCmd, PR_LRN_TEMPLATE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::lrnTemplate(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_MatchTmpl(const PR_MATCH_TEMPLATE_CMD *const pstCmd, PR_MATCH_TEMPLATE_RPY * const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::matchTemplate(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_PickColor(const PR_PICK_COLOR_CMD *const pstCmd, PR_PICK_COLOR_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::pickColor(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_CalibrateCamera(const PR_CALIBRATE_CAMERA_CMD *const pstCmd, PR_CALIBRATE_CAMERA_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calibrateCamera(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_RestoreImage(const PR_RESTORE_IMG_CMD *const pstCmd, PR_RESTORE_IMG_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::restoreImage(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_CalcUndistortRectifyMap(const PR_CALC_UNDISTORT_RECTIFY_MAP_CMD *const pstCmd, PR_CALC_UNDISTORT_RECTIFY_MAP_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calcUndistortRectifyMap(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_AutoLocateLead(const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd, PR_AUTO_LOCATE_LEAD_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::autoLocateLead(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_InspBridge(const PR_INSP_BRIDGE_CMD *const pstCmd, PR_INSP_BRIDGE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::inspBridge(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_LrnChip(const PR_LRN_CHIP_CMD *const pstCmd, PR_LRN_CHIP_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::lrnChip(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_InspChip(const PR_INSP_CHIP_CMD *const pstCmd, PR_INSP_CHIP_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::inspChip(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_LrnContour(const PR_LRN_CONTOUR_CMD *const pstCmd, PR_LRN_CONTOUR_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::lrnContour(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_InspContour(const PR_INSP_CONTOUR_CMD *const pstCmd, PR_INSP_CONTOUR_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::inspContour(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_InspHole(const PR_INSP_HOLE_CMD *const pstCmd, PR_INSP_HOLE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::inspHole(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_InspLead(const PR_INSP_LEAD_CMD *const pstCmd, PR_INSP_LEAD_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::inspLead(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_InspLeadTmpl(const PR_INSP_LEAD_TMPL_CMD *const pstCmd, PR_INSP_LEAD_TMPL_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::inspLeadTmpl(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_InspPolarity(const PR_INSP_POLARITY_CMD *const pstCmd, PR_INSP_POLARITY_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::inspPolarity(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_GridAvgGrayScale(const PR_GRID_AVG_GRAY_SCALE_CMD *const pstCmd, PR_GRID_AVG_GRAY_SCALE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::gridAvgGrayScale(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_Calib3DBase(const PR_CALIB_3D_BASE_CMD *const pstCmd, PR_CALIB_3D_BASE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calib3DBase(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_Calc3DBase(const PR_CALC_3D_BASE_CMD *const pstCmd, PR_CALC_3D_BASE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calc3DBase(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_Calib3DHeight(const PR_CALIB_3D_HEIGHT_CMD *const pstCmd, PR_CALIB_3D_HEIGHT_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calib3DHeight(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_MotorCalib3D(const PR_MOTOR_CALIB_3D_CMD *const pstCmd, PR_MOTOR_CALIB_3D_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::motorCalib3D(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_MotorCalib3DNew(const PR_MOTOR_CALIB_3D_CMD *const pstCmd, PR_MOTOR_CALIB_3D_NEW_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::motorCalib3DNew(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_CalibDlpOffset(const PR_CALIB_DLP_OFFSET_CMD *const pstCmd, PR_CALIB_DLP_OFFSET_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calibDlpOffset(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_Integrate3DCalib(const PR_INTEGRATE_3D_CALIB_CMD *const pstCmd, PR_INTEGRATE_3D_CALIB_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::integrate3DCalib(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_Calc3DHeight(const PR_CALC_3D_HEIGHT_CMD *const pstCmd, PR_CALC_3D_HEIGHT_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calc3DHeight(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_Calc3DHeightNew(const PR_CALC_3D_HEIGHT_NEW_CMD *const pstCmd, PR_CALC_3D_HEIGHT_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calc3DHeightNew(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_SetDlpParamsToGpu(const PR_SET_DLP_PARAMS_TO_GPU_CMD* const pstCmd, PR_SET_DLP_PARAMS_TO_GPU_RPY* const pstRpy)
{
    return CudaAlgorithm::setDlpParams(pstCmd, pstRpy);
}

VisionAPI VisionStatus PR_ClearDlpParams()
{
    return CudaAlgorithm::clearDlpParams();
}

VisionAPI VisionStatus PR_Calc3DHeightGpu(const PR_CALC_3D_HEIGHT_GPU_CMD *const pstCmd, PR_CALC_3D_HEIGHT_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calc3DHeightGpu(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_Merge3DHeight(const PR_MERGE_3D_HEIGHT_CMD *const pstCmd, PR_MERGE_3D_HEIGHT_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::merge3DHeight(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_CalcMerge4DlpHeight(const PR_CALC_MERGE_4_DLP_HEIGHT_CMD *const pstCmd, PR_CALC_MERGE_4_DLP_HEIGHT_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calcMerge4DlpHeight(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_Calc3DHeightDiff(const PR_CALC_3D_HEIGHT_DIFF_CMD *const pstCmd, PR_CALC_3D_HEIGHT_DIFF_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calc3DHeightDiff(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_Insp3DSolder(const PR_INSP_3D_SOLDER_CMD *pstCmd, PR_INSP_3D_SOLDER_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::insp3DSolder(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_CalcDlpOffset(const PR_CALC_DLP_OFFSET_CMD *const pstCmd, PR_CALC_DLP_OFFSET_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calcDlpOffset(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionStatus PR_CalcFrameValue(const PR_CALC_FRAME_VALUE_CMD *const pstCmd, PR_CALC_FRAME_VALUE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calcFrameValue(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_CalcCameraMTF(const PR_CALC_CAMERA_MTF_CMD *const pstCmd, PR_CALC_CAMERA_MTF_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calcCameraMTF(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_CalcMTF(const PR_CALC_MTF_CMD *const pstCmd, PR_CALC_MTF_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calcMTF(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_CalcPD(const PR_CALC_PD_CMD *const pstCmd, PR_CALC_PD_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calcPD(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_CombineImg(const PR_COMBINE_IMG_CMD *const pstCmd, PR_COMBINE_IMG_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::combineImg(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_CombineImgNew(const PR_COMBINE_IMG_NEW_CMD *const pstCmd, PR_COMBINE_IMG_NEW_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::combineImgNew(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_Threshold(const PR_THRESHOLD_CMD *const pstCmd, PR_THRESHOLD_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::threshold(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_HeightToGray(const PR_HEIGHT_TO_GRAY_CMD *const pstCmd, PR_HEIGHT_TO_GRAY_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::heightToGray(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_LrnOcv(const PR_LRN_OCV_CMD *const pstCmd, PR_LRN_OCV_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::lrnOcv(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_GetOcvRecordInfo(Int32 nRecordId, PR_OCV_RECORD_INFO *const pstRecordInfo)
{
    auto ptrRecord = RecordManagerInstance->get(nRecordId);
    if (nullptr == ptrRecord)
        return VisionStatus::INVALID_PARAM;
    if (ptrRecord->getType() != PR_RECORD_TYPE::OCV)
        return VisionStatus::INVALID_PARAM;
    OcvRecordPtr ptrOcvRecord = std::static_pointer_cast<OcvRecord>(ptrRecord);
    pstRecordInfo->matTmplImg = ptrOcvRecord->getBigTmpl();
    pstRecordInfo->vecCharRects = ptrOcvRecord->getCharRects();
    return VisionStatus::OK;
}

VisionAPI VisionStatus PR_SetOcvRecordInfo(Int32 nRecordId, const PR_OCV_RECORD_INFO *const pstRecordInfo)
{
    auto ptrRecord = RecordManagerInstance->get(nRecordId);
    if (nullptr == ptrRecord)
        return VisionStatus::INVALID_PARAM;
    if (ptrRecord->getType() != PR_RECORD_TYPE::OCV)
        return VisionStatus::INVALID_PARAM;
    OcvRecordPtr ptrOcvRecord = std::static_pointer_cast<OcvRecord>(ptrRecord);
    ptrOcvRecord->setCharRects(pstRecordInfo->vecCharRects);
    RecordManagerInstance->update(nRecordId, ptrOcvRecord);
    return VisionStatus::OK;
}

VisionAPI VisionStatus PR_Ocv(const PR_OCV_CMD *const pstCmd, PR_OCV_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::ocv(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_Read2DCode(const PR_READ_2DCODE_CMD *const pstCmd, PR_READ_2DCODE_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::read2DCode(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_LrnSimilarity(const PR_LRN_SIMILARITY_CMD *const pstCmd, PR_LRN_SIMILARITY_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::lrnSimilarity(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_InspSimilarity(const PR_INSP_SIMILARITY_CMD *const pstCmd, PR_INSP_SIMILARITY_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::inspSimilarity(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_TableMapping(const PR_TABLE_MAPPING_CMD *const pstCmd, PR_TABLE_MAPPING_RPY *const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::tableMapping(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_CalcTableOffset(const PR_CALC_TABLE_OFFSET_CMD* const pstCmd, PR_CALC_TABLE_OFFSET_RPY* const pstRpy)
{
PR_FUNCTION_ENTRY
    return TableMapping::calcTableOffsetNew1(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

VisionAPI VisionStatus PR_CalcRestoreIdx(const PR_CALC_RESTORE_IDX_CMD* const pstCmd, PR_CALC_RESTORE_IDX_RPY* const pstRpy)
{
PR_FUNCTION_ENTRY
    return VisionAlgorithm::calcRestoreIdx(pstCmd, pstRpy);
PR_FUNCTION_EXIT
}

}
}
