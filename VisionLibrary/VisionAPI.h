#ifndef _VISION_API_H_
#define _VISION_API_H_

#ifdef DLLEXPORT
    #define VisionAPI   __declspec(dllexport)
#else
    #define VisionAPI   __declspec(dllimport)
#endif

#include "VisionHeader.h"
#include <memory>

namespace AOI
{
namespace Vision
{
VisionAPI void         PR_GetVersion(PR_VERSION_INFO *pstVersionInfo);
VisionAPI VisionStatus PR_Init();
VisionAPI void         PR_SetDebugMode(PR_DEBUG_MODE enDebugMode);
VisionAPI VisionStatus PR_RunLogCase(const std::string &strPath);
VisionAPI void         PR_DumpTimeLog(const std::string &strPath);
VisionAPI VisionStatus PR_FreeRecord(Int32 nRecordId);
VisionAPI VisionStatus PR_FreeAllRecord();
VisionAPI VisionStatus PR_LrnObj(const PR_LRN_OBJ_CMD *const pstCmd, PR_LRN_OBJ_RPY *const pstRpy);
VisionAPI VisionStatus PR_SrchObj(const PR_SRCH_OBJ_CMD *const pstCmd, PR_SRCH_OBJ_RPY *const pstRpy);
VisionAPI VisionStatus PR_InspSurface(PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy);
VisionAPI VisionStatus PR_LrnDevice(PR_LRN_DEVICE_CMD *pstLrnDeviceCmd, PR_LRN_DEVICE_RPY *pstLrnDeivceRpy);
VisionAPI VisionStatus PR_InspDevice(PR_INSP_DEVICE_CMD *pstInspDeviceCmd, PR_INSP_DEVICE_RPY *pstInspDeivceRpy);
VisionAPI VisionStatus PR_SrchFiducialMark(PR_SRCH_FIDUCIAL_MARK_CMD *pstCmd, PR_SRCH_FIDUCIAL_MARK_RPY *pstRpy);
VisionAPI VisionStatus PR_FitLine(const PR_FIT_LINE_CMD *const pstCmd, PR_FIT_LINE_RPY *const pstRpy);

/** @brief Use projection method to find a single line.
Example:
@code{.cpp}
    PR_DETECT_LINE_CMD stCmd;
    ...//Initialize the input iamge
    PR_DETECT_LINE_RPY stRpy;
    PR_DetectLine(stCmd, stRpy);
@endcode
@param pstCmd pointer of input command.
@param pstRpy pointer of reply.
*/
VisionAPI VisionStatus PR_Caliper(const PR_CALIPER_CMD *const pstCmd, PR_CALIPER_RPY *const pstRpy);
VisionAPI VisionStatus PR_FitParallelLine(const PR_FIT_PARALLEL_LINE_CMD *const pstCmd, PR_FIT_PARALLEL_LINE_RPY *const pstRpy);
VisionAPI VisionStatus PR_FitRect(PR_FIT_RECT_CMD *pstCmd, PR_FIT_RECT_RPY *pstRpy);

/** @brief Use hough tranform to find all the lines in the specification.
Example:
@code{.cpp}
    PR_FIND_EDGE_CMD stCmd;
    ...//Initialize the input iamge
    PR_FIND_EDGE_RPY stRpy;
    PR_FindEdge(stCmd, stRpy);
@endcode
@param pstCmd pointer of input command.
@param pstRpy pointer of reply.
*/
VisionAPI VisionStatus PR_FindEdge(const PR_FIND_EDGE_CMD *const pstCmd, PR_FIND_EDGE_RPY *const pstRpy);
VisionAPI VisionStatus PR_FitCircle(PR_FIT_CIRCLE_CMD *pstCmd, PR_FIT_CIRCLE_RPY *pstRpy);
VisionAPI VisionStatus PR_GetErrorInfo(VisionStatus enStatus, PR_GET_ERROR_INFO_RPY *pstRpy);
VisionAPI VisionStatus PR_Ocr(PR_OCR_CMD *pstCmd, PR_OCR_RPY *pstRpy);
VisionAPI VisionStatus PR_PointLineDistance(PR_POINT_LINE_DISTANCE_CMD *pstCmd, PR_POINT_LINE_DISTANCE_RPY *pstRpy);
VisionAPI VisionStatus PR_ColorToGray(PR_COLOR_TO_GRAY_CMD *pstCmd, PR_COLOR_TO_GRAY_RPY *pstRpy);
VisionAPI VisionStatus PR_Filter(PR_FILTER_CMD *pstCmd, PR_FILTER_RPY *pstRpy);
VisionAPI VisionStatus PR_AutoThreshold(PR_AUTO_THRESHOLD_CMD *pstCmd, PR_AUTO_THRESHOLD_RPY *pstRpy);
VisionAPI VisionStatus PR_RemoveCC(PR_REMOVE_CC_CMD *pstCmd, PR_REMOVE_CC_RPY *pstRpy);

/** @brief Use Canny algorithm to find all the edges.
Example:
@code{.cpp}
    PR_FIND_EDGE_CMD stCmd;
    ...//Initialize the input iamge
    PR_FIND_EDGE_RPY stRpy;
    PR_FindEdge(stCmd, stRpy);
@endcode
@param pstCmd pointer of input command.
@param pstRpy pointer of reply.
*/
VisionAPI VisionStatus PR_DetectEdge(PR_DETECT_EDGE_CMD *pstCmd, PR_DETECT_EDGE_RPY *pstRpy);
VisionAPI VisionStatus PR_InspCircle(PR_INSP_CIRCLE_CMD *pstCmd, PR_INSP_CIRCLE_RPY *pstRpy);
VisionAPI VisionStatus PR_FillHole(PR_FILL_HOLE_CMD *const pstCmd, PR_FILL_HOLE_RPY *pstRpy);
VisionAPI VisionStatus PR_LrnTmpl ( const PR_LRN_TEMPLATE_CMD *const pstCmd, PR_LRN_TEMPLATE_RPY *const pstRpy);
VisionAPI VisionStatus PR_MatchTmpl(const PR_MATCH_TEMPLATE_CMD *const pstCmd, PR_MATCH_TEMPLATE_RPY * const pstRpy);
VisionAPI VisionStatus PR_PickColor(const PR_PICK_COLOR_CMD *const pstCmd, PR_PICK_COLOR_RPY *const pstRpy);
VisionAPI VisionStatus PR_CalibrateCamera(const PR_CALIBRATE_CAMERA_CMD *const pstCmd, PR_CALIBRATE_CAMERA_RPY *const pstRpy);
VisionAPI VisionStatus PR_RestoreImage(const PR_RESTORE_IMG_CMD *const pstCmd, PR_RESTORE_IMG_RPY *const pstRpy);
VisionAPI VisionStatus PR_CalcUndistortRectifyMap(const PR_CALC_UNDISTORT_RECTIFY_MAP_CMD *const pstCmd, PR_CALC_UNDISTORT_RECTIFY_MAP_RPY *const pstRpy);
VisionAPI VisionStatus PR_AutoLocateLead(const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd, PR_AUTO_LOCATE_LEAD_RPY *const pstRpy);
VisionAPI VisionStatus PR_InspBridge(const PR_INSP_BRIDGE_CMD *const pstCmd, PR_INSP_BRIDGE_RPY *const pstRpy);
VisionAPI VisionStatus PR_LrnChip(const PR_LRN_CHIP_CMD *const pstCmd, PR_LRN_CHIP_RPY *const pstRpy);
VisionAPI VisionStatus PR_InspChip(const PR_INSP_CHIP_CMD *const pstCmd, PR_INSP_CHIP_RPY *const pstRpy);
VisionAPI VisionStatus PR_LrnContour(const PR_LRN_CONTOUR_CMD *const pstCmd, PR_LRN_CONTOUR_RPY *const pstRpy);
VisionAPI VisionStatus PR_InspContour(const PR_INSP_CONTOUR_CMD *const pstCmd, PR_INSP_CONTOUR_RPY *const pstRpy);
VisionAPI VisionStatus PR_InspHole(const PR_INSP_HOLE_CMD *const pstCmd, PR_INSP_HOLE_RPY *const pstRpy);
VisionAPI VisionStatus PR_InspLead(const PR_INSP_LEAD_CMD *const pstCmd, PR_INSP_LEAD_RPY *const pstRpy);
}
}

#endif