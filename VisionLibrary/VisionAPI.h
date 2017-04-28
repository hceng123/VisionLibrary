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
VisionAPI VisionStatus PR_LearnTmpl(PR_LRN_TMPL_CMD *const pLrnTmplCmd, PR_LRN_TMPL_RPY *pLrnTmplRpy);
VisionAPI VisionStatus PR_SrchTmpl(PR_SRCH_TMPL_CMD *const pFindObjCmd, PR_SRCH_TMPL_RPY *pFindObjRpy);
VisionAPI VisionStatus PR_InspSurface(PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy);
VisionAPI void         PR_DumpTimeLog(const std::string &strPath);
VisionAPI VisionStatus PR_FreeRecord(Int32 nRecordID);
VisionAPI VisionStatus PR_FreeAllRecord();
VisionAPI VisionStatus PR_LrnDevice(PR_LRN_DEVICE_CMD *pstLrnDeviceCmd, PR_LRN_DEVICE_RPY *pstLrnDeivceRpy);
VisionAPI VisionStatus PR_InspDevice(PR_INSP_DEVICE_CMD *pstInspDeviceCmd, PR_INSP_DEVICE_RPY *pstInspDeivceRpy);
VisionAPI void         PR_SetDebugMode(PR_DEBUG_MODE enDebugMode);
VisionAPI VisionStatus PR_RunLogCase(const std::string &strPath);
VisionAPI VisionStatus PR_SrchFiducialMark(PR_SRCH_FIDUCIAL_MARK_CMD *pstCmd, PR_SRCH_FIDUCIAL_MARK_RPY *pstRpy);

VisionAPI VisionStatus PR_FitLine(PR_FIT_LINE_CMD *pstCmd, PR_FIT_LINE_RPY *pstRpy);

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
VisionAPI VisionStatus PR_DetectLine(PR_DETECT_LINE_CMD *pstCmd, PR_DETECT_LINE_RPY *pstRpy);
VisionAPI VisionStatus PR_FitParallelLine(PR_FIT_PARALLEL_LINE_CMD *pstCmd, PR_FIT_PARALLEL_LINE_RPY *pstRpy);
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
VisionAPI VisionStatus PR_FindEdge(PR_FIND_EDGE_CMD *pstCmd, PR_FIND_EDGE_RPY *pstRpy);
VisionAPI VisionStatus PR_FitCircle(PR_FIT_CIRCLE_CMD *pstCmd, PR_FIT_CIRCLE_RPY *pstRpy);
VisionAPI VisionStatus PR_GetErrorStr(VisionStatus enStatus, PR_GET_ERROR_STR_RPY *pstRpy);
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
VisionAPI VisionStatus PR_CircleRoundness(PR_CIRCLE_ROUNDNESS_CMD *pstCmd, PR_CIRCLE_ROUNDNESS_RPY *pstRpy);
VisionAPI VisionStatus PR_FillHole(PR_FILL_HOLE_CMD *const pstCmd, PR_FILL_HOLE_RPY *pstRpy);
VisionAPI VisionStatus PR_MatchTmpl(PR_MATCH_TEMPLATE_CMD *const pstCmd, PR_MATCH_TEMPLATE_RPY *pstRpy);
VisionAPI VisionStatus PR_PickColor(PR_PICK_COLOR_CMD *const pstCmd, PR_PICK_COLOR_RPY *pstRpy);
VisionAPI VisionStatus PR_CalibrateCamera(const PR_CALIBRATE_CAMERA_CMD *const pstCmd, PR_CALIBRATE_CAMERA_RPY *const pstRpy);
VisionAPI VisionStatus PR_RestoreImage(const PR_RESTORE_IMG_CMD *const pstCmd, PR_RESTORE_IMG_RPY *const pstRpy);
VisionAPI VisionStatus PR_AutoLocateLead(const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd, PR_AUTO_LOCATE_LEAD_RPY *const pstRpy);

}
}

#endif