#ifndef _VISION_API_H_
#define _VISION_API_H_

#ifdef DLLEXPORT
    #define VisionAPI   __declspec(dllexport) 
#else
    #define VisionAPI   __declspec(dllimport)
#endif

#include "VisionHeader.h"

namespace AOI
{
namespace Vision
{

VisionAPI void          PR_GetVersion(PR_VERSION_INFO *pstVersionInfo);
VisionAPI VisionStatus  PR_LearnTmpl(PR_LRN_TMPL_CMD * const pLrnTmplCmd, PR_LRN_TMPL_RPY *pLrnTmplRpy);
VisionAPI VisionStatus  PR_SrchTmpl(PR_SRCH_TMPL_CMD *const pFindObjCmd, PR_SRCH_TMPL_RPY *pFindObjRpy);
VisionAPI VisionStatus  PR_InspSurface(PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy);
VisionAPI void          PR_DumpTimeLog(const std::string &strPath);
VisionAPI VisionStatus  PR_FreeRecord(Int32 nRecordID);
VisionAPI VisionStatus  PR_FreeAllRecord();
VisionAPI VisionStatus  PR_LrnDevice(PR_LRN_DEVICE_CMD *pstLrnDeviceCmd, PR_LRN_DEVICE_RPY *pstLrnDeivceRpy);
VisionAPI VisionStatus  PR_InspDevice(PR_INSP_DEVICE_CMD *pstInspDeviceCmd, PR_INSP_DEVICE_RPY *pstInspDeivceRpy);
VisionAPI void          PR_SetDebugMode(PR_DEBUG_MODE enDebugMode);
VisionAPI VisionStatus  PR_RunLogCase(const std::string &strPath);
VisionAPI VisionStatus  PR_SrchFiducialMark(PR_SRCH_FIDUCIAL_MARK_CMD *pstCmd, PR_SRCH_FIDUCIAL_MARK_RPY *pstRpy);
VisionAPI VisionStatus  PR_FitLine(PR_FIT_LINE_CMD *pstCmd, PR_FIT_LINE_RPY *pstRpy);
VisionAPI VisionStatus  PR_FitParallelLine(PR_FIT_PARALLEL_LINE_CMD *pstCmd, PR_FIT_PARALLEL_LINE_RPY *pstRpy);
VisionAPI VisionStatus  PR_FitRect(PR_FIT_RECT_CMD *pstCmd, PR_FIT_RECT_RPY *pstRpy);

}
}

#endif