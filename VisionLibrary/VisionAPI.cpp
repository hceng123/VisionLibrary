
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

namespace AOI
{
namespace Vision
{

VisionAPI void PR_GetVersion(PR_VERSION_INFO *pstVersionInfo)
{
    if ( nullptr != pstVersionInfo )
        _snprintf(pstVersionInfo->chArrVersion, sizeof (pstVersionInfo->chArrVersion), AOI_VISION_VERSION );
}

VisionAPI VisionStatus PR_LearnTmpl(PR_LRN_TMPL_CMD * const pLrnTmplCmd, PR_LRN_TMPL_RPY *pLrnTmplRpy)
{
    VisionAlgorithm va;
    return va.lrnTmpl ( pLrnTmplCmd, pLrnTmplRpy );
}

VisionAPI VisionStatus PR_SrchTmpl(PR_SRCH_TMPL_CMD *const pSrchTmplCmd, PR_SRCH_TMPL_RPY *pSrchTmplRpy)
{
    VisionAlgorithm va;
    return va.srchTmpl ( pSrchTmplCmd, pSrchTmplRpy );
}

VisionAPI VisionStatus PR_InspSurface(PR_INSP_SURFACE_CMD *const pInspCmd, PR_INSP_SURFACE_RPY *pInspRpy)
{
    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    return pVA->inspSurface ( pInspCmd, pInspRpy );
}

VisionAPI void PR_DumpTimeLog(const std::string &strPath)
{
    TimeLog::GetInstance()->dumpTimeLog(strPath);
}

VisionAPI VisionStatus PR_FreeRecord(Int32 nRecordID)
{
    return RecordManager::getInstance()->free(nRecordID);
}

VisionAPI VisionStatus  PR_FreeAllRecord()
{
    return RecordManager::getInstance()->freeAllRecord();
}

VisionAPI VisionStatus  PR_LrnDevice(PR_LRN_DEVICE_CMD *pstLrnDeviceCmd, PR_LRN_DEVICE_RPY *pstLrnDeivceRpy)
{
    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    return pVA->lrnDevice ( pstLrnDeviceCmd, pstLrnDeivceRpy );
}

VisionAPI VisionStatus PR_InspDevice(PR_INSP_DEVICE_CMD *pstInspDeviceCmd, PR_INSP_DEVICE_RPY *pstInspDeivceRpy)
{
    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    return pVA->inspDevice ( pstInspDeviceCmd, pstInspDeivceRpy );
}

VisionAPI void PR_SetDebugMode(PR_DEBUG_MODE enDebugMode)
{
    Config::GetInstance()->setDebugMode(enDebugMode);
}

VisionAPI VisionStatus  PR_RunLogCase(const std::string &strPath)
{
    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    return pVA->runLogCase ( strPath );
}

VisionAPI VisionStatus PR_SrchFiducialMark(PR_SRCH_FIDUCIAL_MARK_CMD *pstCmd, PR_SRCH_FIDUCIAL_MARK_RPY *pstRpy)
{
    try
    {
        std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
        return pVA->srchFiducialMark ( pstCmd, pstRpy );
    }catch(std::exception &e)
    {
        WriteLog(e.what());
        return VisionStatus::OPENCV_EXCEPTION;
    }
}

VisionAPI VisionStatus  PR_FitLine(PR_FIT_LINE_CMD *pstCmd, PR_FIT_LINE_RPY *pstRpy)
{
    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    return pVA->fitLine ( pstCmd, pstRpy );
}

VisionAPI VisionStatus  PR_FitParallelLine(PR_FIT_PARALLEL_LINE_CMD *pstCmd, PR_FIT_PARALLEL_LINE_RPY *pstRpy)
{
    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    return pVA->fitParallelLine ( pstCmd, pstRpy );
}

VisionAPI VisionStatus  PR_FitRect(PR_FIT_RECT_CMD *pstCmd, PR_FIT_RECT_RPY *pstRpy)
{
    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    return pVA->fitRect ( pstCmd, pstRpy );
}

VisionAPI VisionStatus  PR_FindEdge(PR_FIND_EDGE_CMD *pstCmd, PR_FIND_EDGE_RPY *pstRpy)
{
    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    return pVA->findEdge ( pstCmd, pstRpy );
}

VisionStatus  PR_FitCircle(PR_FIT_CIRCLE_CMD *pstCmd, PR_FIT_CIRCLE_RPY *pstRpy)
{
    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    return pVA->fitCircle ( pstCmd, pstRpy );
}

VisionStatus  PR_GetErrorStr(VisionStatus enStatus, PR_GET_ERROR_STR_RPY *pstRpy)
{
    return GetErrorInfo(enStatus, pstRpy);
}

VisionAPI VisionStatus  PR_Ocr(PR_OCR_CMD *pstCmd, PR_OCR_RPY *pstRpy)
{
    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    return pVA->ocr ( pstCmd, pstRpy );
}

}
}