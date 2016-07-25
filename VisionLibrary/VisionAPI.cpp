
#include <iostream>

#define DLLEXPORT
#include "VisionAPI.h"
#include "VisionAlgorithm.h"
#include "TimeLog.h"
#include "RecordManager.h"
#include "Config.h"

namespace AOI
{
namespace Vision
{

VisionAPI VisionStatus PR_LearnTmpl(PR_LEARN_TMPL_CMD * const pLearnTmplCmd, PR_LEARN_TMPL_RPY *pLearnTmplRpy)
{
    VisionAlgorithm va;
    return va.lrnTmpl(pLearnTmplCmd, pLearnTmplRpy );
}

VisionAPI VisionStatus PR_SrchTmpl(PR_SRCH_TMPL_CMD *const pSrchTmplCmd, PR_SRCH_TMPL_RPY *pSrchTmplRpy)
{
    VisionAlgorithm va;
    return va.srchTmpl(pSrchTmplCmd, pSrchTmplRpy );
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

VisionAPI VisionStatus PR_InspDevice(PR_INSP_DEVICE_CMD *pstInspDeviceCmd, PR_INSP_DEVICE_RPY *pstInspDeivceRpy)
{
    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    return pVA->inspDevice ( pstInspDeviceCmd, pstInspDeivceRpy );
}

VisionAPI void PR_SetDebugMode(PR_DEBUG_MODE enDebugMode)
{
    Config::GetInstance()->setDebugMode(enDebugMode);
}

}
}