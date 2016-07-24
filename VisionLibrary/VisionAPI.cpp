
#include <iostream>

#define DLLEXPORT
#include "VisionAPI.h"
#include "VisionAlgorithm.h"
#include "TimeLog.h"
#include "RecordManager.h"

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
    return VisionStatus::OK;
}

}
}