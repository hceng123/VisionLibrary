#ifndef _VISION_API_H_
#define _VISION_API_H_

#ifdef DLLEXPORT
    #define VisionAPI   __declspec(dllexport) 
#else
    #define VisionAPI   __declspec(dllimport)
#endif

#include "VisionStruct.h"
#include "VisionStatus.h"

namespace AOI
{
namespace Vision
{

VisionAPI VisionStatus PR_LearnTmpl(PR_LEARN_TMPL_CMD * const pLearnTmplCmd, PR_LEARN_TMPL_RPY *pLearnTmplRpy);
VisionAPI VisionStatus PR_SrchTmpl(PR_SRCH_TMPL_CMD *const pFindObjCmd, PR_SRCH_TMPL_RPY *pFindObjRpy);
VisionAPI void PR_DumpTimeLog(const std::string &strPath);
VisionAPI VisionStatus PR_FreeRecord(Int32 nRecordID);
VisionAPI VisionStatus PR_FreeAllRecord();
VisionAPI VisionStatus PR_InspDevice(PR_INSP_DEVICE_CMD *pstInspDeviceCmd, PR_INSP_DEVICE_RPY *pstInspDeivceRpy);

}
}

#endif