#include "VisionHeader.h"

namespace AOI
{
namespace Vision
{

VisionStatus GetErrorInfo(VisionStatus enStatus, PR_GET_ERROR_STR_RPY *pstRpy)
{
    switch(enStatus)
    {
    case VisionStatus::OK:
        pstRpy->achErrorStr[0] = '\0';
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_NO_ERROR;
        break;
    case VisionStatus::INVALID_PARAM:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Input parameter is invalid");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::NOT_ENOUGH_POINTS_TO_FIT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Not enough points to fit");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::TOO_MUCH_NOISE_TO_FIT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Too much noise to fit");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    default:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Undefined error code!");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    }
    return VisionStatus::OK;
}

}
}