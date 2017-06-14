#include "VisionHeader.h"

namespace AOI
{
namespace Vision
{

VisionStatus GetErrorInfo(VisionStatus enStatus, PR_GET_ERROR_STR_RPY *pstRpy)
{
    switch (enStatus)
    {
    case VisionStatus::OK:
        pstRpy->achErrorStr[0] = '\0';
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_NO_ERROR;
        break;
    case VisionStatus::INVALID_PARAM:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Input parameter is invalid.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::FIND_ELECTRODE_FAIL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to find electrode of chip.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::NOT_ENOUGH_POINTS_TO_FIT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Not enough points to fit.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::TOO_MUCH_NOISE_TO_FIT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Too much noise to fit.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::FAIL_TO_FIT_CIRCLE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to fit circle, the fitting result is invalid.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::OPENCV_EXCEPTION:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Exception encounted, please check log file.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::GUASSIAN_FILTER_KERNEL_INVALID:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Guassian kernel size only support odd integer.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::MEDIAN_FILTER_KERNEL_INVALID:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Median kernel size only support odd integer.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::TOO_MUCH_CC_TO_REMOVE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "There are too many connected components to remove.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::TMPL_IS_BIGGER_THAN_ROI:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The template is bigger than the search window.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::PICK_PT_NOT_IN_ROI:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The pick point must in ROI.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::FAILED_TO_FIND_CHESS_BOARD_BLOCK_SIZE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to find chessboard block size.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::FAILED_TO_FIND_FIRST_CB_CORNER:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to find first chessboard corner to calibrate camera.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::FAILED_TO_FIND_ENOUGH_CB_CORNERS:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to find enough chessboard corners to calibrate camera.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::CHESSBOARD_PATTERN_NOT_CORRECT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The chessboard pattern is not correct.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::CAN_NOT_FIND_CAE_LINE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Can not find the line of CAE.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::CAN_NOT_FIND_SQUARE_EDGE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Can not find square chip edges.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::CAN_NOT_FIND_CIRCULAR_CHIP:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Can not find circular chip.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::CAN_NOT_FIND_CHIP_BODY:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Can not find chip body.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_STATUS_FATAL_ERROR;
        break;
    case VisionStatus::CALIPER_CAN_NOT_FIND_LINE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Caliper can not find the line.");
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