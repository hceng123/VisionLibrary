#include "VisionHeader.h"

namespace AOI
{
namespace Vision
{

VisionStatus GetErrorInfo(VisionStatus enStatus, PR_GET_ERROR_INFO_RPY *pstRpy)
{
    switch (enStatus)
    {
    case VisionStatus::OK:
        pstRpy->achErrorStr[0] = '\0';
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_NO_ERROR;
        break;

    case VisionStatus::OPENCV_EXCEPTION:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Exception encounted, please check log file.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_FATAL_ERROR;
        break;

    case VisionStatus::INVALID_PARAM:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Input parameter is invalid.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_FATAL_ERROR;
        break;

    case VisionStatus::PATH_NOT_EXIST:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The input logcase path not exist.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_FATAL_ERROR;
        break;

    case VisionStatus::INVALID_LOGCASE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The input logcase file is invalid.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_FATAL_ERROR;
        break;

    case VisionStatus::OPEN_FILE_FAIL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to open file.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_FATAL_ERROR;
        break;

    case VisionStatus::INVALID_RECORD_TYPE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The record type is invalid.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_FATAL_ERROR;
        break;
    
    case VisionStatus::GUASSIAN_FILTER_KERNEL_INVALID:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Guassian kernel size only support odd integer.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_FATAL_ERROR;
        break;
    case VisionStatus::MEDIAN_FILTER_KERNEL_INVALID:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Median kernel size only support odd integer.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_FATAL_ERROR;
        break;

    //The inspection and learn status.
    case VisionStatus::LEARN_OBJECT_FAIL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Learn object fail.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_LEARN_FAIL;
        break;

    case VisionStatus::SRCH_OBJ_FAIL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Search object fail.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::OBJECT_SHIFT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Object shift over tolerance.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::OBJECT_ROTATE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Object rotation over tolerance.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::OBJECT_SCALE_FAIL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Object scale over tolerance.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::MATCH_SCORE_REJECT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Template match score is under minimum value.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::FIND_ELECTRODE_FAIL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to find electrode of chip.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;    

    case VisionStatus::FIND_CHESS_BOARD_BLOCK_SIZE_FAIL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to find chessboard block size.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::FIND_FIRST_CB_CORNER_FAIL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to find first chessboard corner to calibrate camera.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::NOT_FIND_ENOUGH_CB_CORNERS:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to find enough chessboard corners to calibrate camera.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::CHESSBOARD_PATTERN_NOT_CORRECT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The chessboard pattern is not correct.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::FIND_DEVICE_EDGE_FAIL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to find device edge.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;
    case VisionStatus::NOT_ENOUGH_POINTS_TO_FIT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Not enough points to fit.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;
    case VisionStatus::TOO_MUCH_NOISE_TO_FIT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Too much noise to fit.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;
    case VisionStatus::FAIL_TO_FIT_CIRCLE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to fit circle, the fitting result is invalid.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;    
    case VisionStatus::TOO_MUCH_CC_TO_REMOVE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "There are too many connected components to remove.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::CAN_NOT_FIND_CAE_LINE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Cannot find the line of CAE.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::CAN_NOT_FIND_SQUARE_EDGE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Cannot find square chip edges.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::CAN_NOT_FIND_CIRCULAR_CHIP:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Cannot find circular chip.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::CAN_NOT_FIND_CHIP_BODY:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Cannot find chip body.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::CAN_NOT_FIND_CONTOUR:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Cannot find contour.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::CONTOUR_DEFECT_REJECT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Rejected by contour defect.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::RATIO_UNDER_LIMIT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The ratio under limitation.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::RATIO_OVER_LIMIT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The ratio over limitation.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::BLOB_COUNT_OUT_OF_RANGE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The blob count is out of range.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::NOT_FIND_LEAD:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Lead is missing.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::AUTO_LOCATE_LEAD_FAIL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to auto locate lead position.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::LEAD_OFFSET_OVER_LIMIT:
         _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Lead offset over limit.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::POLARITY_FAIL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Polarity difference is lower than the requirement.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::CALIB_3D_HEIGHT_SURFACE_TOO_SMALL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The step area is too small to calibrate 3D height.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::CALIB_3D_HEIGHT_NO_BASE_STEP:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The H 0 step is not found.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::BRIDGE_DEFECT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The inspect window with bridge defect.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::FAILED_TO_SPLIT_IMAGE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Failed to split the image.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::CAN_NOT_FIND_MTF_PATTERN:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Cannot find MTF pattern in the given ROI.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::MTF_PATTERN_TOO_SMALL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "The MTF pattern is too small.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::CALIPER_CAN_NOT_FIND_LINE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Caliper cannot find the line.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::PROJECTION_CANNOT_FIND_LINE:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Projection cannot find the line.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::CALIPER_NOT_ENOUGH_EDGE_POINTS:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Caliper cannot enough edge points.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::LINE_LINEARITY_REJECT:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Line linearity is under minimum value.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    case VisionStatus::LINE_ANGLE_OUT_OF_TOL:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Line angle difference is out of tolerance.");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_INSP_STATUS;
        break;

    default:
        _snprintf(pstRpy->achErrorStr, PR_MAX_ERR_STR_LEN, "Undefined error code!");
        pstRpy->enErrorLevel = PR_STATUS_ERROR_LEVEL::PR_FATAL_ERROR;
        break;
    }
    return VisionStatus::OK;
}

}
}