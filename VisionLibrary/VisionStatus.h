#ifndef _VISION_STATUS_H_
#define _VISION_STATUS_H_

namespace AOI
{
namespace Vision
{

enum class PR_STATUS_ERROR_LEVEL
{  
    NO_ERROR,               // No error status.
    INSP_STATUS,            // Normal inspection status.
    LEARN_FAIL,             // Learn fail, may need to investigate or report to developer engineer.
    FATAL_ERROR,            // Fatal error need to report to developer engineer.
};

enum class VisionStatus
{
	OK,

    //Fatal system Error
    OPENCV_EXCEPTION,
    INVALID_PARAM,
    PATH_NOT_EXIST,
    INVALID_LOGCASE,    
    OPEN_FILE_FAIL,
    INVALID_RECORD_FILE,
    INVALID_RECORD_TYPE,    
    GUASSIAN_FILTER_KERNEL_INVALID,
    MEDIAN_FILTER_KERNEL_INVALID,

    LEARN_OBJECT_FAIL,
    SRCH_OBJ_FAIL,
    OBJECT_SHIFT,
    OBJECT_ROTATE,
    OBJECT_SCALE_FAIL,
    
    FIND_DEVICE_EDGE_FAIL,
    FIND_ELECTRODE_FAIL,    
    NOT_ENOUGH_POINTS_TO_FIT,
    TOO_MUCH_NOISE_TO_FIT,
    FAIL_TO_FIT_CIRCLE,
    
    OCR_FAIL,    
    TOO_MUCH_CC_TO_REMOVE,
    PICK_PT_NOT_IN_ROI,
    FAILED_TO_FIND_CHESS_BOARD_BLOCK_SIZE,
    FAILED_TO_FIND_FIRST_CB_CORNER,
    FAILED_TO_FIND_ENOUGH_CB_CORNERS,
    CHESSBOARD_PATTERN_NOT_CORRECT,
    FAILED_TO_FIND_LEAD,

    //Learn and inspect chip status
    CAN_NOT_FIND_CAE_LINE,      //Can not find the line of capacitor
    CAN_NOT_FIND_SQUARE_EDGE,
    CAN_NOT_FIND_CIRCULAR_CHIP,
    CAN_NOT_FIND_CHIP_BODY,    
    
    //Learn and inspect contour status
    CAN_NOT_FIND_CONTOUR,
    CONTOUR_DEFECT_REJECT,

    //Inspect hole status
    RATIO_UNDER_LIMIT,
    RATIO_OVER_LIMIT,
    BLOB_COUNT_OUT_OF_RANGE,

    //Inspect lead status
    NOT_FIND_LEAD,

    //3D Status
    CALIB_3D_HEIGHT_SURFACE_TOO_SMALL   = 100,
    CALIB_3D_HEIGHT_NO_BASE_STEP        = 101,

    //Camera MTF
    CAN_NOT_FIND_MTF_PATTERN            = 150,
    MTF_PATTERN_TOO_SMALL               = 151,

    //Caliper related errors
    CALIPER_CAN_NOT_FIND_LINE           = 200,
    PROJECTION_CANNOT_FIND_LINE         = 201,
    CALIPER_NOT_ENOUGH_EDGE_POINTS      = 202,
};

}
}

#endif /*_VISION_STATUS_H_*/