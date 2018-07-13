#ifndef _VISION_STATUS_H_
#define _VISION_STATUS_H_

namespace AOI
{
namespace Vision
{

enum class PR_STATUS_ERROR_LEVEL
{  
    PR_NO_ERROR,               // No error status.
    PR_INSP_STATUS,            // Normal inspection status.
    PR_LEARN_FAIL,             // Learn fail, may need to investigate or report to developer engineer.
    PR_FATAL_ERROR,            // Fatal error need to report to developer engineer.
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

    //Srch object and template match
    LEARN_OBJECT_FAIL                   = 20,
    SRCH_OBJ_FAIL                       = 21,
    OBJECT_SHIFT                        = 22,
    OBJECT_ROTATE                       = 23,
    OBJECT_SCALE_FAIL                   = 24,
    MATCH_SCORE_REJECT                  = 25,

    //Calibrate Camera
    FIND_CHESS_BOARD_BLOCK_SIZE_FAIL    = 30,
    FIND_FIRST_CB_CORNER_FAIL           = 31,
    NOT_FIND_ENOUGH_CB_CORNERS          = 32,
    CHESSBOARD_PATTERN_NOT_CORRECT      = 33,
    
    //Assist function fail.
    FIND_DEVICE_EDGE_FAIL               = 40,
    FIND_ELECTRODE_FAIL                 = 41,    
    NOT_ENOUGH_POINTS_TO_FIT            = 42,
    TOO_MUCH_NOISE_TO_FIT               = 43,
    FAIL_TO_FIT_CIRCLE                  = 44,    
    OCR_FAIL                            = 45,    
    TOO_MUCH_CC_TO_REMOVE               = 46,    

    //Learn and inspect chip status
    CAN_NOT_FIND_CAE_LINE               = 50,   //Can not find the line of capacitor
    CAN_NOT_FIND_SQUARE_EDGE            = 51,
    CAN_NOT_FIND_CIRCULAR_CHIP          = 52,
    CAN_NOT_FIND_CHIP_BODY              = 53,    
    
    //Learn and inspect contour status
    CAN_NOT_FIND_CONTOUR                = 60,
    CONTOUR_DEFECT_REJECT               = 61,

    //Inspect hole status
    RATIO_UNDER_LIMIT                   = 70,
    RATIO_OVER_LIMIT                    = 71,
    BLOB_COUNT_OUT_OF_RANGE             = 72,

    //Inspect lead status
    NOT_FIND_LEAD                       = 80,
    AUTO_LOCATE_LEAD_FAIL               = 81,
    LEAD_OFFSET_OVER_LIMIT              = 82,

    //Inspect polarity status
    POLARITY_FAIL                       = 90,

    //3D Status
    CALIB_3D_HEIGHT_SURFACE_TOO_SMALL   = 100,
    CALIB_3D_HEIGHT_NO_BASE_STEP        = 101,

    //Bridge status
    BRIDGE_DEFECT                       = 120,

    //Camera MTF
    CAN_NOT_FIND_MTF_PATTERN            = 150,
    MTF_PATTERN_TOO_SMALL               = 151,

    //Caliper related errors
    CALIPER_CAN_NOT_FIND_LINE           = 200,
    PROJECTION_CANNOT_FIND_LINE         = 201,
    CALIPER_NOT_ENOUGH_EDGE_POINTS      = 202,
    LINE_LINEARITY_REJECT               = 203,
    LINE_ANGLE_OUT_OF_TOL               = 204,
};

}
}

#endif /*_VISION_STATUS_H_*/