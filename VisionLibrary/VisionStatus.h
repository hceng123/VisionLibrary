#ifndef _VISION_STATUS_H_
#define _VISION_STATUS_H_

namespace AOI
{
namespace Vision
{

enum class PR_STATUS_ERROR_LEVEL
{  
    PR_STATUS_NO_ERROR,					// No error status
    PR_STATUS_WARNING,					// Warning status
    PR_STATUS_INFORMATION,				// Informative
    PR_STATUS_GENERAL_ERROR,			// Unexpected output but no need to restart application
    PR_STATUS_FATAL_ERROR				// Fatal error and need to restart application
};

enum class VisionStatus
{
	OK,
    INVALID_PARAM,
    SRCH_TMPL_FAIL,
    OPEN_FILE_FAIL,
    OBJECT_MISSING,
    OBJECT_SHIFT,           //5
    OBJECT_ROTATE,
    OBJECT_SCALE_FAIL,
    INVALID_RECORD_TYPE,
    FIND_EDGE_FAIL,
    FIND_ELECTRODE_FAIL,    //10
    NOT_SUPPORTED_OPTION,
    PATH_NOT_EXIST,
    INVALID_LOGCASE,
    LEARN_FAIL,
    NOT_ENOUGH_POINTS_TO_FIT,
    TOO_MUCH_NOISE_TO_FIT,
    FAIL_TO_FIT_CIRCLE,
    OPENCV_EXCEPTION,
    OCR_FAIL,
    GUASSIAN_FILTER_KERNEL_INVALID,
    MEDIAN_FILTER_KERNEL_INVALID,
    TOO_MUCH_CC_TO_REMOVE,
    TMPL_IS_BIGGER_THAN_ROI,
    PICK_PT_NOT_IN_ROI,
};

}
}

#endif