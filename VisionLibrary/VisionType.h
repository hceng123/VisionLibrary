#ifndef _VISION_TYPE_H_
#define _VISION_TYPE_H_

namespace AOI
{
namespace Vision
{

const int MAX_NUM_OF_DEFECT_CRITERIA =          5;
const int MAX_NUM_OF_DEFECT_RESULT =            20;
const float PARALLEL_LINE_SLOPE_DIFF_LMT =		0.1f;
const float PARALLEL_LINE_MERGE_DIST_LMT =      10;
const int PR_ELECTRODE_COUNT =                  2;
const int PR_MAX_GRAY_LEVEL =                   255;
const int PR_MIN_GRAY_LEVEL =                   0;
const int PR_RECT_EDGE_COUNT =                  4;
const int PR_MAX_ERR_STR_LEN =                  100;
const int PR_AUTO_THRESHOLD_MAX_NUM =           4;
const int PR_FIT_LINE_MAX_POINT_COUNT =         10000;
const int PR_FIT_CIRCLE_MAX_POINT =             10000;
const double PR_MM_TO_UM =                      1000.;
const int PR_GROUP_TEXTURE_IMG_COUNT =          4;
const int PR_MAX_AUTO_THRESHOLD_COUNT =         10;

enum class PR_SRCH_OBJ_ALGORITHM {
	SIFT,
	SURF,
};

enum class PR_DEFECT_ATTRIBUTE {
	BRIGHT, // Only bright defect (those defects brighter than the reference and the 
	// absolute value of the difference is larger than the defect threshold) is detected. 
	DARK,	// Only dark defect (those defects darker than the reference and the absolute 
	// value of the difference is larger than the defect threshold) is detected
	BOTH,	// Both bright and dark defect is detected
	BRIGHT_LOOP,	// special for AOI
	DARK_LOOP,		// special for AOI
	END
};

enum class PR_DEFECT_TYPE {
	BLOB,
	LINE,
	END,
};

enum class PR_DEBUG_MODE {
    DISABLED,
    SHOW_IMAGE,
    LOG_ALL_CASE,
    LOG_FAIL_CASE,
};

enum class PR_RECORD_TYPE {
    INVALID = -1,
    OBJECT,
    DEVICE,
    CHIP,
    CONTOUR,
    TEMPLATE,
    LEAD,
};

enum class PR_FIDUCIAL_MARK_TYPE {
    SQUARE,
    CIRCLE,
};

enum class PR_RM_FIT_NOISE_METHOD {
    ABSOLUTE_ERR,   //Remove the point with absolute error over the error tolerance.
    POSITIVE_ERR,   //Remove the point with positive error over the error tolerance. If it is circle, remove the standalone outter points of the fitted circle.
    NEGATIVE_ERR,   //Remove the point with negative error over the error tolerance. If it is circle, remove the standalone innter points of the fitted circle.
};

enum class PR_EDGE_DIRECTION {
    HORIZONTAL,
    VERTIAL,
    BOTH,
};

enum class PR_FIT_METHOD {
    LEAST_SQUARE,
    LEAST_SQUARE_REFINE,
    RANSAC,
};

enum class PR_DIRECTION {
    UP,
    DOWN,
    LEFT,
    RIGHT,
};

enum class PR_OBJECT_ATTRIBUTE {
    BRIGHT,
	DARK,
    END,
};

enum class PR_FILTER_TYPE {
    NORMALIZED_BOX_FILTER,
    GAUSSIAN_FILTER,
    MEDIAN_FILTER,
    BILATERIAL_FILTER,
    END,
};

enum class PR_COMPARE_TYPE {
    SMALLER,
    LARGER,
};

//If the dark part coordinate is larger than bright part, then choose BRIGHT_TO_DARK mode.
//In other words, dark part on the right, then use BRIGHT_TO_DARK mode.
//If dark part on the left, then use DARK_TO_BRIGHT.
enum class PR_CALIPER_DIR {
    DARK_TO_BRIGHT,
    BRIGHT_TO_DARK,
    AUTO,
};

enum class PR_FIND_LINE_ALGORITHM {
    PROJECTION,
    CALIPER,
};

enum class PR_CALIPER_SELECT_EDGE {
    MAX_EDGE,
    FIRST_EDGE,
};

enum class PR_FILL_HOLE_METHOD {
    CONTOUR,
    MORPHOLOGY,
};

enum class PR_MATCH_TMPL_ALGORITHM {
    SQUARE_DIFF,
    HIERARCHICAL_EDGE       = 10,
    HIERARCHICAL_AREA       = 11,
};

enum class PR_OBJECT_MOTION {
    TRANSLATION,    //Only move in XY
    EUCLIDEAN,      //Move and rotation in XY
    AFFINE,         //Move and rotation in XY, and have one direction shape change.
    HOMOGRAPHY      //Move and rotation in XY, and have multiple shape change.
};

enum class PR_INSP_BRIDGE_MODE {
    INNER,
    OUTER,
};

enum class PR_INSP_CHIP_MODE {
    HEAD,
    BODY,
    SQUARE,
    RNET,
    CAE,
    CIRCULAR,
};

enum class PR_IMG_SEGMENT_METHOD {
    GRAY_SCALE_RANGE,
    COLOR_RANGE,
};

enum class PR_INSP_HOLE_MODE {
    RATIO,
    BLOB,
    SIM,
};

enum class PR_FIND_LEAD_END_METHOD {
    AVERAGE,
    INDIVIDUAL,
    FIXED,
};

enum class PR_SCAN_IMAGE_DIR {
    LEFT_TO_RIGHT,
    RIGHT_TO_LEFT,
};

enum class PR_AUTO_LOCATE_LEAD_METHOD {
    TEMPLATE_MATCH,
    FIND_CONTOUR,
};

}
}

#endif