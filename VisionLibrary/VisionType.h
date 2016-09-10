#ifndef _VISION_TYPE_H_
#define _VISION_TYPE_H_

namespace AOI
{
namespace Vision
{

#define MAX_NUM_OF_DEFECT_CRITERIA				(5)
#define MAX_NUM_OF_DEFECT_RESULT				(20)
#define PARALLEL_LINE_SLOPE_DIFF_LMT			(0.1)
#define PARALLEL_LINE_MERGE_DIST_LMT			(20)
#define DEVICE_ELECTRODE_COUNT                  (2)
#define PR_MAX_GRAY_LEVEL                       (256)

enum class PR_ALIGN_ALGORITHM
{
	SIFT,
	SURF,
};

enum class PR_DEFECT_ATTRIBUTE
{
	BRIGHT, // Only bright defect (those defects brighter than the reference and the 
	// absolute value of the difference is larger than the defect threshold) is detected. 
	DARK,	// Only dark defect (those defects darker than the reference and the absolute 
	// value of the difference is larger than the defect threshold) is detected
	BOTH,	// Both bright and dark defect is detected
	BRIGHT_LOOP,	// special for AOI
	DARK_LOOP,		// special for AOI
	END
};

enum class PR_DEFECT_TYPE
{
	BLOB,
	LINE,
	END,
};

enum class PR_DEBUG_MODE
{
    DISABLED,
    SHOW_IMAGE,
    LOG_ALL_CASE,
    LOG_FAIL_CASE,
};

enum class PR_RECORD_TYPE
{
    INVALID = -1,
    ALIGNMENT,
    DEVICE,
    CHIP,
    LEAD,
};

enum class PR_FIDUCIAL_MARK_TYPE
{
    SQUARE,
    CIRCLE,
};

}
}

#endif