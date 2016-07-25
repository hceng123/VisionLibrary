#ifndef _VISION_STATUS_H_
#define _VISION_STATUS_H_

namespace AOI
{
namespace Vision
{

enum class VisionStatus
{
	OK,
    INVALID_PARAM,
    SRCH_TMPL_FAIL,
    OPEN_FILE_FAIL,
    OBJECT_MISSING,
    OBJECT_SHIFT,
    OBJECT_ROTATE,
    OBJECT_SCALE_FAIL,
    INVALID_RECORD_TYPE,
    FIND_EDGE_FAIL,
    FIND_ELECTRODE_FAIL,
};

}
}

#endif