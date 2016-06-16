#ifndef PR_TYPE_H
#define PR_TYPE_H

namespace AOI
{
namespace Vision
{

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

}
}

#endif