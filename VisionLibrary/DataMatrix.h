#ifndef _DATA_MATRIX_H_
#define _DATA_MATRIX_H_

#include "BaseType.h"
#include "opencv2/core.hpp"

namespace AOI
{
namespace Vision
{

class DataMatrix
{
    DataMatrix();
    DataMatrix(DataMatrix const &);
    DataMatrix &operator=(DataMatrix const &);
    ~DataMatrix();

public:
    static bool read(const cv::Mat &matInput, String &strResult);

};

}
}

#endif /*_DATA_MATRIX_H_*/
