#ifndef _RECORD_H_
#define _RECORD_H_

#include "BaseType.h"
#include "VisionHeader.h"
#include "opencv2/core/core.hpp"

namespace AOI
{
namespace Vision
{

class IRecord
{
public:
    IRecord(PR_RECORD_TYPE enType):_enType(enType)   {}
    virtual VisionStatus load(cv::FileStorage &fileStorage) = 0;
    virtual VisionStatus save(const String& strFilePath) = 0;
    virtual PR_RECORD_TYPE getType() { return _enType;};
protected:
    PR_RECORD_TYPE                       _enType;
};

class TmplRecord:public IRecord
{
public:
    TmplRecord(PR_RECORD_TYPE enType):IRecord(enType)   {}
    virtual VisionStatus load(cv::FileStorage &fileStorage) override;
    virtual VisionStatus save(const String& strFilePath) override;
    void setModelKeyPoint(const VectorOfKeyPoint vecModelKeyPoint);
    const VectorOfKeyPoint& getModelKeyPoint() const;
    void setModelDescriptor(const cv::Mat &matModelDescritor);
    const cv::Mat& getModelDescriptor() const;
protected:
    VectorOfKeyPoint            _vecModelKeyPoint;
	cv::Mat                     _matModelDescritor;
};

}
}

#endif