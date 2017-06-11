#ifndef _RECORD_H_
#define _RECORD_H_

#include "BaseType.h"
#include "VisionHeader.h"
#include "opencv2/core.hpp"
#include <memory>

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
    PR_RECORD_TYPE  _enType;
    const String    _strKeyType     = "type";
};
using IRecordPtr = std::shared_ptr<IRecord>;

class ObjRecord : public IRecord
{
public:
    ObjRecord(PR_RECORD_TYPE enType):IRecord(enType)   {}
    virtual VisionStatus load(cv::FileStorage &fileStorage) override;
    virtual VisionStatus save(const String& strFilePath) override;
    void setModelKeyPoint(const VectorOfKeyPoint vecModelKeyPoint);
    const VectorOfKeyPoint& getModelKeyPoint() const;
    void setModelDescriptor(const cv::Mat &matModelDescritor);
    const cv::Mat& getModelDescriptor() const;
    void setObjCenter(const cv::Point2f ptObjCenter);
    cv::Point2f getObjCenter() const;
protected:
    cv::Point2f                 _ptObjCenter;
    VectorOfKeyPoint            _vecModelKeyPoint;
	cv::Mat                     _matModelDescritor;
};
using ObjRecordPtr = std::shared_ptr<ObjRecord>;

class DeviceRecord : public IRecord
{
public:
    DeviceRecord(PR_RECORD_TYPE enType):IRecord(enType)   {}
    virtual VisionStatus load(cv::FileStorage &fileStorage) override;
    virtual VisionStatus save(const String& strFilePath) override;
    void setSize(const cv::Size2f &size);
    const cv::Size2f& getSize() const;
    void setElectrodeThreshold(Int16 nElectrodeThreshold);
    Int16 getElectrodeThreshold() const;
protected:
    cv::Size2f          _size;
    Int16               _nElectrodeThreshold;
};
using DeviceRecordPtr = std::shared_ptr<DeviceRecord>;

class ChipRecord : public IRecord
{
public:
    ChipRecord(PR_RECORD_TYPE enType) : IRecord(enType)   {}
    virtual VisionStatus load(cv::FileStorage &fileStorage) override;
    virtual VisionStatus save(const String& strFilePath) override;
    void setInspMode ( PR_INSP_CHIP_MODE enInspMode );
    PR_INSP_CHIP_MODE getInspMode() const;
    void setSize(const cv::Size2f &size);
    const cv::Size2f& getSize() const;
    void setThreshold(Int16 nThreshold);
    Int16 getThreshold() const;
protected:
    PR_INSP_CHIP_MODE   _enInspMode;
    cv::Size2f          _size;
    Int16               _nThreshold;

    String              _strKeyInspMode     = "InspMode";
    String              _strKeySize         = "Size";
    String              _strKeyThreshold    = "Threshold";
};
using ChipRecordPtr = std::shared_ptr<ChipRecord>;

}
}

#endif