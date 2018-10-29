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

class Record
{
public:
    Record(PR_RECORD_TYPE enType) : _enType(enType) {}
    virtual ~Record() = default;
    virtual VisionStatus load(cv::FileStorage &fileStorage, const String& strFilePath = "") = 0;
    virtual VisionStatus save(const String& strFilePath) = 0;
    virtual cv::Mat getImage() const = 0;
    virtual PR_RECORD_TYPE getType() { return _enType; };

protected:
    PR_RECORD_TYPE  _enType;
    const String    _strKeyType         = "type";
    String          _strTmplFileName    = "Tmpl.png";
    String          _strMaskFileName    = "Mask.png";
};
using RecordPtr = std::shared_ptr<Record>;

class ObjRecord : public Record
{
public:
    ObjRecord(PR_RECORD_TYPE enType):Record(enType)   {}
    virtual VisionStatus load(cv::FileStorage &fileStorage, const String& strFilePath = "") override;
    virtual VisionStatus save(const String& strFilePath) override;
    void setModelKeyPoint(const VectorOfKeyPoint vecModelKeyPoint);
    const VectorOfKeyPoint& getModelKeyPoint() const;
    void setModelDescriptor(const cv::Mat &matModelDescritor);
    const cv::Mat& getModelDescriptor() const;
    void setObjCenter(const cv::Point2f ptObjCenter);
    cv::Point2f getObjCenter() const;
    virtual cv::Mat getImage() const { return cv::Mat(); }

private:
    cv::Point2f                 _ptObjCenter;
    VectorOfKeyPoint            _vecModelKeyPoint;
	cv::Mat                     _matModelDescritor;
};
using ObjRecordPtr = std::shared_ptr<ObjRecord>;

class DeviceRecord : public Record
{
public:
    DeviceRecord(PR_RECORD_TYPE enType):Record(enType)   {}
    virtual VisionStatus load(cv::FileStorage &fileStorage, const String& strFilePath = "") override;
    virtual VisionStatus save(const String& strFilePath) override;
    void setSize(const cv::Size2f &size);
    const cv::Size2f& getSize() const;
    void setElectrodeThreshold(Int16 nElectrodeThreshold);
    Int16 getElectrodeThreshold() const;
    virtual cv::Mat getImage() const { return cv::Mat(); }

private:
    cv::Size2f          _size;
    Int16               _nElectrodeThreshold;
};
using DeviceRecordPtr = std::shared_ptr<DeviceRecord>;

class ChipRecord : public Record
{
public:
    ChipRecord(PR_RECORD_TYPE enType) : Record(enType) {}
    virtual VisionStatus load(cv::FileStorage &fileStorage, const String& strFilePath = "") override;
    virtual VisionStatus save(const String& strFilePath) override;
    void setInspMode(PR_INSP_CHIP_MODE enInspMode);
    PR_INSP_CHIP_MODE getInspMode() const;
    void setSize(const cv::Size2f &size);
    const cv::Size2f& getSize() const;
    void setThreshold(Int16 nThreshold);
    Int16 getThreshold() const;
    void setTmpl(const cv::Mat &matTmpl) { _matTmpl = matTmpl; }
    virtual cv::Mat getImage() const { return _matTmpl; }

private:
    PR_INSP_CHIP_MODE   _enInspMode;
    cv::Size2f          _size;
    Int16               _nThreshold;
    cv::Mat             _matTmpl;

    String              _strKeyInspMode     = "InspMode";
    String              _strKeySize         = "Size";
    String              _strKeyThreshold    = "Threshold";
};
using ChipRecordPtr = std::shared_ptr<ChipRecord>;

class ContourRecord : public Record
{
public:
    ContourRecord(PR_RECORD_TYPE enType) : Record(enType)   {}
    virtual VisionStatus load(cv::FileStorage &fileStorage, const String& strFilePath = "") override;
    virtual VisionStatus save(const String& strFilePath) override;
    void setSize(const cv::Size2f &size);
    const cv::Size2f& getSize() const;
    void setThreshold(Int16 nThreshold);
    Int16 getThreshold() const;
    cv::Mat getTmpl() const;
    void setTmpl(const cv::Mat &matTmpl);
    void setContourMat(const cv::Mat &matContour);
    void setContour ( const VectorOfVectorOfPoint &vecContours );
    VectorOfVectorOfPoint getContour() const;
    String getTmplFileName() const;
    String getContourFileName() const;
    virtual cv::Mat getImage() const { return _matTmpl; }

private:
    Int16                   _nThreshold;
    cv::Mat                 _matTmpl;
    cv::Mat                 _matContour;
    VectorOfVectorOfPoint   _vecContours;

    String                  _strKeyThreshold    = "Threshold";
    String                  _strContourFileName = "Contour.png";
};
using ContourRecordPtr = std::shared_ptr<ContourRecord>;

class TmplRecord : public Record
{
public:
    TmplRecord(
        PR_RECORD_TYPE          enType,
        PR_MATCH_TMPL_ALGORITHM enAlgorithm,
        const cv::Mat          &matTmpl,
        const cv::Mat          &matMask,
        const cv::Mat          &matEdgeMask) :
        Record(enType),
        _enAlgorithm(enAlgorithm),
        _matTmpl(matTmpl),
        _matMask(matMask),
        _matEdgeMask(matEdgeMask) {}
    TmplRecord(PR_RECORD_TYPE enType) : Record(enType)   {}
    virtual VisionStatus load(cv::FileStorage &fileStorage, const String& strFilePath = "") override;
    virtual VisionStatus save(const String& strFilePath) override;
    void setAlgorithm(PR_MATCH_TMPL_ALGORITHM enAlgorithm);
    PR_MATCH_TMPL_ALGORITHM getAlgorithm() const;
    void setTmpl(const cv::Mat &matTmpl);
    cv::Mat getTmpl() const;
    void setEdgeMask(const cv::Mat &matTmpl);
    cv::Mat getEdgeMask() const;
    cv::Mat getMask() const { return _matMask; }
    virtual cv::Mat getImage() const { return _matTmpl; }

private:
    PR_MATCH_TMPL_ALGORITHM _enAlgorithm;
    cv::Mat                 _matTmpl;
    cv::Mat                 _matMask;
    cv::Mat                 _matEdgeMask;
    String                  _strKeyAlgorithm        = "Algorithm";
    String                  _strEdgeMaskFileName    = "EdgeMask.png";
};
using TmplRecordPtr = std::shared_ptr<TmplRecord>;

class OcvRecord : public Record
{
public:
    OcvRecord(PR_RECORD_TYPE enType = PR_RECORD_TYPE::OCV) : Record(enType) {}
    OcvRecord(const cv::Mat &matBigTmpl, const VectorOfRect &vecCharRects, PR_RECORD_TYPE enType = PR_RECORD_TYPE::OCV);
    virtual VisionStatus load(cv::FileStorage &fs, const String& strFilePath = "") override;
    virtual VisionStatus save(const String& strFilePath) override;
    void setBigTmpl(const cv::Mat &matBigTmpl);
    cv::Mat getBigTmpl() const;
    VectorOfRect getCharRects() const;
    void setCharRects(const VectorOfRect &vecRects);
    virtual cv::Mat getImage() const { return _matBigTmpl; }

private:
    cv::Mat                 _matBigTmpl;
    VectorOfRect            _vecCharRects;
    String                  _strKeyCharCount    = "CharCount";
    String                  _strKeyCharRect     = "CharRect_";
};
using OcvRecordPtr = std::shared_ptr<OcvRecord>;

}
}

#endif