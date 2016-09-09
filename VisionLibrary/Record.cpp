#include "Record.h"

namespace AOI
{
namespace Vision
{

/******************************************
* Template Record *
******************************************/
VisionStatus TmplRecord::load(cv::FileStorage &fs)
{
    cv::FileNode fileNode = fs["keypoints"];
    cv::read(fileNode, _vecModelKeyPoint);

    fileNode = fs["descriptor"];
    cv::read(fileNode, _matModelDescritor);
    return VisionStatus::OK;
}

VisionStatus TmplRecord::save(const String& strFilePath)
{
    cv::FileStorage fs(strFilePath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return VisionStatus::OPEN_FILE_FAIL;

    write ( fs, "type", static_cast<int>(PR_RECORD_TYPE::ALIGNMENT));
    write ( fs, "keypoints", _vecModelKeyPoint );
    write ( fs, "descriptor", _matModelDescritor );
    fs.release();
    return VisionStatus::OK;
}

void TmplRecord::setModelKeyPoint(const VectorOfKeyPoint vecModelKeyPoint)
{
    _vecModelKeyPoint = vecModelKeyPoint;
}

const VectorOfKeyPoint& TmplRecord::getModelKeyPoint() const
{
    return _vecModelKeyPoint;
}

void TmplRecord::setModelDescriptor(const cv::Mat &matModelDescritor)
{
    _matModelDescritor = matModelDescritor;
}

const cv::Mat& TmplRecord::getModelDescriptor() const
{
    return _matModelDescritor;
}

/******************************************
* Device Record *
******************************************/
VisionStatus DeviceRecord::load(cv::FileStorage &fs)
{
    cv::FileNode fileNode = fs["size"];
    cv::read<float>(fileNode, _size, cv::Size2f(0, 0));

    fileNode = fs["threshold"];
    cv::read(fileNode, _nElectrodeThreshold, 0 );
    return VisionStatus::OK;
}

VisionStatus DeviceRecord::save(const String& strFilePath)
{
    cv::FileStorage fs(strFilePath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return VisionStatus::OPEN_FILE_FAIL;

    write ( fs, "type", static_cast<int>(PR_RECORD_TYPE::DEVICE));
    write ( fs, "size", _size);
    write ( fs, "threshold", _nElectrodeThreshold );
    fs.release();
    return VisionStatus::OK;
}

void DeviceRecord::setSize(const cv::Size2f &size)
{
    _size = size;
}

const cv::Size2f& DeviceRecord::getSize() const
{
    return _size;
}

void DeviceRecord::setElectrodeThreshold(Int16 nElectrodeThreshold)
{
    _nElectrodeThreshold;
}

Int16 DeviceRecord::getElectrodeThreshold() const
{
    return _nElectrodeThreshold;
}

}
}
