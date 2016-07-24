#include "Record.h"

namespace AOI
{
namespace Vision
{

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
    FileStorage fs(strFilePath, FileStorage::WRITE);
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

}
}
