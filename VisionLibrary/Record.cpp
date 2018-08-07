#include "Record.h"
#include "Config.h"
#include "opencv2/highgui.hpp"
#include "FileUtils.h"

namespace AOI
{
namespace Vision
{

/******************************************
* Object Record *
******************************************/
VisionStatus ObjRecord::load(cv::FileStorage &fs, const String& strFilePath) {
    cv::FileNode fileNode = fs["keypoints"];
    cv::read(fileNode, _vecModelKeyPoint);

    fileNode = fs["descriptor"];
    cv::read(fileNode, _matModelDescritor);
    return VisionStatus::OK;
}

VisionStatus ObjRecord::save(const String& strFilePath) {
    cv::FileStorage fs(strFilePath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return VisionStatus::OPEN_FILE_FAIL;

    cv::write ( fs, "type", static_cast<int>(PR_RECORD_TYPE::OBJECT));
    cv::write ( fs, "keypoints", _vecModelKeyPoint );
    cv::write ( fs, "descriptor", _matModelDescritor );
    fs.release();
    return VisionStatus::OK;
}

void ObjRecord::setModelKeyPoint(const VectorOfKeyPoint vecModelKeyPoint) {
    _vecModelKeyPoint = vecModelKeyPoint;
}

const VectorOfKeyPoint& ObjRecord::getModelKeyPoint() const {
    return _vecModelKeyPoint;
}

void ObjRecord::setModelDescriptor(const cv::Mat &matModelDescritor) {
    _matModelDescritor = matModelDescritor;
}

const cv::Mat& ObjRecord::getModelDescriptor() const {
    return _matModelDescritor;
}

void ObjRecord::setObjCenter(const cv::Point2f ptObjCenter) {
    _ptObjCenter = ptObjCenter;
}

cv::Point2f ObjRecord::getObjCenter() const {
    return _ptObjCenter;
}

/******************************************
* Device Record *
******************************************/
VisionStatus DeviceRecord::load(cv::FileStorage &fs, const String& strFilePath) {
    cv::FileNode fileNode = fs["size"];
    cv::read<float>(fileNode, _size, cv::Size2f(0, 0));

    fileNode = fs["threshold"];
    cv::read(fileNode, _nElectrodeThreshold, 0 );
    return VisionStatus::OK;
}

VisionStatus DeviceRecord::save(const String& strFilePath) {
    String strParamFilePath = strFilePath + "/" + Config::GetInstance()->getRecordParamFile();
    cv::FileStorage fs(strParamFilePath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return VisionStatus::OPEN_FILE_FAIL;

    cv::write ( fs, "type", static_cast<int>(PR_RECORD_TYPE::DEVICE));
    cv::write ( fs, "size", _size);
    cv::write ( fs, "threshold", _nElectrodeThreshold );
    fs.release();
    return VisionStatus::OK;
}

void DeviceRecord::setSize(const cv::Size2f &size) {
    _size = size;
}

const cv::Size2f& DeviceRecord::getSize() const {
    return _size;
}

void DeviceRecord::setElectrodeThreshold(Int16 nElectrodeThreshold) {
    _nElectrodeThreshold;
}

Int16 DeviceRecord::getElectrodeThreshold() const {
    return _nElectrodeThreshold;
}

/******************************************
* Chip Record *
******************************************/
VisionStatus ChipRecord::load(cv::FileStorage &fs, const String& strFilePath) {
    cv::FileNode fileNode = fs[_strKeySize];
    cv::read<float>(fileNode, _size, cv::Size2f(0, 0));

    fileNode = fs[_strKeyInspMode];
    int nInspMode = 0;
    cv::read(fileNode, nInspMode, 0);
    _enInspMode = static_cast<PR_INSP_CHIP_MODE> (nInspMode);

    fileNode = fs[_strKeyThreshold];
    cv::read(fileNode, _nThreshold, 0);

    _matTmpl = cv::imread(strFilePath + "/" + _strTmplFileName, cv::IMREAD_GRAYSCALE);
    return VisionStatus::OK;
}

VisionStatus ChipRecord::save(const String& strFilePath) {
    String strParamFilePath = strFilePath + "/" + Config::GetInstance()->getRecordParamFile();
    cv::FileStorage fs(strParamFilePath, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        return VisionStatus::OPEN_FILE_FAIL;

    cv::imwrite(strFilePath + "/" + _strTmplFileName, _matTmpl);

    cv::write(fs, _strKeyType, ToInt32(PR_RECORD_TYPE::CHIP));
    cv::write(fs, _strKeyInspMode, ToInt32(_enInspMode));
    cv::write(fs, _strKeySize, _size);
    cv::write(fs, _strKeyThreshold, _nThreshold);
    fs.release();
    return VisionStatus::OK;
}

void ChipRecord::setInspMode ( PR_INSP_CHIP_MODE enInspMode ) {
    _enInspMode = enInspMode;
}

PR_INSP_CHIP_MODE ChipRecord::getInspMode () const {
    return _enInspMode;
}

void ChipRecord::setSize(const cv::Size2f &size) {
    _size = size;
}

const cv::Size2f& ChipRecord::getSize() const {
    return _size;
}

void ChipRecord::setThreshold(Int16 nThreshold) {
    _nThreshold = nThreshold;
}

Int16 ChipRecord::getThreshold() const {
    return _nThreshold;
}

/******************************************
* Contour Record *
******************************************/
VisionStatus ContourRecord::load(cv::FileStorage &fs, const String& strFilePath)
{
    cv::FileNode fileNode;

    fileNode = fs[_strKeyThreshold];
    cv::read(fileNode, _nThreshold, 0 );

    _matTmpl = cv::imread(strFilePath + "/" + _strTmplFileName, cv::IMREAD_GRAYSCALE);
    if (_matTmpl.empty())
        return VisionStatus::INVALID_RECORD_FILE;

    _matContour = cv::imread(strFilePath + "/" + _strContourFileName, cv::IMREAD_GRAYSCALE);
    if (_matContour.empty())
        return VisionStatus::INVALID_RECORD_FILE;

    VectorOfVectorOfPoint vecContours;
    cv::findContours(_matContour, vecContours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // Filter contours
    for (auto contour : vecContours) {
        auto area = cv::contourArea(contour);
        if (area > 1000)
            _vecContours.push_back(contour);
    }
    return VisionStatus::OK;
}

VisionStatus ContourRecord::save(const String& strFilePath) {
    String strParamFilePath = strFilePath + "/" + Config::GetInstance()->getRecordParamFile();
    cv::FileStorage fs(strParamFilePath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return VisionStatus::OPEN_FILE_FAIL;

    cv::write(fs, _strKeyType, ToInt32(PR_RECORD_TYPE::CONTOUR));
    cv::write(fs, _strKeyThreshold, _nThreshold);
    cv::imwrite(strFilePath + "/" + _strTmplFileName, _matTmpl);
    cv::imwrite(strFilePath + "/" + _strContourFileName, _matContour);

    fs.release();
    return VisionStatus::OK;
}

void ContourRecord::setThreshold(Int16 nThreshold) {
    _nThreshold = nThreshold;
}

Int16 ContourRecord::getThreshold() const {
    return _nThreshold;
}

cv::Mat ContourRecord::getTmpl() const {
    return _matTmpl;
}

void ContourRecord::setTmpl( const cv::Mat &matTmpl ) {
    _matTmpl = matTmpl;
}

void ContourRecord::setContourMat(const cv::Mat &matContour) {
    _matContour = matContour;
}

void ContourRecord::setContour ( const VectorOfVectorOfPoint &vecContours ) {
    _vecContours = vecContours;
}

VectorOfVectorOfPoint ContourRecord::getContour() const {
    return _vecContours;
}

String ContourRecord::getTmplFileName() const {
    return _strTmplFileName;
}

String ContourRecord::getContourFileName() const {
    return _strContourFileName;
}

/******************************************
* Template Record *
******************************************/
VisionStatus TmplRecord::load(cv::FileStorage &fs, const String& strFilePath) {
    cv::FileNode fileNode;

    fileNode = fs[_strKeyAlgorithm];
    Int32 nAlgorithm = 0;
    cv::read(fileNode, nAlgorithm, 0 );
    _enAlgorithm = static_cast<PR_MATCH_TMPL_ALGORITHM> ( nAlgorithm );

    _matTmpl = cv::imread(strFilePath + "/" + _strTmplFileName, cv::IMREAD_GRAYSCALE);
    if (_matTmpl.empty())
        return VisionStatus::INVALID_RECORD_FILE;

    if (PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_EDGE == _enAlgorithm) {
        _matEdgeMask = cv::imread(strFilePath + "/" + _strEdgeMaskFileName, cv::IMREAD_GRAYSCALE);
        if (_matEdgeMask.empty())
            return VisionStatus::INVALID_RECORD_FILE;
    }

    String strMaskFileName = strFilePath + "/" + _strMaskFileName;
    if (FileUtils::Exists(strMaskFileName))
        _matMask = cv::imread(strMaskFileName);

    return VisionStatus::OK;
}

VisionStatus TmplRecord::save(const String& strFilePath) {
    String strParamFilePath = strFilePath + "/" + Config::GetInstance()->getRecordParamFile();
    cv::FileStorage fs(strParamFilePath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return VisionStatus::OPEN_FILE_FAIL;

    cv::write(fs, _strKeyType, ToInt32(PR_RECORD_TYPE::TEMPLATE));
    cv::write(fs, _strKeyAlgorithm, ToInt32(_enAlgorithm));
    cv::imwrite(strFilePath + "/" + _strTmplFileName, _matTmpl);
    if (PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_EDGE == _enAlgorithm)
        cv::imwrite(strFilePath + "/" + _strEdgeMaskFileName, _matEdgeMask);

    if (!_matMask.empty())
        cv::imwrite(strFilePath + "/" + _strMaskFileName, _matMask);
    
    fs.release();
    return VisionStatus::OK;
}

void TmplRecord::setAlgorithm(PR_MATCH_TMPL_ALGORITHM enAlgorithm) {
    _enAlgorithm = enAlgorithm;
}

PR_MATCH_TMPL_ALGORITHM TmplRecord::getAlgorithm() const {
    return _enAlgorithm;
}

void TmplRecord::setTmpl( const cv::Mat &matTmpl ) {
    _matTmpl = matTmpl;
}

cv::Mat TmplRecord::getTmpl() const {
    return _matTmpl;
}

void TmplRecord::setEdgeMask(const cv::Mat &matEdgeMask) {
    _matEdgeMask = matEdgeMask;
}

cv::Mat TmplRecord::getEdgeMask() const {
    return _matEdgeMask;
}

OcvRecord::OcvRecord(
    const cv::Mat      &matBigTmpl,
    const VectorOfRect &vecCharRects,
    PR_RECORD_TYPE      enType) :
    _matBigTmpl(matBigTmpl),
    _vecCharRects(vecCharRects),
    Record(enType) {
}

VisionStatus OcvRecord::load(cv::FileStorage &fs, const String& strFilePath) {
    _matBigTmpl = cv::imread(strFilePath + "/" + _strTmplFileName, cv::IMREAD_GRAYSCALE);
    if (_matBigTmpl.empty())
        return VisionStatus::INVALID_RECORD_FILE;

    cv::FileNode fileNode;
    fileNode = fs[_strKeyCharCount];
    Int32 nCharCount = 0;
    cv::read(fileNode, nCharCount, 0 );
    for (int i = 0; i < nCharCount; ++i) {
        std::string strKeyCharRect = _strKeyCharRect + std::to_string(i);
        fileNode = fs[strKeyCharRect];
        cv::Rect charRect;
        cv::read(fileNode, charRect, cv::Rect(0, 0, 0, 0));
        _vecCharRects.push_back(charRect);
    }
    return VisionStatus::OK;
}

VisionStatus OcvRecord::save(const String& strFilePath) {
    String strParamFilePath = strFilePath + "/" + Config::GetInstance()->getRecordParamFile();
    cv::FileStorage fs(strParamFilePath, cv::FileStorage::WRITE);
    if ( ! fs.isOpened() )
        return VisionStatus::OPEN_FILE_FAIL;

    cv::write(fs, _strKeyType, ToInt32(_enType));
    cv::write(fs, _strKeyCharCount, ToInt32(_vecCharRects.size()));
    for (size_t i = 0; i < _vecCharRects.size(); ++ i) {
        std::string strKeyCharRect = _strKeyCharRect + std::to_string(i);
        cv::write(fs, strKeyCharRect, _vecCharRects[i]);
    }
    cv::imwrite(strFilePath + "/" + _strTmplFileName, _matBigTmpl);
    
    fs.release();
    return VisionStatus::OK;
}

void OcvRecord::setBigTmpl(const cv::Mat &matBigTmpl) {
    _matBigTmpl = matBigTmpl;
}

cv::Mat OcvRecord::getBigTmpl() const {
    return _matBigTmpl;
}

VectorOfRect OcvRecord::getCharRects() const {
    return _vecCharRects;
}


}
}
