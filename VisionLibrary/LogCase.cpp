#include "LogCase.h"
#include <vector>
#include <sstream>
#include "opencv2/highgui.hpp"
#include "boost/format.hpp"
#include "boost/filesystem.hpp"
#include "StopWatch.h"
#include "SimpleIni.h"
#include "VisionAlgorithm.h"
#include "FileUtils.h"
#include "CalcUtils.h"
#include "JoinSplit.h"

namespace bfs = boost::filesystem;

namespace AOI
{
namespace Vision
{

//Define local functions
namespace
{
    StringVector split(const String &s, char delim)
    {
        StringVector elems;
        std::stringstream ss(s);
        String item;
        while (getline(ss, item, delim)) {
            elems.push_back(item);
        }
        return elems;
    }
}

LogCase::LogCase(const String &strPath, bool bReplay) :_strLogCasePath(strPath), _bReplay(bReplay) {
    if (bReplay)
        _strLogCasePath = strPath;
}

String LogCase::_formatCoordinate(const cv::Point2f &pt) {
    char chArray[200];
    _snprintf(chArray, sizeof (chArray), "%.2f, %.2f", pt.x, pt.y);
    return String(chArray);
}

cv::Point2f LogCase::_parseCoordinate(const String &strCoordinate) {
    cv::Point2f pt(0.f, 0.f);
    StringVector vecStrCoordinate = split(strCoordinate, ',');
    if (vecStrCoordinate.size() == 2) {
        pt.x = std::stof(vecStrCoordinate[0]);
        pt.y = std::stof(vecStrCoordinate[1]);
    }
    return pt;
}

String LogCase::_formatRect(const cv::Rect2f &pt) {
    char chArray[200];
    _snprintf(chArray, sizeof(chArray), "%.2f, %.2f, %.2f, %.2f", pt.tl().x, pt.tl().y, pt.br().x, pt.br().y);
    return String(chArray);
}

cv::Rect2f LogCase::_parseRect(const String &strCoordinate) {
    cv::Point2f pt1(0.f, 0.f), pt2(0.f, 0.f);
    StringVector vecStrCoordinate = split(strCoordinate, ',');
    if (vecStrCoordinate.size() == 4) {
        pt1.x = std::stof(vecStrCoordinate[0]);
        pt1.y = std::stof(vecStrCoordinate[1]);
        pt2.x = std::stof(vecStrCoordinate[2]);
        pt2.y = std::stof(vecStrCoordinate[3]);
    }
    return cv::Rect2f(pt1, pt2);
}

String  LogCase::_formatSize(const cv::Size &size) {
    char chArray[200];
    _snprintf(chArray, sizeof(chArray), "%d, %d", size.width, size.height);
    return String(chArray);
}

cv::Size LogCase::_parseSize(const String &strSize) {
    cv::Size size(0, 0);
    StringVector vecStrSize = split(strSize, ',');
    if (vecStrSize.size() == 2) {
        size.width = std::stoi(vecStrSize[0]);
        size.height = std::stoi(vecStrSize[1]);
    }
    return size;
}

String LogCase::_generateLogCaseName(const String &strFolderPrefix) {
    auto timeT = std::time(nullptr);
    auto stTM = std::localtime(&timeT);
    CStopWatch stopWatch;
    auto nMilliseconds = stopWatch.AbsNow() - timeT * 1000;
    String strFmt = "_%04s_%02s_%02s_%02s_%02s_%02s_%03s";
    String strTime = (boost::format(strFmt) % (stTM->tm_year + 1900) % (stTM->tm_mon + 1) % stTM->tm_mday % stTM->tm_hour % stTM->tm_min % stTM->tm_sec % nMilliseconds).str();
    String strLogCasePath = _strLogCasePath + strFolderPrefix + strTime + "\\";
    return strLogCasePath;
}

void LogCase::_zip() {
    String strPath = _strLogCasePath;
    if (! strPath.empty() && (strPath.back() == '\\' || strPath.back() == '/'))
        strPath = strPath.substr(0, strPath.length() - 1);
    joinDir(strPath.c_str(), _EXTENSION.c_str());
    FileUtils::RemoveAll(strPath);
}

String LogCase::unzip(const String &strFilePath) {
    char chFolderToken = '\\';
    size_t pos = strFilePath.find_last_of('\\');
    if (String::npos == pos) {
        chFolderToken = '/';
        pos = strFilePath.find_last_of(chFolderToken);
    }

    size_t posDot = strFilePath.find_last_of('.');
    String strTargetDir = strFilePath.substr(0, posDot);
    strTargetDir.push_back(chFolderToken);
    if (! FileUtils::Exists(strTargetDir))
        FileUtils::MakeDirectory(strTargetDir);
    splitFiles(strFilePath.c_str(), strTargetDir.c_str());
    FileUtils::Remove(strFilePath);
    return strTargetDir;
}

/*static*/ String LogCaseLrnObj::StaticGetFolderPrefix() {
    return "LrnObj";
}

VisionStatus LogCaseLrnObj::WriteCmd(const PR_LRN_OBJ_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), static_cast<long>(pstCmd->enAlgorithm));
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyLrnWindow.c_str(), _formatRect(pstCmd->rectLrn).c_str());
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    if (! pstCmd->matMask.empty())
        cv::imwrite(_strLogCasePath + _MASK_NAME, pstCmd->matMask);
    return VisionStatus::OK;
}

VisionStatus LogCaseLrnObj::WriteRpy(const PR_LRN_OBJ_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyCenterPos.c_str(), _formatCoordinate(pstRpy->ptCenter).c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyRecordId.c_str(), pstRpy->nRecordId);
    ini.SaveFile(cmdRpyFilePath.c_str());

    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseLrnObj::RunLogCase() {
    PR_LRN_OBJ_CMD stCmd;
    PR_LRN_OBJ_RPY stRpy;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.enAlgorithm = static_cast<PR_SRCH_OBJ_ALGORITHM> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), ToInt32(PR_SRCH_OBJ_ALGORITHM::SURF)));
    stCmd.rectLrn = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyLrnWindow.c_str(), ""));
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_GRAYSCALE);
    String strMaskPath = _strLogCasePath + _MASK_NAME;
    if (FileUtils::Exists(strMaskPath))
        stCmd.matMask = cv::imread(strMaskPath, cv::IMREAD_GRAYSCALE);

    enStatus = VisionAlgorithm::lrnObj(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseSrchObj::StaticGetFolderPrefix() {
    return "SrchObj";
}

VisionStatus LogCaseSrchObj::WriteCmd(const PR_SRCH_OBJ_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), static_cast<long>(pstCmd->enAlgorithm));
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _formatRect(pstCmd->rectSrchWindow).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyExpectedPos.c_str(), _formatCoordinate(pstCmd->ptExpectedPos).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyRecordId.c_str(), pstCmd->nRecordId);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseSrchObj::WriteRpy(const PR_SRCH_OBJ_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyObjPos.c_str(), _formatCoordinate(pstRpy->ptObjPos).c_str());
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyRotation.c_str(), pstRpy->fRotation);
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyOffset.c_str(), _formatSize(pstRpy->szOffset).c_str());
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyMatchScore.c_str(), pstRpy->fMatchScore);
    ini.SaveFile(cmdRpyFilePath.c_str());

    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseSrchObj::RunLogCase() {
    PR_SRCH_OBJ_CMD stCmd;
    PR_SRCH_OBJ_RPY stRpy;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.enAlgorithm = static_cast<PR_SRCH_OBJ_ALGORITHM> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), ToInt32(PR_SRCH_OBJ_ALGORITHM::SURF)));
    stCmd.rectSrchWindow = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), ""));
    stCmd.ptExpectedPos = _parseCoordinate(ini.GetValue(_CMD_SECTION.c_str(), _strKeyExpectedPos.c_str(), _DEFAULT_COORD.c_str()));
    stCmd.nRecordId = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyRecordId.c_str(), 0);

    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_GRAYSCALE);

    enStatus = VisionAlgorithm::srchObj(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseFitCircle::StaticGetFolderPrefix() {
    return "FitCircle";
}

VisionStatus LogCaseFitCircle::WriteCmd(const PR_FIT_CIRCLE_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMethod.c_str(), static_cast<long>(pstCmd->enMethod));
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyRmNoiseMethod.c_str(), static_cast<long>(pstCmd->enRmNoiseMethod));
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyPreprocessed.c_str(), pstCmd->bPreprocessed);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), pstCmd->fErrTol);
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyAutoThreshold.c_str(), pstCmd->bAutoThreshold);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), pstCmd->nThreshold);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), ToInt32(pstCmd->enAttribute));
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    if (! pstCmd->matMask.empty())
        cv::imwrite(_strLogCasePath + _MASK_NAME, pstCmd->matMask);
    return VisionStatus::OK;
}

VisionStatus LogCaseFitCircle::WriteRpy(const PR_FIT_CIRCLE_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyResultCtr.c_str(), _formatCoordinate(pstRpy->ptCircleCtr).c_str());
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyRadius.c_str(), pstRpy->fRadius);
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseFitCircle::RunLogCase() {
    PR_FIT_CIRCLE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);

    String strMaskPath = _strLogCasePath + _MASK_NAME;
    if (FileUtils::Exists(strMaskPath))
        stCmd.matMask = cv::imread(strMaskPath, cv::IMREAD_GRAYSCALE);

    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enMethod = static_cast<PR_FIT_METHOD> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMethod.c_str(), 0));
    stCmd.enRmNoiseMethod = static_cast<PR_RM_FIT_NOISE_METHOD> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyRmNoiseMethod.c_str(), 0));
    stCmd.bPreprocessed = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyPreprocessed.c_str(), true);
    stCmd.fErrTol = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), 0);
    stCmd.bAutoThreshold = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyAutoThreshold.c_str(), false);
    stCmd.nThreshold = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), 0);
    stCmd.enAttribute = static_cast<PR_OBJECT_ATTRIBUTE>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), 0));

    PR_FIT_CIRCLE_RPY stRpy;
    enStatus = VisionAlgorithm::fitCircle(&stCmd, &stRpy, true);

    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseFindCircle::StaticGetFolderPrefix() {
    return "FindCircle";
}

VisionStatus LogCaseFindCircle::WriteCmd(const PR_FIND_CIRCLE_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyInnerAttribute.c_str(), ToInt32(pstCmd->enInnerAttribute));
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyExpCircleCtr.c_str(), _formatCoordinate(pstCmd->ptExpectedCircleCtr).c_str());
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyFindCirclePair.c_str(), pstCmd->bFindCirclePair);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinSrchRadius.c_str(), pstCmd->fMinSrchRadius);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxSrchRadius.c_str(), pstCmd->fMaxSrchRadius);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyStartSrchAngle.c_str(), pstCmd->fStartSrchAngle);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyEndSrchAngle.c_str(), pstCmd->fEndSrchAngle);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyCaliperCount.c_str(), pstCmd->nCaliperCount);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyCaliperWidth.c_str(), pstCmd->fCaliperWidth);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyDiffFilterHalfW.c_str(), pstCmd->nDiffFilterHalfW);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyDiffFilterSigma.c_str(), pstCmd->fDiffFilterSigma);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyEdgeThreshold.c_str(), pstCmd->nEdgeThreshold);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeySelectEdge.c_str(), ToInt32(pstCmd->enSelectEdge));
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyRmStrayPtRatio.c_str(), pstCmd->fRmStrayPointRatio);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    if (! pstCmd->matMask.empty())
        cv::imwrite(_strLogCasePath + _MASK_NAME, pstCmd->matMask);
    return VisionStatus::OK;
}

VisionStatus LogCaseFindCircle::WriteRpy(const PR_FIND_CIRCLE_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyResultCtr.c_str(), _formatCoordinate(pstRpy->ptCircleCtr).c_str());
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyRadius.c_str(), pstRpy->fRadius);
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseFindCircle::RunLogCase() {
    PR_FIND_CIRCLE_CMD stCmd;
    PR_FIND_CIRCLE_RPY stRpy;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);
    String strMaskPath = _strLogCasePath + _MASK_NAME;
    if (FileUtils::Exists(strMaskPath))
        stCmd.matMask = cv::imread(strMaskPath, cv::IMREAD_GRAYSCALE);

    stCmd.enInnerAttribute = static_cast<PR_OBJECT_ATTRIBUTE> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyInnerAttribute.c_str(), 0));
    stCmd.ptExpectedCircleCtr = _parseCoordinate(ini.GetValue(_CMD_SECTION.c_str(), _strKeyExpCircleCtr.c_str(), _DEFAULT_COORD.c_str()));
    stCmd.bFindCirclePair = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyFindCirclePair.c_str(), false);
    stCmd.fMinSrchRadius = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinSrchRadius.c_str(), 0.));
    stCmd.fMaxSrchRadius = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxSrchRadius.c_str(), 0.));
    stCmd.fStartSrchAngle = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyStartSrchAngle.c_str(), 0.));
    stCmd.fEndSrchAngle = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyEndSrchAngle.c_str(), 0.));
    stCmd.nCaliperCount = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyCaliperCount.c_str(), 20);
    stCmd.fCaliperWidth = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyCaliperWidth.c_str(), 30.));
    stCmd.nDiffFilterHalfW = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyDiffFilterHalfW.c_str(), 2);
    stCmd.fDiffFilterSigma = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyDiffFilterSigma.c_str(), 1.));
    stCmd.nEdgeThreshold = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyEdgeThreshold.c_str(), 50);
    stCmd.enSelectEdge = static_cast<PR_CALIPER_SELECT_EDGE> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeySelectEdge.c_str(), 0));
    stCmd.fRmStrayPointRatio = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyRmStrayPtRatio.c_str(), 0.2));

    enStatus = VisionAlgorithm::findCircle(&stCmd, &stRpy, true);

    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseInspCircle::StaticGetFolderPrefix() {
    return "InspCircle";
}

VisionStatus LogCaseInspCircle::WriteCmd(const PR_INSP_CIRCLE_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    if (! pstCmd->matMask.empty())
        cv::imwrite(_strLogCasePath + _MASK_NAME, pstCmd->matMask);
    return VisionStatus::OK;
}

VisionStatus LogCaseInspCircle::WriteRpy(const PR_INSP_CIRCLE_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyResultCtr.c_str(), _formatCoordinate(pstRpy->ptCircleCtr).c_str());
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyDiameter.c_str(), pstRpy->fDiameter);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyRoundness.c_str(), pstRpy->fRoundness);
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseInspCircle::RunLogCase() {
    PR_INSP_CIRCLE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);

    String strMaskPath = _strLogCasePath + _MASK_NAME;
    if (FileUtils::Exists(strMaskPath))
        stCmd.matMask = cv::imread(strMaskPath, cv::IMREAD_GRAYSCALE);

    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));

    PR_INSP_CIRCLE_RPY stRpy;
    enStatus = VisionAlgorithm::inspCircle(&stCmd, &stRpy, true);

    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseFitLine::StaticGetFolderPrefix() {
    return "FitLine";
}

VisionStatus LogCaseFitLine::WriteCmd(const PR_FIT_LINE_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());

    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMethod.c_str(), ToInt32(pstCmd->enMethod));
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), pstCmd->fErrTol);
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyPreprocessed.c_str(), pstCmd->bPreprocessed);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), pstCmd->nThreshold);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), ToInt32(pstCmd->enAttribute));

    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    if (! pstCmd->matMask.empty())
        cv::imwrite(_strLogCasePath + _MASK_NAME, pstCmd->matMask);
    return VisionStatus::OK;
}

VisionStatus LogCaseFitLine::WriteRpy(const PR_FIT_LINE_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());

    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetBoolValue(_RPY_SECTION.c_str(), _strKeyReversedFit.c_str(), pstRpy->bReversedFit);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySlope.c_str(), pstRpy->fSlope);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept.c_str(), pstRpy->fIntercept);
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyPoint1.c_str(), _formatCoordinate(pstRpy->stLine.pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyPoint2.c_str(), _formatCoordinate(pstRpy->stLine.pt2).c_str());

    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseFitLine::RunLogCase() {
    PR_FIT_LINE_CMD stCmd;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    String strMaskPath = _strLogCasePath + _MASK_NAME;
    if (FileUtils::Exists(strMaskPath))
        stCmd.matMask = cv::imread(strMaskPath, cv::IMREAD_GRAYSCALE);

    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);

    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enMethod = static_cast<PR_FIT_METHOD>          (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMethod.c_str(), 0));
    stCmd.fErrTol = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), 0));
    stCmd.bPreprocessed = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyPreprocessed.c_str(), true);
    stCmd.nThreshold = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), 0);
    stCmd.enAttribute = static_cast<PR_OBJECT_ATTRIBUTE> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), 0));

    PR_FIT_LINE_RPY stRpy;
    VisionStatus enStatus = VisionAlgorithm::fitLine(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseFindLine::StaticGetFolderPrefix() {
    return "FindLine";
}

VisionStatus LogCaseFindLine::WriteCmd(const PR_FIND_LINE_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyRoiCenter.c_str(), _formatCoordinate(pstCmd->rectRotatedROI.center).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyRoiSize.c_str(), _formatSize(pstCmd->rectRotatedROI.size).c_str());
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyRoiAngle.c_str(), pstCmd->rectRotatedROI.angle);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), ToInt32(pstCmd->enAlgorithm));
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyFindPair.c_str(), pstCmd->bFindPair);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyDir.c_str(), ToInt32(pstCmd->enDetectDir));
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyCaliperCount.c_str(), pstCmd->nCaliperCount);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyCaliperWidth.c_str(), pstCmd->fCaliperWidth);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyDiffFilterHalfW.c_str(), pstCmd->nDiffFilterHalfW);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyDiffFilterSigma.c_str(), pstCmd->fDiffFilterSigma);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyEdgeThreshold.c_str(), pstCmd->nEdgeThreshold);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeySelectEdge.c_str(), ToInt32(pstCmd->enSelectEdge));
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyRmStrayPtRatio.c_str(), pstCmd->fRmStrayPointRatio);

    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyCheckLinearity.c_str(), pstCmd->bCheckLinearity);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyPointMaxOffset.c_str(), pstCmd->fPointMaxOffset);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinLinearity.c_str(), pstCmd->fMinLinearity);
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyCheckAngle.c_str(), pstCmd->bCheckAngle);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyExpectedAngle.c_str(), pstCmd->fExpectedAngle);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyAngleDiffTol.c_str(), pstCmd->fAngleDiffTolerance);
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    if (! pstCmd->matMask.empty())
        cv::imwrite(_strLogCasePath + _MASK_NAME, pstCmd->matMask);
    return VisionStatus::OK;
}

VisionStatus LogCaseFindLine::WriteRpy(const PR_FIND_LINE_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetBoolValue(_RPY_SECTION.c_str(), _strKeyReversedFit.c_str(), pstRpy->bReversedFit);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySlope.c_str(), pstRpy->fSlope);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept.c_str(), pstRpy->fIntercept);
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLine1Point1.c_str(), _formatCoordinate(pstRpy->stLine.pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLine1Point2.c_str(), _formatCoordinate(pstRpy->stLine.pt2).c_str());
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept2.c_str(), pstRpy->fIntercept2);
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLine2Point1.c_str(), _formatCoordinate(pstRpy->stLine2.pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLine2Point2.c_str(), _formatCoordinate(pstRpy->stLine2.pt2).c_str());
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyDistance.c_str(), pstRpy->fDistance);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyLinearity.c_str(), pstRpy->fLinearity);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyAngle.c_str(), pstRpy->fAngle);
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseFindLine::RunLogCase() {
    PR_FIND_LINE_CMD stCmd;
    VisionStatus enStatus;

    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);
    String strMaskPath = _strLogCasePath + _MASK_NAME;
    if (FileUtils::Exists(strMaskPath))
        stCmd.matMask = cv::imread(strMaskPath, cv::IMREAD_GRAYSCALE);

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.rectRotatedROI.center = _parseCoordinate(ini.GetValue(_CMD_SECTION.c_str(), _strKeyRoiCenter.c_str(), _DEFAULT_COORD.c_str()));
    stCmd.rectRotatedROI.size = _parseSize(ini.GetValue(_CMD_SECTION.c_str(), _strKeyRoiSize.c_str(), _DEFAULT_SIZE.c_str()));
    stCmd.rectRotatedROI.angle = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyRoiAngle.c_str(), 0.));
    stCmd.enAlgorithm = static_cast<PR_FIND_LINE_ALGORITHM>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), 0));
    stCmd.bFindPair = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyFindPair.c_str(), false);
    stCmd.enDetectDir = static_cast<PR_CALIPER_DIR>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyDir.c_str(), 0));
    stCmd.nCaliperCount = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyCaliperCount.c_str(), 20);
    stCmd.fCaliperWidth = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyCaliperWidth.c_str(), 30.));
    stCmd.nDiffFilterHalfW = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyDiffFilterHalfW.c_str(), 2);
    stCmd.fDiffFilterSigma = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyDiffFilterSigma.c_str(), 1.));
    stCmd.nEdgeThreshold = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyEdgeThreshold.c_str(), 50);
    stCmd.enSelectEdge = static_cast<PR_CALIPER_SELECT_EDGE> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeySelectEdge.c_str(), 0));
    stCmd.fRmStrayPointRatio = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyRmStrayPtRatio.c_str(), 0.2));

    stCmd.bCheckLinearity = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyCheckLinearity.c_str(), false);
    stCmd.fPointMaxOffset = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyPointMaxOffset.c_str(), 0.));
    stCmd.fMinLinearity = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinLinearity.c_str(), 100.f));
    stCmd.bCheckAngle = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyCheckAngle.c_str(), false);
    stCmd.fExpectedAngle = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyExpectedAngle.c_str(), 0.));
    stCmd.fAngleDiffTolerance = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyAngleDiffTol.c_str(), 0.));

    PR_FIND_LINE_RPY stRpy;
    enStatus = VisionAlgorithm::findLine(&stCmd, &stRpy, true);

    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseFitParallelLine::StaticGetFolderPrefix() {
    return "FitParallelLine";
}

VisionStatus LogCaseFitParallelLine::WriteCmd(const PR_FIT_PARALLEL_LINE_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow1.c_str(), _formatRect(pstCmd->rectArrROI[0]).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow2.c_str(), _formatRect(pstCmd->rectArrROI[1]).c_str());
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), pstCmd->fErrTol);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), pstCmd->nThreshold);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), ToInt32(pstCmd->enAttribute));
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseFitParallelLine::WriteRpy(const PR_FIT_PARALLEL_LINE_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetBoolValue(_RPY_SECTION.c_str(), _strKeyReversedFit.c_str(), pstRpy->bReversedFit);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySlope.c_str(), pstRpy->fSlope);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept1.c_str(), pstRpy->fIntercept1);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept2.c_str(), pstRpy->fIntercept2);
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineOnePoint1.c_str(), _formatCoordinate(pstRpy->stLine1.pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineOnePoint2.c_str(), _formatCoordinate(pstRpy->stLine1.pt2).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineTwoPoint1.c_str(), _formatCoordinate(pstRpy->stLine2.pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineTwoPoint2.c_str(), _formatCoordinate(pstRpy->stLine2.pt2).c_str());
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseFitParallelLine::RunLogCase() {
    PR_FIT_PARALLEL_LINE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectArrROI[0] = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow1.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.rectArrROI[1] = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow2.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.fErrTol = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), 0);
    stCmd.nThreshold = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), 0);
    stCmd.enAttribute = static_cast<PR_OBJECT_ATTRIBUTE>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), 0));

    PR_FIT_PARALLEL_LINE_RPY stRpy;
    enStatus = VisionAlgorithm::fitParallelLine(&stCmd, &stRpy, true);

    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseFitRect::StaticGetFolderPrefix() {
    return "FitRect";
}

VisionStatus LogCaseFitRect::WriteCmd(const PR_FIT_RECT_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow1.c_str(), _formatRect(pstCmd->rectArrROI[0]).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow2.c_str(), _formatRect(pstCmd->rectArrROI[1]).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow3.c_str(), _formatRect(pstCmd->rectArrROI[0]).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow4.c_str(), _formatRect(pstCmd->rectArrROI[1]).c_str());
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), pstCmd->fErrTol);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), pstCmd->nThreshold);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), ToInt32(pstCmd->enAttribute));
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseFitRect::WriteRpy(const PR_FIT_RECT_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetBoolValue(_RPY_SECTION.c_str(), _strKeyReversedFit1.c_str(), pstRpy->bLineOneReversedFit);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySlope1.c_str(), pstRpy->fSlope1);
    ini.SetBoolValue(_RPY_SECTION.c_str(), _strKeyReversedFit2.c_str(), pstRpy->bLineTwoReversedFit);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySlope2.c_str(), pstRpy->fSlope2);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept1.c_str(), pstRpy->fArrIntercept[0]);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept2.c_str(), pstRpy->fArrIntercept[1]);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept3.c_str(), pstRpy->fArrIntercept[2]);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept4.c_str(), pstRpy->fArrIntercept[3]);
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineOnePoint1.c_str(), _formatCoordinate(pstRpy->arrLines[0].pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineOnePoint2.c_str(), _formatCoordinate(pstRpy->arrLines[0].pt2).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineTwoPoint1.c_str(), _formatCoordinate(pstRpy->arrLines[1].pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineTwoPoint2.c_str(), _formatCoordinate(pstRpy->arrLines[1].pt2).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineThreePoint1.c_str(), _formatCoordinate(pstRpy->arrLines[2].pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineThreePoint2.c_str(), _formatCoordinate(pstRpy->arrLines[2].pt2).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineFourPoint1.c_str(), _formatCoordinate(pstRpy->arrLines[3].pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineFourPoint2.c_str(), _formatCoordinate(pstRpy->arrLines[3].pt2).c_str());
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseFitRect::RunLogCase() {
    PR_FIT_RECT_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectArrROI[0] = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow1.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.rectArrROI[1] = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow2.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.rectArrROI[2] = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow3.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.rectArrROI[3] = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow4.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.fErrTol = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), 0);
    stCmd.nThreshold = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), 0);
    stCmd.enAttribute = static_cast<PR_OBJECT_ATTRIBUTE>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), 0));

    PR_FIT_RECT_RPY stRpy;
    enStatus = VisionAlgorithm::fitRect(&stCmd, &stRpy, true);

    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseFindEdge::StaticGetFolderPrefix() {
    return "FindEdge";
}

VisionStatus LogCaseFindEdge::WriteCmd(const PR_FIND_EDGE_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyAutoThreshold.c_str(), pstCmd->bAutoThreshold);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), pstCmd->nThreshold);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyDirection.c_str(), ToInt32(pstCmd->enDirection));
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinLength.c_str(), pstCmd->fMinLength);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseFindEdge::WriteRpy(const PR_FIND_EDGE_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyEdgeCount.c_str(), ToInt32(pstRpy->nEdgeCount));
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseFindEdge::RunLogCase() {
    PR_FIND_EDGE_CMD stCmd;
    PR_FIND_EDGE_RPY stRpy;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.bAutoThreshold = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyAutoThreshold.c_str(), 0);
    stCmd.nThreshold = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), 0);
    stCmd.enDirection = static_cast<PR_EDGE_DIRECTION> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyDirection.c_str(), 0));
    stCmd.fMinLength = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinLength.c_str(), 0);

    enStatus = VisionAlgorithm::findEdge(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseSrchFiducial::StaticGetFolderPrefix() {
    return "SrchFiducial";
}

VisionStatus LogCaseSrchFiducial::WriteCmd(PR_SRCH_FIDUCIAL_MARK_CMD *pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _formatRect(pstCmd->rectSrchWindow).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyType.c_str(), ToInt32(pstCmd->enType));
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeySize.c_str(), pstCmd->fSize);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMargin.c_str(), pstCmd->fMargin);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseSrchFiducial::WriteRpy(PR_SRCH_FIDUCIAL_MARK_RPY *pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyPos.c_str(), _formatCoordinate(pstRpy->ptPos).c_str());

    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseSrchFiducial::RunLogCase() {
    PR_SRCH_FIDUCIAL_MARK_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectSrchWindow = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enType = (PR_FIDUCIAL_MARK_TYPE)ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyType.c_str(), 0);
    stCmd.fSize = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeySize.c_str(), 0);
    stCmd.fMargin = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMargin.c_str(), 0);

    PR_SRCH_FIDUCIAL_MARK_RPY stRpy;

    enStatus = VisionAlgorithm::srchFiducialMark(&stCmd, &stRpy, true);

    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseOcr::StaticGetFolderPrefix() {
    return "Ocr";
}

VisionStatus LogCaseOcr::WriteCmd(PR_OCR_CMD *pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _formatRect(pstCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyDirection.c_str(), ToInt32(pstCmd->enDirection));
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseOcr::WriteRpy(PR_OCR_RPY *pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyResultStr.c_str(), pstRpy->strResult.c_str());
    ini.SaveFile(cmdRpyFilePath.c_str());
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseOcr::RunLogCase() {
    PR_OCR_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enDirection = (PR_DIRECTION)ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyDirection.c_str(), 0);

    PR_OCR_RPY stRpy;
    enStatus = VisionAlgorithm::ocr(&stCmd, &stRpy, true);

    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseRemoveCC::StaticGetFolderPrefix() {
    return "RemoveCC";
}

VisionStatus LogCaseRemoveCC::WriteCmd(PR_REMOVE_CC_CMD *pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyConnectivity.c_str(), pstCmd->nConnectivity);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyCompareType.c_str(), ToInt32(pstCmd->enCompareType));
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyAreaThreshold.c_str(), pstCmd->fAreaThreshold);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseRemoveCC::WriteRpy(PR_REMOVE_CC_RPY *pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyTotalCC.c_str(), pstRpy->nTotalCC);
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyRemovedCC.c_str(), pstRpy->nRemovedCC);
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseRemoveCC::RunLogCase() {
    PR_REMOVE_CC_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.nConnectivity = (Int16)ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyConnectivity.c_str(), 8);
    stCmd.enCompareType = static_cast<PR_COMPARE_TYPE>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyCompareType.c_str(), 0));
    stCmd.fAreaThreshold = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyAreaThreshold.c_str(), 100);

    PR_REMOVE_CC_RPY stRpy;
    enStatus = VisionAlgorithm::removeCC(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseDetectEdge::StaticGetFolderPrefix() {
    return "DetectEdge";
}

VisionStatus LogCaseDetectEdge::WriteCmd(PR_DETECT_EDGE_CMD *pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold1.c_str(), pstCmd->nThreshold1);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold2.c_str(), pstCmd->nThreshold2);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyApertureSize.c_str(), pstCmd->nApertureSize);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseDetectEdge::WriteRpy(PR_DETECT_EDGE_RPY *pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseDetectEdge::RunLogCase() {
    PR_DETECT_EDGE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.nThreshold1 = (Int16)ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold1.c_str(), 100);
    stCmd.nThreshold2 = (Int16)ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold2.c_str(), 40);
    stCmd.nApertureSize = (Int16)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyApertureSize.c_str(), 3);

    PR_DETECT_EDGE_RPY stRpy;
    enStatus = VisionAlgorithm::detectEdge(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseAutoThreshold::StaticGetFolderPrefix() {
    return "AutoThreshold";
}

VisionStatus LogCaseAutoThreshold::WriteCmd(PR_AUTO_THRESHOLD_CMD *pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThresholdNum.c_str(), pstCmd->nThresholdNum);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseAutoThreshold::WriteRpy(PR_AUTO_THRESHOLD_RPY *pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyThreshold.c_str(), _formatVector<Int16>(pstRpy->vecThreshold).c_str());
    ini.SaveFile(cmdRpyFilePath.c_str());
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseAutoThreshold::RunLogCase() {
    PR_AUTO_THRESHOLD_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.nThresholdNum = (Int16)ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThresholdNum.c_str(), 1);

    PR_AUTO_THRESHOLD_RPY stRpy;
    enStatus = VisionAlgorithm::autoThreshold(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseFillHole::StaticGetFolderPrefix() {
    return "FillHole";
}

VisionStatus LogCaseFillHole::WriteCmd(PR_FILL_HOLE_CMD *pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), ToInt32(pstCmd->enAttribute));
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMethod.c_str(), ToInt32(pstCmd->enMethod));
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMorphShape.c_str(), ToInt32(pstCmd->enMorphShape));
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMorphKernelX.c_str(), pstCmd->szMorphKernel.width);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMorphKernelY.c_str(), pstCmd->szMorphKernel.height);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMorphIteration.c_str(), pstCmd->nMorphIteration);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseFillHole::WriteRpy(PR_FILL_HOLE_RPY *pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseFillHole::RunLogCase() {
    PR_FILL_HOLE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enAttribute = static_cast<PR_OBJECT_ATTRIBUTE>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), 0));
    stCmd.enMethod = static_cast<PR_FILL_HOLE_METHOD>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMethod.c_str(), 0));
    stCmd.enMorphShape = static_cast<cv::MorphShapes>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMorphShape.c_str(), 0));
    stCmd.szMorphKernel.width = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMorphKernelX.c_str(), 0);
    stCmd.szMorphKernel.height = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMorphKernelY.c_str(), 0);
    stCmd.nMorphIteration = (Int16)(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMorphIteration.c_str(), 3));

    PR_FILL_HOLE_RPY stRpy;
    enStatus = VisionAlgorithm::fillHole(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseLrnTmpl::StaticGetFolderPrefix() {
    return "LrnTmpl";
}

VisionStatus LogCaseLrnTmpl::WriteCmd(const PR_LRN_TEMPLATE_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), ToInt32(pstCmd->enAlgorithm));
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseLrnTmpl::WriteRpy(const PR_LRN_TEMPLATE_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyRecordId.c_str(), pstRpy->nRecordId);
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matTmpl.empty())
        cv::imwrite(_strLogCasePath + _strTmplFileName, pstRpy->matTmpl);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseLrnTmpl::RunLogCase() {
    PR_LRN_TEMPLATE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_GRAYSCALE);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enAlgorithm = static_cast<PR_MATCH_TMPL_ALGORITHM> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), 0));

    PR_LRN_TEMPLATE_RPY stRpy;
    enStatus = VisionAlgorithm::lrnTemplate(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseMatchTmpl::StaticGetFolderPrefix() {
    return "MatchTmpl";
}

VisionStatus LogCaseMatchTmpl::WriteCmd(const PR_MATCH_TEMPLATE_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _formatRect(pstCmd->rectSrchWindow).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMotion.c_str(), ToInt32(pstCmd->enMotion));
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyRecordId.c_str(), pstCmd->nRecordId);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), ToInt32(pstCmd->enAlgorithm));
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseMatchTmpl::WriteRpy(const PR_MATCH_TEMPLATE_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyObjPos.c_str(), _formatCoordinate(pstRpy->ptObjPos).c_str());
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyRotation.c_str(), pstRpy->fRotation);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyMatchScore.c_str(), pstRpy->fMatchScore);
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseMatchTmpl::RunLogCase() {
    PR_MATCH_TEMPLATE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enMotion = static_cast<PR_OBJECT_MOTION>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMotion.c_str(), 0));
    stCmd.nRecordId = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyRecordId.c_str(), 0);
    stCmd.enAlgorithm = static_cast<PR_MATCH_TMPL_ALGORITHM> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), 0));

    PR_MATCH_TEMPLATE_RPY stRpy;
    enStatus = VisionAlgorithm::matchTemplate(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCasePickColor::StaticGetFolderPrefix() {
    return "PickColor";
}

VisionStatus LogCasePickColor::WriteCmd(const PR_PICK_COLOR_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyPickPoint.c_str(), _formatCoordinate(pstCmd->ptPick).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyColorDiff.c_str(), pstCmd->nColorDiff);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyGrayDiff.c_str(), pstCmd->nGrayDiff);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCasePickColor::WriteRpy(const PR_PICK_COLOR_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCasePickColor::RunLogCase() {
    PR_PICK_COLOR_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.ptPick = _parseCoordinate(ini.GetValue(_CMD_SECTION.c_str(), _strKeyPickPoint.c_str(), _DEFAULT_COORD.c_str()));
    stCmd.nColorDiff = static_cast<Int16> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyColorDiff.c_str(), 10));
    stCmd.nGrayDiff = static_cast<Int16> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyGrayDiff.c_str(), 10));

    PR_PICK_COLOR_RPY stRpy;
    enStatus = VisionAlgorithm::pickColor(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseCalibrateCamera::StaticGetFolderPrefix() {
    return "CalibrateCamera";
}

VisionStatus LogCaseCalibrateCamera::WriteCmd(const PR_CALIBRATE_CAMERA_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyBoardPattern.c_str(), _formatSize(pstCmd->szBoardPattern).c_str());
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyObjectSize.c_str(), pstCmd->fPatternDist);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseCalibrateCamera::WriteRpy(const PR_CALIBRATE_CAMERA_RPY * const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    if (VisionStatus::OK == pstRpy->enStatus) {
        ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyResolutionX.c_str(), pstRpy->dResolutionX);
        ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyResolutionY.c_str(), pstRpy->dResolutionY);
    }
    ini.SaveFile(cmdRpyFilePath.c_str());

    if (! pstRpy->matCornerPointsImg.empty())
        cv::imwrite(_strLogCasePath + _strCornerPointsImage, pstRpy->matCornerPointsImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseCalibrateCamera::RunLogCase() {
    PR_CALIBRATE_CAMERA_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);
    stCmd.szBoardPattern = _parseSize(ini.GetValue(_CMD_SECTION.c_str(), _strKeyBoardPattern.c_str(), _DEFAULT_SIZE.c_str()));
    stCmd.fPatternDist = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyObjectSize.c_str(), 0);

    PR_CALIBRATE_CAMERA_RPY stRpy;
    enStatus = VisionAlgorithm::calibrateCamera(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseRestoreImg::StaticGetFolderPrefix() {
    return "RestoreImg";
}

VisionStatus LogCaseRestoreImg::WriteCmd(const PR_RESTORE_IMG_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;

    cv::FileStorage fs(_strLogCasePath + _strYmlFile, cv::FileStorage::WRITE);
    cv::write(fs, _strKeyMapX, pstCmd->vecMatRestoreImage[0]);
    cv::write(fs, _strKeyMapY, pstCmd->vecMatRestoreImage[1]);
    fs.release();

    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseRestoreImg::WriteRpy(PR_RESTORE_IMG_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseRestoreImg::RunLogCase() {
    PR_RESTORE_IMG_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);

    cv::FileStorage fs(_strLogCasePath + _strYmlFile, cv::FileStorage::READ);
    cv::Mat matMapX, matMapY;
    cv::FileNode fileNodeMapX = fs[_strKeyMapX];
    cv::FileNode fileNodeMapY = fs[_strKeyMapY];
    cv::read(fileNodeMapX, matMapX);
    cv::read(fileNodeMapY, matMapY);
    fs.release();

    stCmd.vecMatRestoreImage.push_back(matMapX);
    stCmd.vecMatRestoreImage.push_back(matMapY);

    PR_RESTORE_IMG_RPY stRpy;
    enStatus = VisionAlgorithm::restoreImage(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);

    if (VisionStatus::OK == enStatus) {
        cv::Mat matInputFloat, matResultFloat;
        stCmd.matInputImg.convertTo(matInputFloat, CV_32FC1);
        stRpy.matResultImg.convertTo(matResultFloat, CV_32FC1);

        cv::Mat matDiff = cv::Scalar::all(255.f) - (matInputFloat - matResultFloat);
        cv::imwrite(_strLogCasePath + _strDiffImg, matDiff);
    }
    return enStatus;
}

/*static*/ String LogCaseAutoLocateLead::StaticGetFolderPrefix() {
    return "AutoLocateLead";
}

VisionStatus LogCaseAutoLocateLead::WriteCmd(const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _formatRect(pstCmd->rectSrchWindow).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyChipWindow.c_str(), _formatRect(pstCmd->rectChipBody).c_str());
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseAutoLocateLead::WriteRpy(PR_AUTO_LOCATE_LEAD_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    for (size_t i = 0; i < pstRpy->vecLeadLocation.size(); ++ i) {
        String strKey = _strLeadLocation + std::to_string(i);
        auto rect = pstRpy->vecLeadLocation[i];
        ini.SetValue(_RPY_SECTION.c_str(), strKey.c_str(), _formatRect(rect).c_str());
    }
    ini.SaveFile(cmdRpyFilePath.c_str());

    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseAutoLocateLead::RunLogCase() {
    PR_AUTO_LOCATE_LEAD_CMD stCmd;
    PR_AUTO_LOCATE_LEAD_RPY stRpy;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);

    stCmd.rectSrchWindow = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.rectChipBody = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyChipWindow.c_str(), _DEFAULT_RECT.c_str()));

    VisionStatus enStatus = VisionAlgorithm::autoLocateLead(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseInspBridge::StaticGetFolderPrefix() {
    return "InspBridge";
}

VisionStatus LogCaseInspBridge::WriteCmd(const PR_INSP_BRIDGE_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());

    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyInspMode.c_str(), ToInt32(pstCmd->enInspMode));
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());

    if (PR_INSP_BRIDGE_MODE::OUTER == pstCmd->enInspMode) {
        String strDirection, strMaxLength;
        for (const auto enDirection : pstCmd->vecOuterInspDirection)
            strDirection += std::to_string(ToInt32(enDirection)) + ", ";
        strDirection.resize(strDirection.size() - 2);
        ini.SetValue(_CMD_SECTION.c_str(), _strKeyDirection.c_str(), strDirection.c_str());
        ini.SetValue(_CMD_SECTION.c_str(), _strKeyOuterSrchWindow.c_str(), _formatRect(pstCmd->rectOuterSrchWindow).c_str());
    }
    else {
        ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxLengthX.c_str(), pstCmd->stInnerInspCriteria.fMaxLengthX);
        ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxLengthY.c_str(), pstCmd->stInnerInspCriteria.fMaxLengthY);
    }

    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseInspBridge::WriteRpy(PR_INSP_BRIDGE_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    int nInspResultIndex = 0;

    for (size_t nBridgeWindowIndex = 0; nBridgeWindowIndex < pstRpy->vecBridgeWindow.size(); ++ nBridgeWindowIndex) {
        String strKeyBridgeWindow = _strKeyBridgeWindow + "_" + std::to_string(nBridgeWindowIndex);
        ini.SetValue(_RPY_SECTION.c_str(), strKeyBridgeWindow.c_str(), _formatRect(pstRpy->vecBridgeWindow[nBridgeWindowIndex]).c_str());
    }

    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseInspBridge::RunLogCase() {
    PR_INSP_BRIDGE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);
    stCmd.enInspMode = static_cast<PR_INSP_BRIDGE_MODE>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyInspMode.c_str(), 0));
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));

    if (PR_INSP_BRIDGE_MODE::OUTER == stCmd.enInspMode) {
        stCmd.rectOuterSrchWindow = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyOuterSrchWindow.c_str(), _DEFAULT_RECT.c_str()));
        String strDirection = ini.GetValue(_CMD_SECTION.c_str(), _strKeyDirection.c_str(), "");
        StringVector vecStrDirection = split(strDirection, ',');
        for (size_t index = 0; index < vecStrDirection.size(); ++ index) {
            PR_INSP_BRIDGE_DIRECTION enDirection = static_cast<PR_INSP_BRIDGE_DIRECTION> (std::atoi(vecStrDirection[index].c_str()));
            stCmd.vecOuterInspDirection.push_back(enDirection);
        }
    }
    else {
        stCmd.stInnerInspCriteria.fMaxLengthX = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxLengthX.c_str(), 0.f));
        stCmd.stInnerInspCriteria.fMaxLengthY = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxLengthY.c_str(), 0.f));
    }

    PR_INSP_BRIDGE_RPY stRpy;
    enStatus = VisionAlgorithm::inspBridge(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseInspChip::StaticGetFolderPrefix() {
    return "InspChip";
}

VisionStatus LogCaseInspChip::WriteCmd(const PR_INSP_CHIP_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _formatRect(pstCmd->rectSrchWindow).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyInspMode.c_str(), ToInt32(pstCmd->enInspMode));
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyRecordId.c_str(), pstCmd->nRecordId);
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseInspChip::WriteRpy(const PR_INSP_CHIP_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyCenter.c_str(), _formatCoordinate(pstRpy->rotatedRectResult.center).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeySize.c_str(), _formatCoordinate(pstRpy->rotatedRectResult.size).c_str());
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyAngle.c_str(), pstRpy->rotatedRectResult.angle);
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseInspChip::RunLogCase() {
    PR_INSP_CHIP_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);
    stCmd.rectSrchWindow = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enInspMode = static_cast<PR_INSP_CHIP_MODE> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyInspMode.c_str(), 0));
    stCmd.nRecordId = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyRecordId.c_str(), -1);

    PR_INSP_CHIP_RPY stRpy;
    enStatus = VisionAlgorithm::inspChip(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseLrnContour::StaticGetFolderPrefix() {
    return "LrnContour";
}

VisionStatus LogCaseLrnContour::WriteCmd(const PR_LRN_CONTOUR_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyAutoThreshold.c_str(), pstCmd->bAutoThreshold);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), pstCmd->nThreshold);
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseLrnContour::WriteRpy(const PR_LRN_CONTOUR_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyThreshold.c_str(), pstRpy->nThreshold);
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyRecordId.c_str(), pstRpy->nRecordId);
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseLrnContour::RunLogCase() {
    PR_LRN_CONTOUR_CMD stCmd;
    PR_LRN_CONTOUR_RPY stRpy;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.bAutoThreshold = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyAutoThreshold.c_str(), true);
    stCmd.nThreshold = ToInt32(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), 0));

    VisionStatus enStatus;
    enStatus = VisionAlgorithm::lrnContour(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseInspContour::StaticGetFolderPrefix() {
    return "InspContour";
}

VisionStatus LogCaseInspContour::WriteCmd(const PR_INSP_CONTOUR_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyRecordId.c_str(), pstCmd->nRecordId);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyDefectThreshold.c_str(), pstCmd->nDefectThreshold);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinDefectArea.c_str(), pstCmd->fMinDefectArea);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyInnerLengthTol.c_str(), pstCmd->fDefectInnerLengthTol);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyOuterLengthTol.c_str(), pstCmd->fDefectOuterLengthTol);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyInnerMaskDepth.c_str(), pstCmd->fInnerMaskDepth);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyOuterMaskDepth.c_str(), pstCmd->fOuterMaskDepth);
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    if (! pstCmd->matMask.empty())
        cv::imwrite(_strLogCasePath + _MASK_NAME, pstCmd->matMask);
    return VisionStatus::OK;
}

VisionStatus LogCaseInspContour::WriteRpy(const PR_INSP_CONTOUR_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseInspContour::RunLogCase() {
    PR_INSP_CONTOUR_CMD stCmd;
    PR_INSP_CONTOUR_RPY stRpy;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);
    String strMaskPath = _strLogCasePath + _MASK_NAME;
    if (FileUtils::Exists(strMaskPath))
        stCmd.matMask = cv::imread(strMaskPath, cv::IMREAD_GRAYSCALE);

    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.nRecordId = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyRecordId.c_str(), 0);
    stCmd.nDefectThreshold = ToInt32(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyDefectThreshold.c_str(), 0));
    stCmd.fMinDefectArea = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinDefectArea.c_str(), 0));
    stCmd.fDefectInnerLengthTol = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyInnerLengthTol.c_str(), 0));
    stCmd.fDefectOuterLengthTol = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyOuterLengthTol.c_str(), 0));
    stCmd.fInnerMaskDepth = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyInnerMaskDepth.c_str(), 0));
    stCmd.fOuterMaskDepth = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyOuterMaskDepth.c_str(), 0));

    VisionStatus enStatus = VisionStatus::OK;
    enStatus = VisionAlgorithm::inspContour(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseInspHole::StaticGetFolderPrefix() {
    return "InspHole";
}

VisionStatus LogCaseInspHole::WriteCmd(const PR_INSP_HOLE_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pstCmd->rectROI).c_str());
    //ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyRecordId.c_str(), pstCmd->nRecordId );
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeySegmentMethod.c_str(), ToInt32(pstCmd->enSegmentMethod));
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyInspMode.c_str(), ToInt32(pstCmd->enInspMode));
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyGrayRangeStart.c_str(), pstCmd->stGrayScaleRange.nStart);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyGrayRangeEnd.c_str(), pstCmd->stGrayScaleRange.nEnd);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyBlueRangeStart.c_str(), pstCmd->stColorRange.nStartB);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyBlueRangeEnd.c_str(), pstCmd->stColorRange.nEndB);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyGreenRangeStart.c_str(), pstCmd->stColorRange.nStartG);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyGreenRangeEnd.c_str(), pstCmd->stColorRange.nEndG);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyRedRangeStart.c_str(), pstCmd->stColorRange.nStartR);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyRedRangeEnd.c_str(), pstCmd->stColorRange.nEndR);

    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxRatio.c_str(), pstCmd->stRatioModeCriteria.fMaxRatio);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinRatio.c_str(), pstCmd->stRatioModeCriteria.fMinRatio);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxArea.c_str(), pstCmd->stBlobModeCriteria.fMaxArea);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinArea.c_str(), pstCmd->stBlobModeCriteria.fMinArea);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMaxBlobCount.c_str(), pstCmd->stBlobModeCriteria.nMaxBlobCount);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMinBlobCount.c_str(), pstCmd->stBlobModeCriteria.nMinBlobCount);
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyAdvBlobCriteria.c_str(), pstCmd->stBlobModeCriteria.bEnableAdvancedCriteria);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxLWRatio.c_str(), pstCmd->stBlobModeCriteria.stAdvancedCriteria.fMaxLengthWidthRatio);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinLWRatio.c_str(), pstCmd->stBlobModeCriteria.stAdvancedCriteria.fMinLengthWidthRatio);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxCircularity.c_str(), pstCmd->stBlobModeCriteria.stAdvancedCriteria.fMaxCircularity);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinCircularity.c_str(), pstCmd->stBlobModeCriteria.stAdvancedCriteria.fMinCircularity);
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    if (! pstCmd->matMask.empty())
        cv::imwrite(_strLogCasePath + _MASK_NAME, pstCmd->matMask);
    return VisionStatus::OK;
}

VisionStatus LogCaseInspHole::WriteRpy(const PR_INSP_HOLE_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyRatio.c_str(), pstRpy->stRatioModeResult.fRatio);
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyBlobCount.c_str(), ToInt32(pstRpy->stBlobModeResult.vecBlobs.size()));
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseInspHole::RunLogCase() {
    PR_INSP_HOLE_CMD stCmd;
    PR_INSP_HOLE_RPY stRpy;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);
    String strMaskPath = _strLogCasePath + _MASK_NAME;
    if (FileUtils::Exists(strMaskPath))
        stCmd.matMask = cv::imread(strMaskPath, cv::IMREAD_GRAYSCALE);

    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    //stCmd.nRecordId = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyRecordId.c_str(), 0 );
    stCmd.enSegmentMethod = static_cast<PR_IMG_SEGMENT_METHOD>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeySegmentMethod.c_str(), 0));
    stCmd.enInspMode = static_cast<PR_INSP_HOLE_MODE>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyInspMode.c_str(), 0));
    stCmd.stGrayScaleRange.nStart = ToInt16(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyGrayRangeStart.c_str(), 0));
    stCmd.stGrayScaleRange.nEnd = ToInt16(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyGrayRangeEnd.c_str(), 0));
    stCmd.stColorRange.nStartB = ToInt16(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyBlueRangeStart.c_str(), 0));
    stCmd.stColorRange.nEndB = ToInt16(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyBlueRangeEnd.c_str(), 0));
    stCmd.stColorRange.nStartG = ToInt16(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyGreenRangeStart.c_str(), 0));
    stCmd.stColorRange.nEndG = ToInt16(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyGreenRangeEnd.c_str(), 0));
    stCmd.stColorRange.nStartR = ToInt16(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyRedRangeStart.c_str(), 0));
    stCmd.stColorRange.nEndR = ToInt16(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyRedRangeEnd.c_str(), 0));

    stCmd.stRatioModeCriteria.fMaxRatio = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxRatio.c_str(), 1.));
    stCmd.stRatioModeCriteria.fMinRatio = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinRatio.c_str(), 0.));
    stCmd.stBlobModeCriteria.fMaxArea = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxArea.c_str(), 100000.));
    stCmd.stBlobModeCriteria.fMinArea = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinArea.c_str(), 200.));
    stCmd.stBlobModeCriteria.nMaxBlobCount = ToInt16(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMaxBlobCount.c_str(), 10));
    stCmd.stBlobModeCriteria.nMinBlobCount = ToInt16(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMinBlobCount.c_str(), 0));
    stCmd.stBlobModeCriteria.bEnableAdvancedCriteria = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyAdvBlobCriteria.c_str(), false);
    stCmd.stBlobModeCriteria.stAdvancedCriteria.fMaxLengthWidthRatio = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxLWRatio.c_str(), 1.));
    stCmd.stBlobModeCriteria.stAdvancedCriteria.fMinLengthWidthRatio = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinLWRatio.c_str(), 0.));
    stCmd.stBlobModeCriteria.stAdvancedCriteria.fMaxCircularity = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMaxCircularity.c_str(), 1.));
    stCmd.stBlobModeCriteria.stAdvancedCriteria.fMinCircularity = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinCircularity.c_str(), 0.1));

    VisionStatus enStatus = VisionStatus::OK;
    enStatus = VisionAlgorithm::inspHole(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseInspLead::StaticGetFolderPrefix() {
    return "InspLead";
}

VisionStatus LogCaseInspLead::WriteCmd(const PR_INSP_LEAD_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyChipCenter.c_str(), _formatCoordinate(pstCmd->rectChipWindow.center).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyChipSize.c_str(), _formatSize(pstCmd->rectChipWindow.size).c_str());
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyChipAngle.c_str(), pstCmd->rectChipWindow.angle);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyLeadCount.c_str(), ToInt32(pstCmd->vecLeads.size()));
    for (size_t i = 0; i < pstCmd->vecLeads.size(); ++ i) {
        auto stLeadInput = pstCmd->vecLeads[i];
        String strIndex = String("_") + std::to_string(i);
        String strKeyLeadSrchWinCtr = _strKeyLeadSrchWinCtr + strIndex;
        String strKeyLeadSrchWinSize = _strKeyLeadSrchWinSize + strIndex;
        String strKeyLeadSrchWinAngle = _strKeyLeadSrchWinAngle + strIndex;
        String strKeyLeadExpWinCtr = _strKeyLeadExpWinCtr + strIndex;
        String strKeyLeadExpWinSize = _strKeyLeadExpWinSize + strIndex;
        String strKeyLeadExpWinAngle = _strKeyLeadExpWinAngle + strIndex;

        ini.SetValue(_CMD_SECTION.c_str(), strKeyLeadSrchWinCtr.c_str(), _formatCoordinate(stLeadInput.rectSrchWindow.center).c_str());
        ini.SetValue(_CMD_SECTION.c_str(), strKeyLeadSrchWinSize.c_str(), _formatSize(stLeadInput.rectSrchWindow.size).c_str());
        ini.SetDoubleValue(_CMD_SECTION.c_str(), strKeyLeadSrchWinAngle.c_str(), stLeadInput.rectSrchWindow.angle);
        ini.SetValue(_CMD_SECTION.c_str(), strKeyLeadExpWinCtr.c_str(), _formatCoordinate(stLeadInput.rectExpectedWindow.center).c_str());
        ini.SetValue(_CMD_SECTION.c_str(), strKeyLeadExpWinSize.c_str(), _formatSize(stLeadInput.rectExpectedWindow.size).c_str());
        ini.SetDoubleValue(_CMD_SECTION.c_str(), strKeyLeadExpWinAngle.c_str(), stLeadInput.rectExpectedWindow.angle);
    }

    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyLeadStartWidthRatio.c_str(), pstCmd->fLeadStartWidthRatio);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyLeadStartConLen.c_str(), pstCmd->nLeadStartConsecutiveLength);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyLeadEndWidthRatio.c_str(), pstCmd->fLeadEndWidthRatio);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyLeadEndConLen.c_str(), pstCmd->nLeadEndConsecutiveLength);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyFindLeadEndMethod.c_str(), ToInt32(pstCmd->enFindLeadEndMethod));
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);
    return VisionStatus::OK;
}

VisionStatus LogCaseInspLead::WriteRpy(const PR_INSP_LEAD_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    for (size_t i = 0; i < pstRpy->vecLeadResult.size(); ++ i) {
        auto stLeadResult = pstRpy->vecLeadResult[i];
        String strIndex = String("_") + std::to_string(i);
        String strFound = _strFound + strIndex;
        String strKeyLeadWinCtr = _strKeyLeadWinCtr + strIndex;
        String strKeyLeadWinSize = _strKeyLeadWinSize + strIndex;
        String strKeyLeadWinAngle = _strKeyLeadWinAngle + strIndex;
        ini.SetBoolValue(_RPY_SECTION.c_str(), strFound.c_str(), stLeadResult.bFound);
        ini.SetValue(_RPY_SECTION.c_str(), strKeyLeadWinCtr.c_str(), _formatCoordinate(stLeadResult.rectLead.center).c_str());
        ini.SetValue(_RPY_SECTION.c_str(), strKeyLeadWinSize.c_str(), _formatSize(stLeadResult.rectLead.size).c_str());
        ini.SetDoubleValue(_RPY_SECTION.c_str(), strKeyLeadWinAngle.c_str(), stLeadResult.rectLead.angle);
    }
    ini.SaveFile(cmdRpyFilePath.c_str());
    if (! pstRpy->matResultImg.empty())
        cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pstRpy->matResultImg);
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseInspLead::RunLogCase() {
    PR_INSP_LEAD_CMD stCmd;
    PR_INSP_LEAD_RPY stRpy;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);

    stCmd.rectChipWindow.center = _parseCoordinate(ini.GetValue(_CMD_SECTION.c_str(), _strKeyChipCenter.c_str(), _DEFAULT_COORD.c_str()));
    stCmd.rectChipWindow.size = _parseSize(ini.GetValue(_CMD_SECTION.c_str(), _strKeyChipSize.c_str(), _DEFAULT_SIZE.c_str()));
    stCmd.rectChipWindow.angle = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyChipAngle.c_str(), 0.));
    int nLeadCount = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyLeadCount.c_str(), 0);
    for (int i = 0; i < nLeadCount; ++ i) {
        String strIndex = String("_") + std::to_string(i);
        String strKeyLeadSrchWinCtr = _strKeyLeadSrchWinCtr + strIndex;
        String strKeyLeadSrchWinSize = _strKeyLeadSrchWinSize + strIndex;
        String strKeyLeadSrchWinAngle = _strKeyLeadSrchWinAngle + strIndex;
        String strKeyLeadExpWinCtr = _strKeyLeadExpWinCtr + strIndex;
        String strKeyLeadExpWinSize = _strKeyLeadExpWinSize + strIndex;
        String strKeyLeadExpWinAngle = _strKeyLeadExpWinAngle + strIndex;
        PR_INSP_LEAD_CMD::LEAD_INPUT_INFO stLeadInput;
        stLeadInput.rectSrchWindow.center = _parseCoordinate(ini.GetValue(_CMD_SECTION.c_str(), strKeyLeadSrchWinCtr.c_str(), _DEFAULT_COORD.c_str()));
        stLeadInput.rectSrchWindow.size = _parseSize(ini.GetValue(_CMD_SECTION.c_str(), strKeyLeadSrchWinSize.c_str(), _DEFAULT_SIZE.c_str()));
        stLeadInput.rectSrchWindow.angle = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), strKeyLeadSrchWinAngle.c_str(), 0.));
        stLeadInput.rectExpectedWindow.center = _parseCoordinate(ini.GetValue(_CMD_SECTION.c_str(), strKeyLeadExpWinCtr.c_str(), _DEFAULT_COORD.c_str()));
        stLeadInput.rectExpectedWindow.size = _parseSize(ini.GetValue(_CMD_SECTION.c_str(), strKeyLeadExpWinSize.c_str(), _DEFAULT_SIZE.c_str()));
        stLeadInput.rectExpectedWindow.angle = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), strKeyLeadExpWinAngle.c_str(), 0.));
        stCmd.vecLeads.push_back(stLeadInput);
    }
    stCmd.fLeadStartWidthRatio = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyLeadStartWidthRatio.c_str(), 0.5));
    stCmd.nLeadStartConsecutiveLength = ToInt16(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyLeadStartConLen.c_str(), 2));
    stCmd.fLeadEndWidthRatio = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyLeadEndWidthRatio.c_str(), 0.5));
    stCmd.nLeadEndConsecutiveLength = ToInt16(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyLeadEndConLen.c_str(), 2));
    stCmd.enFindLeadEndMethod = static_cast<PR_FIND_LEAD_END_METHOD> (ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyFindLeadEndMethod.c_str(), 0));

    VisionStatus enStatus = VisionStatus::OK;
    enStatus = VisionAlgorithm::inspLead(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseCalib3DBase::StaticGetFolderPrefix() {
    return "Calib3DBase";
}

VisionStatus LogCaseCalib3DBase::WriteCmd(const PR_CALIB_3D_BASE_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyEnableGF.c_str(), pstCmd->bEnableGaussianFilter);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyRemoveHarmonicWaveK.c_str(), pstCmd->fRemoveHarmonicWaveK);
    ini.SaveFile(cmdRpyFilePath.c_str());

    int index = 0;
    for (const auto &matImg : pstCmd->vecInputImgs) {
        String strFileName = _strLogCasePath + _IMAGE_FILE_PREFIX + std::to_string(index) + _IMAGE_FILE_EXT;
        cv::imwrite(strFileName, matImg);
        ++ index;
    }
    return VisionStatus::OK;
}

VisionStatus LogCaseCalib3DBase::WriteRpy(const PR_CALIB_3D_BASE_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyReverseSeq.c_str(), pstRpy->bReverseSeq);

    cv::FileStorage fs(_strLogCasePath + _strResultYmlFileName, cv::FileStorage::WRITE);
    if (! fs.isOpened())
        return VisionStatus::OPEN_FILE_FAIL;

    cv::write(fs, _strKeyThkToThinK, pstRpy->matThickToThinK);
    cv::write(fs, _strKeyThkToThinnestK, pstRpy->matThickToThinnestK);
    fs.release();

    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseCalib3DBase::RunLogCase() {
    PR_CALIB_3D_BASE_CMD stCmd;
    PR_CALIB_3D_BASE_RPY stRpy;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());

    stCmd.bEnableGaussianFilter = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyEnableGF.c_str(), false);
    stCmd.fRemoveHarmonicWaveK = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyRemoveHarmonicWaveK.c_str(), 0.f));

    int index = 0;
    while (true) {
        String strFileName = _strLogCasePath + _IMAGE_FILE_PREFIX + std::to_string(index) + _IMAGE_FILE_EXT;
        auto matImg = cv::imread(strFileName, cv::IMREAD_GRAYSCALE);
        if (! matImg.empty())
            stCmd.vecInputImgs.push_back(matImg);
        else
            break;
        ++ index;
    }

    VisionStatus enStatus = VisionStatus::OK;
    enStatus = VisionAlgorithm::calib3DBase(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseCalib3DHeight::StaticGetFolderPrefix() {
    return "Calib3DHeight";
}

VisionStatus LogCaseCalib3DHeight::WriteCmd(const PR_CALIB_3D_HEIGHT_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyEnableGF.c_str(), pstCmd->bEnableGaussianFilter);
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyReverseSeq.c_str(), pstCmd->bReverseSeq);
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyReverseHeight.c_str(), pstCmd->bReverseHeight);
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyUseThinnestPattern.c_str(), pstCmd->bUseThinnestPattern);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyRemoveHarmonicWaveK.c_str(), pstCmd->fRemoveHarmonicWaveK);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinAmplitude.c_str(), pstCmd->fMinAmplitude);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyBlockStepCount.c_str(), pstCmd->nBlockStepCount);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyBlockStepHeight.c_str(), pstCmd->fBlockStepHeight);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyResultImgGridRow.c_str(), pstCmd->nResultImgGridRow);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyResultImgGridCol.c_str(), pstCmd->nResultImgGridCol);
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::FileStorage fs(_strLogCasePath + _strInputYmlFileName, cv::FileStorage::WRITE);
    if (! fs.isOpened())
        return VisionStatus::OPEN_FILE_FAIL;

    cv::write(fs, _strKeyThickToThinK, pstCmd->matThickToThinK);
    fs.release();

    int index = 0;
    for (const auto &matImg : pstCmd->vecInputImgs) {
        String strFileName = _strLogCasePath + _IMAGE_FILE_PREFIX + std::to_string(index) + _IMAGE_FILE_EXT;
        cv::imwrite(strFileName, matImg);
        ++ index;
    }
    return VisionStatus::OK;
}

VisionStatus LogCaseCalib3DHeight::WriteRpy(const PR_CALIB_3D_HEIGHT_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseCalib3DHeight::RunLogCase() {
    PR_CALIB_3D_HEIGHT_CMD stCmd;
    PR_CALIB_3D_HEIGHT_RPY stRpy;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());

    stCmd.bEnableGaussianFilter = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyEnableGF.c_str(), false);
    stCmd.bReverseSeq = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyReverseSeq.c_str(), false);
    stCmd.bReverseHeight = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyReverseHeight.c_str(), false);
    stCmd.bUseThinnestPattern = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyUseThinnestPattern.c_str(), false);
    stCmd.fRemoveHarmonicWaveK = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyRemoveHarmonicWaveK.c_str(), 0.f));
    stCmd.fMinAmplitude = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinAmplitude.c_str(), 1.5));
    stCmd.nResultImgGridRow = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyResultImgGridRow.c_str(), 8);
    stCmd.nResultImgGridCol = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyResultImgGridCol.c_str(), 8);

    cv::FileStorage fs(_strLogCasePath + _strInputYmlFileName, cv::FileStorage::READ);
    cv::FileNode fileNode = fs[_strKeyThickToThinK];
    cv::read(fileNode, stCmd.matThickToThinK, cv::Mat());
    fileNode = fs[_strKeyThickToThinnestK];
    cv::read(fileNode, stCmd.matThickToThinnestK, cv::Mat());
    fs.release();

    int index = 0;
    while (true) {
        String strFileName = _strLogCasePath + _IMAGE_FILE_PREFIX + std::to_string(index) + _IMAGE_FILE_EXT;
        auto matImg = cv::imread(strFileName, cv::IMREAD_GRAYSCALE);
        if (! matImg.empty())
            stCmd.vecInputImgs.push_back(matImg);
        else
            break;
        ++ index;
    }

    VisionStatus enStatus = VisionStatus::OK;
    enStatus = VisionAlgorithm::calib3DHeight(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseCalc3DHeight::StaticGetFolderPrefix() {
    return "Calc3DHeight";
}

VisionStatus LogCaseCalc3DHeight::WriteCmd(const PR_CALC_3D_HEIGHT_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyEnableGF.c_str(), pstCmd->bEnableGaussianFilter);
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyReverseSeq.c_str(), pstCmd->bReverseSeq);
    ini.SetBoolValue(_CMD_SECTION.c_str(), _strKeyUseThinnestPattern.c_str(), pstCmd->bUseThinnestPattern);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyRemoveHarmonicWaveK.c_str(), pstCmd->fRemoveHarmonicWaveK);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinAmplitude.c_str(), pstCmd->fMinAmplitude);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyRmBetaJumpSpanX.c_str(), pstCmd->nRemoveBetaJumpSpanX);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyRmBetaJumpSpanY.c_str(), pstCmd->nRemoveBetaJumpSpanY);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyRmGammaJumpSpanX.c_str(), pstCmd->nRemoveGammaJumpSpanX);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyRmGammaJumpSpanY.c_str(), pstCmd->nRemoveGammaJumpSpanY);
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::FileStorage fs(_strLogCasePath + _strInputYmlFileName, cv::FileStorage::WRITE);
    if (! fs.isOpened())
        return VisionStatus::OPEN_FILE_FAIL;

    cv::write(fs, _strKeyThickToThinK, pstCmd->matThickToThinK);
    cv::write(fs, _strKeyThickToThinnestK, pstCmd->matThickToThinnestK);
    fs.release();

    int index = 0;
    for (const auto &matImg : pstCmd->vecInputImgs) {
        String strFileName = _strLogCasePath + _IMAGE_FILE_PREFIX + std::to_string(index) + _IMAGE_FILE_EXT;
        cv::imwrite(strFileName, matImg);
        ++ index;
    }
    return VisionStatus::OK;
}

VisionStatus LogCaseCalc3DHeight::WriteRpy(const PR_CALC_3D_HEIGHT_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseCalc3DHeight::RunLogCase() {
    PR_CALC_3D_HEIGHT_CMD stCmd;
    PR_CALC_3D_HEIGHT_RPY stRpy;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());

    stCmd.bEnableGaussianFilter = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyEnableGF.c_str(), false);
    stCmd.bReverseSeq = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyReverseSeq.c_str(), false);
    stCmd.bUseThinnestPattern = ini.GetBoolValue(_CMD_SECTION.c_str(), _strKeyUseThinnestPattern.c_str(), false);
    stCmd.fRemoveHarmonicWaveK = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyRemoveHarmonicWaveK.c_str(), 0.f));
    stCmd.fMinAmplitude = ToFloat(ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMinAmplitude.c_str(), 1.5));
    stCmd.nRemoveBetaJumpSpanX = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyRmBetaJumpSpanX.c_str(), 25);
    stCmd.nRemoveBetaJumpSpanY = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyRmBetaJumpSpanY.c_str(), 7);
    stCmd.nRemoveGammaJumpSpanX = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyRmGammaJumpSpanX.c_str(), 23);
    stCmd.nRemoveGammaJumpSpanY = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyRmGammaJumpSpanY.c_str(), 23);

    cv::FileStorage fs(_strLogCasePath + _strInputYmlFileName, cv::FileStorage::READ);
    cv::FileNode fileNode = fs[_strKeyThickToThinK];
    cv::read(fileNode, stCmd.matThickToThinK, cv::Mat());
    fileNode = fs[_strKeyThickToThinnestK];
    cv::read(fileNode, stCmd.matThickToThinnestK, cv::Mat());

    fs.release();

    int index = 0;
    while (true) {
        String strFileName = _strLogCasePath + _IMAGE_FILE_PREFIX + std::to_string(index) + _IMAGE_FILE_EXT;
        auto matImg = cv::imread(strFileName, cv::IMREAD_GRAYSCALE);
        if (! matImg.empty())
            stCmd.vecInputImgs.push_back(matImg);
        else
            break;
        ++ index;
    }

    VisionStatus enStatus = VisionStatus::OK;
    enStatus = VisionAlgorithm::calc3DHeight(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseCalcCameraMTF::StaticGetFolderPrefix() {
    return "CalcCameraMTF";
}

VisionStatus LogCaseCalcCameraMTF::WriteCmd(const PR_CALC_CAMERA_MTF_CMD *const pstCmd) {
    if (!_bReplay) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyVBigPatternROI.c_str(), _formatRect(pstCmd->rectVBigPatternROI).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyHBigPatternROI.c_str(), _formatRect(pstCmd->rectHBigPatternROI).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyVSmallPatternROI.c_str(), _formatRect(pstCmd->rectVSmallPatternROI).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyHSmallPatternROI.c_str(), _formatRect(pstCmd->rectHSmallPatternROI).c_str());
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pstCmd->matInputImg);

    return VisionStatus::OK;
}

VisionStatus LogCaseCalcCameraMTF::WriteRpy(const PR_CALC_CAMERA_MTF_RPY *const pstRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pstRpy->enStatus));
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyBigPatternAbsMtfV.c_str(), pstRpy->fBigPatternAbsMtfV);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyBigPatternAbsMtfH.c_str(), pstRpy->fBigPatternAbsMtfH);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySmallPatternAbsMtfV.c_str(), pstRpy->fSmallPatternAbsMtfV);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySmallPatternAbsMtfH.c_str(), pstRpy->fSmallPatternAbsMtfH);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySmallPatternRelMtfV.c_str(), pstRpy->fSmallPatternRelMtfV);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySmallPatternRelMtfH.c_str(), pstRpy->fSmallPatternRelMtfH);
    ini.SaveFile(cmdRpyFilePath.c_str());

    _zip();
    return VisionStatus::OK;
}

VisionStatus LogCaseCalcCameraMTF::RunLogCase() {
    PR_CALC_CAMERA_MTF_CMD stCmd;
    PR_CALC_CAMERA_MTF_RPY stRpy;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());

    stCmd.rectVBigPatternROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyVBigPatternROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.rectHBigPatternROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyHBigPatternROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.rectVSmallPatternROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyVSmallPatternROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.rectHSmallPatternROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyHSmallPatternROI.c_str(), _DEFAULT_RECT.c_str()));

    stCmd.matInputImg = cv::imread(_strLogCasePath + _IMAGE_NAME);
    VisionStatus enStatus = VisionStatus::OK;
    enStatus = VisionAlgorithm::calcCameraMTF(&stCmd, &stRpy, true);
    WriteRpy(&stRpy);
    return enStatus;
}

}
}