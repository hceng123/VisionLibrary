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

String LogCase::_formatCoordinate(const cv::Point2f &pt)
{
    char chArray[200];
    _snprintf( chArray, sizeof ( chArray), "%.4f, %.4f", pt.x, pt.y );
    return String (chArray);
}

cv::Point2f LogCase::_parseCoordinate(const String &strCoordinate)
{
    cv::Point2f pt(0.f, 0.f);
    StringVector vecStrCoordinate = split(strCoordinate, ',');
    if ( vecStrCoordinate.size() == 2 ) {
        pt.x = std::stof ( vecStrCoordinate[0] );
        pt.y = std::stof ( vecStrCoordinate[1] );
    }
    return pt;
}

String LogCase::_formatRect(const cv::Rect2f &pt)
{
    char chArray[200];
    _snprintf(chArray, sizeof(chArray), "%.4f, %.4f, %.4f, %.4f", pt.tl().x, pt.tl().y, pt.br().x, pt.br().y);
    return String(chArray);
}

cv::Rect2f LogCase::_parseRect(const String &strCoordinate)
{
    cv::Point2f pt1(0.f, 0.f), pt2(0.f, 0.f);
    StringVector vecStrCoordinate = split(strCoordinate, ',');
    if ( vecStrCoordinate.size() == 4 ) {
        pt1.x = std::stof ( vecStrCoordinate[0] );
        pt1.y = std::stof ( vecStrCoordinate[1] );
        pt2.x = std::stof ( vecStrCoordinate[2] );
        pt2.y = std::stof ( vecStrCoordinate[3] );
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
    if ( vecStrSize.size() == 2 ) {
        size.width = std::stoi ( vecStrSize[0] );
        size.height = std::stoi ( vecStrSize[1] );
    }
    return size;
}

String LogCase::_generateLogCaseName(const String &strFolderPrefix)
{
    auto timeT = std::time(nullptr);
    auto stTM = std::localtime(&timeT);
    CStopWatch stopWatch;
    auto nMilliseconds = stopWatch.AbsNow () - timeT * 1000;
    String strFmt = "_%04s_%02s_%02s_%02s_%02s_%02s_%03s";
    String strTime = (boost::format(strFmt) % (stTM->tm_year + 1900) % (stTM->tm_mon + 1) % stTM->tm_mday % stTM->tm_hour % stTM->tm_min % stTM->tm_sec % nMilliseconds).str();
    String strLogCasePath = _strLogCasePath + strFolderPrefix + strTime + "\\";    
    return strLogCasePath;
}

/*static*/ String LogCaseLrnTmpl::StaticGetFolderPrefix()
{
    return "LrnTmpl";
}

VisionStatus LogCaseLrnTmpl::WriteCmd(PR_LRN_TMPL_CMD *pLrnTmplCmd)
{
    if ( !_bReplay )    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), static_cast<long>(pLrnTmplCmd->enAlgorithm) );
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyLrnWindow.c_str(), _formatRect(pLrnTmplCmd->rectLrn).c_str() );
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + _IMAGE_NAME, pLrnTmplCmd->mat );
    if ( ! pLrnTmplCmd->mask.empty() )
        cv::imwrite( _strLogCasePath + _MASK_NAME, pLrnTmplCmd->mask );
    return VisionStatus::OK;
}

VisionStatus LogCaseLrnTmpl::WriteRpy(PR_LRN_TMPL_RPY *pLrnTmplRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), static_cast<long>(pLrnTmplRpy->nStatus) );
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyCenterPos.c_str(), _formatCoordinate (pLrnTmplRpy->ptCenter ).c_str() );

    ini.SaveFile( cmdRpyFilePath.c_str() );
    return VisionStatus::OK;
}

VisionStatus LogCaseLrnTmpl::RunLogCase()
{
    PR_LRN_TMPL_CMD stLrnTmplCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    stLrnTmplCmd.enAlgorithm = static_cast<PR_ALIGN_ALGORITHM> ( ini.GetLongValue ( _CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), ToInt32(PR_ALIGN_ALGORITHM::SURF) ) );
    stLrnTmplCmd.rectLrn = _parseRect ( ini.GetValue(_CMD_SECTION.c_str(), _strKeyLrnWindow.c_str(), "" ) );
    stLrnTmplCmd.mat = cv::imread( _strLogCasePath + _IMAGE_NAME );
    stLrnTmplCmd.mask = cv::imread( _strLogCasePath + _MASK_NAME );

    PR_LRN_TMPL_RPY stLrnTmplRpy;

    VisionAlgorithmPtr pVA = VisionAlgorithm::create();
    enStatus = pVA->lrnTmpl ( &stLrnTmplCmd, &stLrnTmplRpy, true );

    WriteRpy( &stLrnTmplRpy );
    return enStatus;
}

const String LogCaseLrnDevice::FOLDER_PREFIX = "LrnDevice";

/*static*/ String LogCaseFitCircle::StaticGetFolderPrefix()
{
    return "FitCircle";
}

VisionStatus LogCaseFitCircle::WriteCmd(PR_FIT_CIRCLE_CMD *pCmd)
{
    if ( !_bReplay )    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetValue(_CMD_SECTION.c_str(),     _strKeyROI.c_str(), _formatRect(pCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMethod.c_str(), static_cast<long>(pCmd->enMethod) );
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyRmNoiseMethod.c_str(), static_cast<long>(pCmd->enRmNoiseMethod) );
    ini.SetBoolValue( _CMD_SECTION.c_str(), _strKeyPreprocessed.c_str(), pCmd->bPreprocessed);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), pCmd->fErrTol );
    ini.SetBoolValue( _CMD_SECTION.c_str(), _strKeyAutoThreshold.c_str(), pCmd->bAutoThreshold);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), pCmd->nThreshold );
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), ToInt32 ( pCmd->enAttribute ) );
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + _IMAGE_NAME, pCmd->matInput );
    if ( ! pCmd->matMask.empty() )
        cv::imwrite( _strLogCasePath + _MASK_NAME,  pCmd->matMask );
    return VisionStatus::OK;
}

VisionStatus LogCaseFitCircle::WriteRpy(PR_FIT_CIRCLE_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetLongValue  (_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32( pRpy->enStatus ) );
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyResultCtr.c_str(), _formatCoordinate(pRpy->ptCircleCtr).c_str() );    
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyRadius.c_str(), pRpy->fRadius );
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult );
    return VisionStatus::OK;
}

VisionStatus LogCaseFitCircle::RunLogCase()
{
    PR_FIT_CIRCLE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    stCmd.matInput = cv::imread( _strLogCasePath + _IMAGE_NAME );

    String strMaskPath = _strLogCasePath + _MASK_NAME;
    if ( FileUtils::Exists ( strMaskPath ) )
        stCmd.matMask = cv::imread( strMaskPath, cv::IMREAD_GRAYSCALE );

    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enMethod = static_cast<PR_FIT_CIRCLE_METHOD> ( ini.GetLongValue ( _CMD_SECTION.c_str(), _strKeyMethod.c_str(), 0 ) );
    stCmd.enRmNoiseMethod = static_cast<PR_RM_FIT_NOISE_METHOD> ( ini.GetLongValue ( _CMD_SECTION.c_str(), _strKeyRmNoiseMethod.c_str(), 0 ) );
    stCmd.bPreprocessed = ini.GetBoolValue ( _CMD_SECTION.c_str(), _strKeyPreprocessed.c_str(), true );
    stCmd.fErrTol = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), 0 );
    stCmd.bAutoThreshold = ini.GetBoolValue ( _CMD_SECTION.c_str(), _strKeyAutoThreshold.c_str(), false );
    stCmd.nThreshold = ini.GetLongValue( _CMD_SECTION.c_str(), _strKeyThreshold.c_str(), 0);
    stCmd.enAttribute = static_cast<PR_OBJECT_ATTRIBUTE>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), 0 ) );

    PR_FIT_CIRCLE_RPY stRpy;
    enStatus = VisionAlgorithm::fitCircle( &stCmd, &stRpy, true );

    WriteRpy( &stRpy );
    return enStatus;
}

/*static*/ String LogCaseCircleRoundness::StaticGetFolderPrefix()
{
    return "CircleRoundness";
}

VisionStatus LogCaseCircleRoundness::WriteCmd(PR_CIRCLE_ROUNDNESS_CMD *pCmd)
{
    if ( !_bReplay )    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetValue(_CMD_SECTION.c_str(),     _strKeyROI.c_str(), _formatRect(pCmd->rectROI).c_str());
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + _IMAGE_NAME, pCmd->matInput );
    if ( ! pCmd->matMask.empty() )
        cv::imwrite( _strLogCasePath + _MASK_NAME,  pCmd->matMask );
    return VisionStatus::OK;
}

VisionStatus LogCaseCircleRoundness::WriteRpy(PR_CIRCLE_ROUNDNESS_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetLongValue  (_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32( pRpy->enStatus ) );
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyResultCtr.c_str(), _formatCoordinate(pRpy->ptCircleCtr).c_str() );    
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyRadius.c_str(), pRpy->fRadius );
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyRoundness.c_str(), pRpy->fRoundness );
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult );
    return VisionStatus::OK;
}

VisionStatus LogCaseCircleRoundness::RunLogCase()
{
    PR_CIRCLE_ROUNDNESS_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    stCmd.matInput = cv::imread( _strLogCasePath + _IMAGE_NAME );

    String strMaskPath = _strLogCasePath + _MASK_NAME;
    if ( FileUtils::Exists ( strMaskPath ) )
        stCmd.matMask = cv::imread( strMaskPath, cv::IMREAD_GRAYSCALE );

    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));

    PR_CIRCLE_ROUNDNESS_RPY stRpy;
    enStatus = VisionAlgorithm::circleRoundness( &stCmd, &stRpy, true );

    WriteRpy( &stRpy );
    return enStatus;
}

/*static*/ String LogCaseFitLine::StaticGetFolderPrefix()
{
    return "FitLine";
}

VisionStatus LogCaseFitLine::WriteCmd(PR_FIT_LINE_CMD *pCmd)
{
    if ( !_bReplay )    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );

    ini.SetValue      (_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(),   _formatRect(pCmd->rectROI).c_str() );
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(),     pCmd->fErrTol );
    ini.SetBoolValue  (_CMD_SECTION.c_str(), _strKeyPreprocessed.c_str(), pCmd->bPreprocessed);
    ini.SetLongValue  (_CMD_SECTION.c_str(), _strKeyThreshold.c_str(),    pCmd->nThreshold );
    ini.SetLongValue  (_CMD_SECTION.c_str(), _strKeyAttribute.c_str(),    ToInt32 ( pCmd->enAttribute ) );

    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + _IMAGE_NAME, pCmd->matInput );
    if ( ! pCmd->matMask.empty() )
        cv::imwrite( _strLogCasePath + _MASK_NAME,  pCmd->matMask );
    return VisionStatus::OK;
}

VisionStatus LogCaseFitLine::WriteRpy(PR_FIT_LINE_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );

    ini.SetLongValue  (_RPY_SECTION.c_str(), _strKeyStatus.c_str(),      ToInt32(pRpy->enStatus) );
    ini.SetBoolValue  (_RPY_SECTION.c_str(), _strKeyReversedFit.c_str(), pRpy->bReversedFit );
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySlope.c_str(),       pRpy->fSlope );    
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept.c_str(),   pRpy->fIntercept );
    ini.SetValue      (_RPY_SECTION.c_str(), _strKeyPoint1.c_str(),      _formatCoordinate ( pRpy->stLine.pt1 ).c_str() );
    ini.SetValue      (_RPY_SECTION.c_str(), _strKeyPoint2.c_str(),      _formatCoordinate ( pRpy->stLine.pt2 ).c_str() );

    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult );
    return VisionStatus::OK;
}

VisionStatus LogCaseFitLine::RunLogCase()
{
    PR_FIT_LINE_CMD stCmd;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    String strMaskPath = _strLogCasePath + _MASK_NAME;
    if ( FileUtils::Exists ( strMaskPath ) )
        stCmd.matMask = cv::imread( strMaskPath, cv::IMREAD_GRAYSCALE );

    stCmd.matInput = cv::imread( _strLogCasePath + _IMAGE_NAME );    
    stCmd.rectROI = _parseRect ( ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _DEFAULT_RECT.c_str() ) );
    stCmd.fErrTol = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), 0 );
    stCmd.bPreprocessed = ini.GetBoolValue ( _CMD_SECTION.c_str(), _strKeyPreprocessed.c_str(), true );
    stCmd.nThreshold = ini.GetLongValue( _CMD_SECTION.c_str(), _strKeyThreshold.c_str(), 0);
    stCmd.enAttribute = static_cast<PR_OBJECT_ATTRIBUTE>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), 0 ) );

    PR_FIT_LINE_RPY stRpy;
    VisionStatus enStatus = VisionAlgorithm::fitLine ( &stCmd, &stRpy, true );
    WriteRpy( &stRpy );
    return enStatus;
}

/*static*/ String LogCaseDetectLine::StaticGetFolderPrefix()
{
    return "DetectLine";
}

VisionStatus LogCaseDetectLine::WriteCmd(PR_DETECT_LINE_CMD *pCmd)
{
    if ( !_bReplay )    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _formatRect(pCmd->rectROI).c_str() );
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyDir.c_str(), ToInt32( pCmd->enDetectDir ) );
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + _IMAGE_NAME, pCmd->matInput );
    if ( ! pCmd->matMask.empty() )
        cv::imwrite( _strLogCasePath + _MASK_NAME,  pCmd->matMask );
    return VisionStatus::OK;
}

VisionStatus LogCaseDetectLine::WriteRpy(PR_DETECT_LINE_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetLongValue  (_RPY_SECTION.c_str(), _strKeyStatus.c_str(),      ToInt32(pRpy->enStatus) );
    ini.SetBoolValue  (_RPY_SECTION.c_str(), _strKeyReversedFit.c_str(), pRpy->bReversedFit );
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySlope.c_str(),       pRpy->fSlope );    
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept.c_str(),   pRpy->fIntercept );
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyPoint1.c_str(), _formatCoordinate ( pRpy->stLine.pt1 ).c_str() );
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyPoint2.c_str(), _formatCoordinate ( pRpy->stLine.pt2 ).c_str() );
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult );
    return VisionStatus::OK;
}

VisionStatus LogCaseDetectLine::RunLogCase()
{
    PR_DETECT_LINE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    String strMaskPath = _strLogCasePath + _MASK_NAME;
    if ( FileUtils::Exists ( strMaskPath ) )
        stCmd.matMask = cv::imread( strMaskPath, cv::IMREAD_GRAYSCALE );

    stCmd.matInput = cv::imread( _strLogCasePath + _IMAGE_NAME );    
    stCmd.rectROI = _parseRect ( ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str() ) );
    stCmd.enDetectDir = static_cast<PR_DETECT_LINE_DIR>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyDir.c_str(), 0 ) );

    PR_DETECT_LINE_RPY stRpy;
    enStatus = VisionAlgorithm::detectLine(&stCmd, &stRpy, true);

    WriteRpy( &stRpy );
    return enStatus;
}

/*static*/ String LogCaseFitParallelLine::StaticGetFolderPrefix()
{
    return "FitParallelLine";
}

VisionStatus LogCaseFitParallelLine::WriteCmd(PR_FIT_PARALLEL_LINE_CMD *pCmd)
{
    if (!_bReplay)    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow1.c_str(), _formatRect(pCmd->rectArrROI[0]).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow2.c_str(), _formatRect(pCmd->rectArrROI[1]).c_str());
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), pCmd->fErrTol);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), pCmd->nThreshold);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), ToInt32 ( pCmd->enAttribute ) );
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCaseFitParallelLine::WriteRpy(PR_FIT_PARALLEL_LINE_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pRpy->enStatus) );
    ini.SetBoolValue  (_RPY_SECTION.c_str(), _strKeyReversedFit.c_str(), pRpy->bReversedFit );
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySlope.c_str(), pRpy->fSlope);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept1.c_str(), pRpy->fIntercept1);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept2.c_str(), pRpy->fIntercept2);
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineOnePoint1.c_str(), _formatCoordinate(pRpy->stLine1.pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineOnePoint2.c_str(), _formatCoordinate(pRpy->stLine1.pt2).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineTwoPoint1.c_str(), _formatCoordinate(pRpy->stLine2.pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineTwoPoint2.c_str(), _formatCoordinate(pRpy->stLine2.pt2).c_str());
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult);
    return VisionStatus::OK;
}

VisionStatus LogCaseFitParallelLine::RunLogCase()
{
    PR_FIT_PARALLEL_LINE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectArrROI[0] = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow1.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.rectArrROI[1] = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow2.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.fErrTol = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), 0);
    stCmd.nThreshold = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), 0);
    stCmd.enAttribute = static_cast<PR_OBJECT_ATTRIBUTE>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), 0 ) );

    PR_FIT_PARALLEL_LINE_RPY stRpy;

    VisionAlgorithmPtr pVA = VisionAlgorithm::create();
    enStatus = pVA->fitParallelLine(&stCmd, &stRpy, true);

    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseFitRect::StaticGetFolderPrefix()
{
    return "FitRect";
}

VisionStatus LogCaseFitRect::WriteCmd(PR_FIT_RECT_CMD *pCmd)
{
    if (!_bReplay)    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow1.c_str(), _formatRect(pCmd->rectArrROI[0]).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow2.c_str(), _formatRect(pCmd->rectArrROI[1]).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow3.c_str(), _formatRect(pCmd->rectArrROI[0]).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow4.c_str(), _formatRect(pCmd->rectArrROI[1]).c_str());
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), pCmd->fErrTol);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), pCmd->nThreshold);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), ToInt32 ( pCmd->enAttribute ) );
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCaseFitRect::WriteRpy(PR_FIT_RECT_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(),   ToInt32(pRpy->enStatus));
    ini.SetBoolValue  (_RPY_SECTION.c_str(), _strKeyReversedFit1.c_str(), pRpy->bLineOneReversedFit );
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySlope1.c_str(), pRpy->fSlope1);
    ini.SetBoolValue  (_RPY_SECTION.c_str(), _strKeyReversedFit2.c_str(), pRpy->bLineTwoReversedFit );
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySlope2.c_str(), pRpy->fSlope2);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept1.c_str(), pRpy->fArrIntercept[0]);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept2.c_str(), pRpy->fArrIntercept[1]);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept3.c_str(), pRpy->fArrIntercept[2]);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept4.c_str(), pRpy->fArrIntercept[3]);
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineOnePoint1.c_str(), _formatCoordinate(pRpy->arrLines[0].pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineOnePoint2.c_str(), _formatCoordinate(pRpy->arrLines[0].pt2).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineTwoPoint1.c_str(), _formatCoordinate(pRpy->arrLines[1].pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineTwoPoint2.c_str(), _formatCoordinate(pRpy->arrLines[1].pt2).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineThreePoint1.c_str(), _formatCoordinate(pRpy->arrLines[2].pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineThreePoint2.c_str(), _formatCoordinate(pRpy->arrLines[2].pt2).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineFourPoint1.c_str(), _formatCoordinate(pRpy->arrLines[3].pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineFourPoint2.c_str(), _formatCoordinate(pRpy->arrLines[3].pt2).c_str());
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult);
    return VisionStatus::OK;
}

VisionStatus LogCaseFitRect::RunLogCase()
{
    PR_FIT_RECT_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectArrROI[0] = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow1.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.rectArrROI[1] = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow2.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.rectArrROI[2] = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow3.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.rectArrROI[3] = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow4.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.fErrTol = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), 0);
    stCmd.nThreshold = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), 0);
    stCmd.enAttribute = static_cast<PR_OBJECT_ATTRIBUTE>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), 0 ) );

    PR_FIT_RECT_RPY stRpy;

    VisionAlgorithmPtr pVA = VisionAlgorithm::create();
    enStatus = pVA->fitRect(&stCmd, &stRpy, true);

    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseSrchFiducial::StaticGetFolderPrefix()
{
    return "SrchFiducial";
}

VisionStatus LogCaseSrchFiducial::WriteCmd(PR_SRCH_FIDUCIAL_MARK_CMD *pCmd)
{
    if (!_bReplay)    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _formatRect(pCmd->rectSrchRange).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyType.c_str(), ToInt32(pCmd->enType));
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeySize.c_str(), pCmd->fSize);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyMargin.c_str(), pCmd->fMargin);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCaseSrchFiducial::WriteRpy(PR_SRCH_FIDUCIAL_MARK_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pRpy->enStatus));
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyPos.c_str(), _formatCoordinate(pRpy->ptPos).c_str());

    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult);
    return VisionStatus::OK;
}

VisionStatus LogCaseSrchFiducial::RunLogCase()
{
    PR_SRCH_FIDUCIAL_MARK_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectSrchRange = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enType = (PR_FIDUCIAL_MARK_TYPE)ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyType.c_str(), 0);
    stCmd.fSize = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeySize.c_str(), 0);
    stCmd.fMargin = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMargin.c_str(), 0);

    PR_SRCH_FIDUCIAL_MARK_RPY stRpy;

    VisionAlgorithmPtr pVA = VisionAlgorithm::create();
    enStatus = pVA->srchFiducialMark(&stCmd, &stRpy, true);

    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseOcr::StaticGetFolderPrefix()
{
    return "Ocr";
}

VisionStatus LogCaseOcr::WriteCmd(PR_OCR_CMD *pCmd)
{
    if (!_bReplay)    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _formatRect(pCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyDirection.c_str(), ToInt32(pCmd->enDirection));
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCaseOcr::WriteRpy(PR_OCR_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32(pRpy->enStatus));
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyResultStr.c_str(), pRpy->strResult.c_str());

    ini.SaveFile(cmdRpyFilePath.c_str());
    return VisionStatus::OK;
}

VisionStatus LogCaseOcr::RunLogCase()
{
    PR_OCR_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enDirection = (PR_DIRECTION)ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyDirection.c_str(), 0);

    PR_OCR_RPY stRpy;

    VisionAlgorithmPtr pVA = VisionAlgorithm::create();
    enStatus = pVA->ocr(&stCmd, &stRpy, true);

    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseRemoveCC::StaticGetFolderPrefix()
{
    return "RemoveCC";
}

VisionStatus LogCaseRemoveCC::WriteCmd(PR_REMOVE_CC_CMD *pCmd)
{
    if (!_bReplay)    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(),     _strKeyROI.c_str(), _formatRect(pCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyConnectivity.c_str(), pCmd->nConnectivity);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyCompareType.c_str(),  ToInt32(pCmd->enCompareType));
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyAreaThreshold.c_str(),  pCmd->fAreaThreshold);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCaseRemoveCC::WriteRpy(PR_REMOVE_CC_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(),    ToInt32 ( pRpy->enStatus ) );
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyTotalCC.c_str(),   pRpy->nTotalCC);
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyRemovedCC.c_str(), pRpy->nRemovedCC);
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite(_strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult);
    return VisionStatus::OK;
}

VisionStatus LogCaseRemoveCC::RunLogCase()
{
    PR_REMOVE_CC_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.nConnectivity = (Int16)ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyConnectivity.c_str(), 8);
    stCmd.enCompareType = static_cast<PR_COMPARE_TYPE>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyCompareType.c_str(),  0 ) );
    stCmd.fAreaThreshold = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyAreaThreshold.c_str(), 100 );

    PR_REMOVE_CC_RPY stRpy;
    enStatus = VisionAlgorithm::removeCC( &stCmd, &stRpy, true );
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseDetectEdge::StaticGetFolderPrefix()
{
    return "DetectEdge";
}

VisionStatus LogCaseDetectEdge::WriteCmd(PR_DETECT_EDGE_CMD *pCmd)
{
    if (!_bReplay)    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(),     _strKeyROI.c_str(), _formatRect(pCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold1.c_str(), pCmd->nThreshold1);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold2.c_str(), pCmd->nThreshold2);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyApertureSize.c_str(), pCmd->nApertureSize);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCaseDetectEdge::WriteRpy(PR_DETECT_EDGE_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), ToInt32 ( pRpy->enStatus ) );
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite ( _strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult );
    return VisionStatus::OK;
}

VisionStatus LogCaseDetectEdge::RunLogCase()
{
    PR_DETECT_EDGE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.nThreshold1 = (Int16)ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold1.c_str(), 100);
    stCmd.nThreshold2 = (Int16)ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold2.c_str(), 40);
    stCmd.nApertureSize = (Int16)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyApertureSize.c_str(), 3 );

    PR_DETECT_EDGE_RPY stRpy;
    enStatus = VisionAlgorithm::detectEdge( &stCmd, &stRpy, true );
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseAutoThreshold::StaticGetFolderPrefix()
{
    return "AutoThreshold";
}

VisionStatus LogCaseAutoThreshold::WriteCmd(PR_AUTO_THRESHOLD_CMD *pCmd)
{
    if (!_bReplay)    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(),     _strKeyROI.c_str(), _formatRect(pCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyThresholdNum.c_str(), pCmd->nThresholdNum);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCaseAutoThreshold::WriteRpy(PR_AUTO_THRESHOLD_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(),    ToInt32 ( pRpy->enStatus ) );
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyThreshold.c_str(), _formatVector<Int16>(pRpy->vecThreshold).c_str() );
    ini.SaveFile(cmdRpyFilePath.c_str());

    return VisionStatus::OK;
}

VisionStatus LogCaseAutoThreshold::RunLogCase()
{
    PR_AUTO_THRESHOLD_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.nThresholdNum = (Int16)ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThresholdNum.c_str(), 1);

    PR_AUTO_THRESHOLD_RPY stRpy;
    enStatus = VisionAlgorithm::autoThreshold( &stCmd, &stRpy, true );
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseFillHole::StaticGetFolderPrefix()
{
    return "FillHole";
}

VisionStatus LogCaseFillHole::WriteCmd(PR_FILL_HOLE_CMD *pCmd)
{
    if (!_bReplay)    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(),     _strKeyROI.c_str(), _formatRect(pCmd->rectROI).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), ToInt32(pCmd->enAttribute));
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMethod.c_str(), ToInt32(pCmd->enMethod));
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMorphShape.c_str(), ToInt32(pCmd->enMorphShape));
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMorphKernelX.c_str(), pCmd->szMorphKernel.width);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMorphKernelY.c_str(), pCmd->szMorphKernel.height);
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMorphIteration.c_str(), pCmd->nMorphIteration);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME, pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCaseFillHole::WriteRpy(PR_FILL_HOLE_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(),    ToInt32 ( pRpy->enStatus ) );
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite ( _strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult );

    return VisionStatus::OK;
}

VisionStatus LogCaseFillHole::RunLogCase()
{
    PR_FILL_HOLE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + _IMAGE_NAME);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enAttribute = static_cast<PR_OBJECT_ATTRIBUTE>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyAttribute.c_str(), 0 ) );
    stCmd.enMethod = static_cast<PR_FILL_HOLE_METHOD>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMethod.c_str(), 0 ) );
    stCmd.enMorphShape = static_cast<cv::MorphShapes>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMorphShape.c_str(), 0 ) );
    stCmd.szMorphKernel.width  = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMorphKernelX.c_str(), 0 );
    stCmd.szMorphKernel.height = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMorphKernelY.c_str(), 0 );
    stCmd.nMorphIteration = (Int16)( ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMorphIteration.c_str(), 3 ) );

    PR_FILL_HOLE_RPY stRpy;
    enStatus = VisionAlgorithm::fillHole( &stCmd, &stRpy, true );
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseMatchTmpl::StaticGetFolderPrefix()
{
    return "MatchTmpl";
}

VisionStatus LogCaseMatchTmpl::WriteCmd(PR_MATCH_TEMPLATE_CMD *pCmd)
{
    if (!_bReplay)    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(),     _strKeySrchWindow.c_str(), _formatRect(pCmd->rectSrchWindow).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMotion.c_str(), ToInt32(pCmd->enMotion));
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME,     pCmd->matInput);
    cv::imwrite(_strLogCasePath + _TMPL_FILE_NAME, pCmd->matTmpl);
    return VisionStatus::OK;
}

VisionStatus LogCaseMatchTmpl::WriteRpy(PR_MATCH_TEMPLATE_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(),    ToInt32 ( pRpy->enStatus ) );
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite ( _strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult );

    return VisionStatus::OK;
}

VisionStatus LogCaseMatchTmpl::RunLogCase()
{
    PR_MATCH_TEMPLATE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + _IMAGE_NAME,     cv::IMREAD_GRAYSCALE);
    stCmd.matTmpl  = cv::imread(_strLogCasePath + _TMPL_FILE_NAME, cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enMotion = static_cast<PR_OBJECT_MOTION>(ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyMotion.c_str(), 0 ) );

    PR_MATCH_TEMPLATE_RPY stRpy;
    enStatus = VisionAlgorithm::matchTemplate( &stCmd, &stRpy, true );
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCasePickColor::StaticGetFolderPrefix()
{
    return "PickColor";
}

VisionStatus LogCasePickColor::WriteCmd(PR_PICK_COLOR_CMD *pCmd)
{
    if (!_bReplay)    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(),     _strKeyROI.c_str(), _formatRect(pCmd->rectROI).c_str());
    ini.SetValue(_CMD_SECTION.c_str(),     _strKeyPickPoint.c_str(), _formatCoordinate(pCmd->ptPick).c_str());
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyColorDiff.c_str(), pCmd->nColorDiff );
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyGrayDiff.c_str(),  pCmd->nGrayDiff );
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME,     pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCasePickColor::WriteRpy(PR_PICK_COLOR_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(),    ToInt32 ( pRpy->enStatus ) );
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite ( _strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult );

    return VisionStatus::OK;
}

VisionStatus LogCasePickColor::RunLogCase()
{
    PR_PICK_COLOR_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeyROI.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.ptPick = _parseCoordinate(ini.GetValue(_CMD_SECTION.c_str(), _strKeyPickPoint.c_str(), _DEFAULT_COORD.c_str()));
    stCmd.nColorDiff = static_cast<Int16> ( ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyColorDiff.c_str(), 10 ) );
    stCmd.nGrayDiff  = static_cast<Int16> ( ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyGrayDiff.c_str(),  10 ) );

    PR_PICK_COLOR_RPY stRpy;
    enStatus = VisionAlgorithm::pickColor( &stCmd, &stRpy, true );
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseCalibrateCamera::StaticGetFolderPrefix()
{
    return "CalibrateCamera";
}

VisionStatus LogCaseCalibrateCamera::WriteCmd(const PR_CALIBRATE_CAMERA_CMD *const pCmd)
{
    if (!_bReplay)    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyBoardPattern.c_str(), _formatSize(pCmd->szBoardPattern).c_str());
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyObjectSize.c_str(), pCmd->fPatternDist );
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + _IMAGE_NAME,     pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCaseCalibrateCamera::WriteRpy(const PR_CALIBRATE_CAMERA_RPY * const pRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());    
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(),    ToInt32 ( pRpy->enStatus ) );
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyResolutionX.c_str(), pRpy->dResolutionX );
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyResolutionY.c_str(), pRpy->dResolutionY );
    ini.SaveFile(cmdRpyFilePath.c_str());

    return VisionStatus::OK;
}

VisionStatus LogCaseCalibrateCamera::RunLogCase()
{
    PR_CALIBRATE_CAMERA_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);
    stCmd.szBoardPattern = _parseSize(ini.GetValue(_CMD_SECTION.c_str(), _strKeyBoardPattern.c_str(), _DEFAULT_SIZE.c_str()));
    stCmd.fPatternDist = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyObjectSize.c_str(), 0 );

    PR_CALIBRATE_CAMERA_RPY stRpy;
    enStatus = VisionAlgorithm::calibrateCamera ( &stCmd, &stRpy, true );
    WriteRpy(&stRpy);
    return enStatus;
}

/*static*/ String LogCaseRestoreImg::StaticGetFolderPrefix()
{
    return "RestoreImg";
}

VisionStatus LogCaseRestoreImg::WriteCmd(const PR_RESTORE_IMG_CMD *const pCmd)
{
    if (!_bReplay)    {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;

    cv::FileStorage fs ( _strLogCasePath + _strYmlFile, cv::FileStorage::WRITE );
    cv::write(fs , _strKeyMapX, pCmd->vecMatRestoreImage[0] );
    cv::write(fs , _strKeyMapY, pCmd->vecMatRestoreImage[1] );
    fs.release();

    cv::imwrite(_strLogCasePath + _IMAGE_NAME,     pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCaseRestoreImg::WriteRpy(PR_RESTORE_IMG_RPY *const pRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());    
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(),    ToInt32 ( pRpy->enStatus ) );
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite ( _strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult );
    return VisionStatus::OK;
}

VisionStatus LogCaseRestoreImg::RunLogCase() {
    PR_RESTORE_IMG_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);
    
    cv::FileStorage fs ( _strLogCasePath + _strYmlFile, cv::FileStorage::READ );
    cv::Mat matMapX, matMapY;
    cv::FileNode fileNodeMapX = fs[_strKeyMapX];
    cv::FileNode fileNodeMapY = fs[_strKeyMapY];
    cv::read ( fileNodeMapX , matMapX );
    cv::read ( fileNodeMapY , matMapY );
    fs.release();

    stCmd.vecMatRestoreImage.push_back ( matMapX );
    stCmd.vecMatRestoreImage.push_back ( matMapY );

    PR_RESTORE_IMG_RPY stRpy;
    enStatus = VisionAlgorithm::restoreImage ( &stCmd, &stRpy, true );
    WriteRpy(&stRpy);

    if ( VisionStatus::OK == enStatus ) {
        cv::Mat matInputFloat, matResultFloat;
        stCmd.matInput.convertTo (matInputFloat,  CV_32FC1);
        stRpy.matResult.convertTo(matResultFloat, CV_32FC1);

        cv::Mat matDiff = cv::Scalar::all(255.f) - (matInputFloat - matResultFloat);
        cv::imwrite(_strLogCasePath + _strDiffImg, matDiff);
    }
    return enStatus;
}

/*static*/ String LogCaseAutoLocateLead::StaticGetFolderPrefix()
{
    return "AutoLocateLead";
}

VisionStatus LogCaseAutoLocateLead::WriteCmd(const PR_AUTO_LOCATE_LEAD_CMD *const pCmd) {
    if ( !_bReplay ) {
        _strLogCasePath = _generateLogCaseName(GetFolderPrefix());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _formatRect(pCmd->rectSrchWindow).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyChipWindow.c_str(), _formatRect(pCmd->rectChipBody).c_str());
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite(_strLogCasePath + _IMAGE_NAME,     pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCaseAutoLocateLead::WriteRpy(PR_AUTO_LOCATE_LEAD_RPY *const pRpy) {
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());    
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(),    ToInt32 ( pRpy->enStatus ) );
    for ( int i = 0; i < pRpy->vecLeadLocation.size(); ++ i ) {
        String strKey = _strLeadLocation + std::to_string(i);
        auto rect = pRpy->vecLeadLocation[i];
        ini.SetValue(_RPY_SECTION.c_str(), strKey.c_str(), _formatRect(rect).c_str());
    }
    ini.SaveFile(cmdRpyFilePath.c_str());

    cv::imwrite ( _strLogCasePath + _RESULT_IMAGE_NAME, pRpy->matResult );
    return VisionStatus::OK;
}

VisionStatus LogCaseAutoLocateLead::RunLogCase() {
    PR_AUTO_LOCATE_LEAD_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + _IMAGE_NAME, cv::IMREAD_COLOR);  

    stCmd.rectSrchWindow = _parseRect ( ini.GetValue ( _CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _DEFAULT_RECT.c_str() ) );
    stCmd.rectChipBody   = _parseRect ( ini.GetValue ( _CMD_SECTION.c_str(), _strKeyChipWindow.c_str(), _DEFAULT_RECT.c_str() ) );
    PR_AUTO_LOCATE_LEAD_RPY stRpy;
    enStatus = VisionAlgorithm::autoLocateLead ( &stCmd, &stRpy, true );
    WriteRpy(&stRpy);    
    return enStatus;
}

}
}