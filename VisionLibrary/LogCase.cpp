#include "LogCase.h"
#include <vector>
#include <sstream>
#include "opencv2/highgui.hpp"
#include "boost/format.hpp"
#include "boost/filesystem.hpp"
#include "StopWatch.h"
#include "SimpleIni.h"
#include "VisionAlgorithm.h"

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
        _strLogCasePath = _generateLogCaseName(GetTableName());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyAlgorithm.c_str(), static_cast<long>(pLrnTmplCmd->enAlgorithm) );
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyLrnWindow.c_str(), _formatRect(pLrnTmplCmd->rectLrn).c_str() );
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + "image.jpg", pLrnTmplCmd->mat );
    if ( ! pLrnTmplCmd->mask.empty() )
        cv::imwrite( _strLogCasePath + "mask.jpg", pLrnTmplCmd->mask );
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
    stLrnTmplCmd.mat = cv::imread( _strLogCasePath + "image.jpg" );
    stLrnTmplCmd.mask = cv::imread( _strLogCasePath + "mask.jpg" );

    PR_LRN_TMPL_RPY stLrnTmplRpy;

    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
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
        _strLogCasePath = _generateLogCaseName(GetTableName());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetLongValue(_CMD_SECTION.c_str(), _strKeyMethod.c_str(), static_cast<long>(pCmd->enMethod) );
    ini.SetValue(_CMD_SECTION.c_str(), _strKeyExpectedCtr.c_str(), _formatCoordinate(pCmd->ptRangeCtr).c_str() );
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyInnerRadius.c_str(), pCmd->fRangeInnterRadius );
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyOuterRadius.c_str(), pCmd->fRangeOutterRadius );
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), pCmd->fErrTol );
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), pCmd->nThreshold );    
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + "image.jpg", pCmd->matInput );
    return VisionStatus::OK;
}

VisionStatus LogCaseFitCircle::WriteRpy(PR_FIT_CIRCLE_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetLongValue  (_RPY_SECTION.c_str(), _strKeyStatus.c_str(), pRpy->nStatus );
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyResultCtr.c_str(), _formatCoordinate(pRpy->ptCircleCtr).c_str() );    
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyRadius.c_str(), pRpy->fRadius );
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + "result.jpg", pRpy->matResult );
    return VisionStatus::OK;
}

VisionStatus LogCaseFitCircle::RunLogCase()
{
    PR_FIT_CIRCLE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    stCmd.matInput = cv::imread( _strLogCasePath + "image.jpg" );
    stCmd.enMethod = static_cast<PR_FIT_CIRCLE_METHOD> ( ini.GetLongValue ( _CMD_SECTION.c_str(), _strKeyMethod.c_str(), ToInt32(PR_ALIGN_ALGORITHM::SURF) ) );
    stCmd.ptRangeCtr = _parseCoordinate ( ini.GetValue(_CMD_SECTION.c_str(), _strKeyExpectedCtr.c_str(), _DEFAULT_COORD.c_str() ) );
    stCmd.fRangeInnterRadius = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyInnerRadius.c_str(), 0 );
    stCmd.fRangeOutterRadius = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyOuterRadius.c_str(), 0 );
    stCmd.fErrTol = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), 0 );
    stCmd.nThreshold = ini.GetLongValue( _CMD_SECTION.c_str(), _strKeyThreshold.c_str(), 0);   

    PR_FIT_CIRCLE_RPY stRpy;

    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    enStatus = pVA->fitCircle ( &stCmd, &stRpy, true );

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
        _strLogCasePath = _generateLogCaseName(GetTableName());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _formatRect(pCmd->rectROI).c_str() );
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), pCmd->fErrTol );
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), pCmd->nThreshold );    
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + "image.jpg", pCmd->matInput );
    return VisionStatus::OK;
}

VisionStatus LogCaseFitLine::WriteRpy(PR_FIT_LINE_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetLongValue  (_RPY_SECTION.c_str(), _strKeyStatus.c_str(), pRpy->nStatus );
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySlope.c_str(), pRpy->fSlope );    
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept.c_str(), pRpy->fIntercept );
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyPoint1.c_str(), _formatCoordinate ( pRpy->stLine.pt1 ).c_str() );
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyPoint2.c_str(), _formatCoordinate ( pRpy->stLine.pt2 ).c_str() );
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + "result.jpg", pRpy->matResult );
    return VisionStatus::OK;
}

VisionStatus LogCaseFitLine::RunLogCase()
{
    PR_FIT_LINE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    stCmd.matInput = cv::imread( _strLogCasePath + "image.jpg" );    
    stCmd.rectROI = _parseRect ( ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _DEFAULT_RECT.c_str() ) );
    stCmd.fErrTol = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), 0 );
    stCmd.nThreshold = ini.GetLongValue( _CMD_SECTION.c_str(), _strKeyThreshold.c_str(), 0);   

    PR_FIT_LINE_RPY stRpy;

    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    enStatus = pVA->fitLine ( &stCmd, &stRpy, true );

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
        _strLogCasePath = _generateLogCaseName(GetTableName());
        bfs::path dir(_strLogCasePath);
        bfs::create_directories(dir);
    }

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow1.c_str(), _formatRect(pCmd->rectArrROI[0]).c_str());
    ini.SetValue(_CMD_SECTION.c_str(), _strKeySrchWindow2.c_str(), _formatRect(pCmd->rectArrROI[1]).c_str());
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), pCmd->fErrTol);
    ini.SetDoubleValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), pCmd->nThreshold);
    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + "image.jpg", pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCaseFitParallelLine::WriteRpy(PR_FIT_PARALLEL_LINE_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), pRpy->nStatus);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeySlope.c_str(), pRpy->fSlope);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept1.c_str(), pRpy->fIntercept1);
    ini.SetDoubleValue(_RPY_SECTION.c_str(), _strKeyIntercept2.c_str(), pRpy->fIntercept2);
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineOnePoint1.c_str(), _formatCoordinate(pRpy->stLine1.pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineOnePoint2.c_str(), _formatCoordinate(pRpy->stLine1.pt2).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineTwoPoint1.c_str(), _formatCoordinate(pRpy->stLine2.pt1).c_str());
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyLineTwoPoint1.c_str(), _formatCoordinate(pRpy->stLine2.pt2).c_str());
    ini.SaveFile(cmdRpyFilePath.c_str());
    //cv::imwrite(_strLogCasePath + "result.jpg", pRpy->matResult);
    return VisionStatus::OK;
}

VisionStatus LogCaseFitParallelLine::RunLogCase()
{
    PR_FIT_PARALLEL_LINE_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + "image.jpg");
    stCmd.rectROI = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.fErrTol = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyErrorTol.c_str(), 0);
    stCmd.nThreshold = ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyThreshold.c_str(), 0);

    PR_FIT_PARALLEL_LINE_RPY stRpy;

    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    enStatus = pVA->fitParallelLine(&stCmd, &stRpy, true);

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
        _strLogCasePath = _generateLogCaseName(GetTableName());
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
    cv::imwrite(_strLogCasePath + "image.jpg", pCmd->matInput);
    return VisionStatus::OK;
}

VisionStatus LogCaseSrchFiducial::WriteRpy(PR_SRCH_FIDUCIAL_MARK_RPY *pRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    ini.SetLongValue(_RPY_SECTION.c_str(), _strKeyStatus.c_str(), pRpy->nStatus);
    ini.SetValue(_RPY_SECTION.c_str(), _strKeyPos.c_str(), _formatCoordinate(pRpy->ptPos).c_str());

    ini.SaveFile(cmdRpyFilePath.c_str());
    cv::imwrite(_strLogCasePath + "result.jpg", pRpy->matResult);
    return VisionStatus::OK;
}

VisionStatus LogCaseSrchFiducial::RunLogCase()
{
    PR_SRCH_FIDUCIAL_MARK_CMD stCmd;
    VisionStatus enStatus;

    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _CMD_RPY_FILE_NAME;
    ini.LoadFile(cmdRpyFilePath.c_str());
    stCmd.matInput = cv::imread(_strLogCasePath + "image.jpg");
    stCmd.rectSrchRange = _parseRect(ini.GetValue(_CMD_SECTION.c_str(), _strKeySrchWindow.c_str(), _DEFAULT_RECT.c_str()));
    stCmd.enType = (PR_FIDUCIAL_MARK_TYPE)ini.GetLongValue(_CMD_SECTION.c_str(), _strKeyType.c_str(), 0);
    stCmd.fSize = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeySize.c_str(), 0);
    stCmd.fMargin = (float)ini.GetDoubleValue(_CMD_SECTION.c_str(), _strKeyMargin.c_str(), 0);

    PR_SRCH_FIDUCIAL_MARK_RPY stRpy;

    std::shared_ptr<VisionAlgorithm> pVA = VisionAlgorithm::create();
    enStatus = pVA->srchFiducialMark(&stCmd, &stRpy, true);

    WriteRpy(&stRpy);
    return enStatus;
}
}
}