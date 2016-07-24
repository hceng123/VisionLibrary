#include "LogCase.h"
#include "SimpleIni.h"
#include <vector>
#include <sstream>
#include <ctime>
#include "opencv2/highgui.hpp"
#include "boost/format.hpp"
#include "boost/filesystem.hpp"
#include "StopWatch.h"

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

String LogCase::_formatCoordinate(const Point2f &pt)
{
    char chArray[200];
    _snprintf( chArray, sizeof ( chArray), "%.4f, %.4f", pt.x, pt.y );
    return String (chArray);
}

Point2f LogCase::_parseCoordinate(const String &strCoordinate)
{
    Point2f pt(0.f, 0.f);
    StringVector vecStrCoordinate = split(strCoordinate, ',');
    if ( vecStrCoordinate.size() == 2 ) {
        pt.x = std::stof ( vecStrCoordinate[0] );
        pt.y = std::stof ( vecStrCoordinate[1] );
    }
    return pt;
}

String LogCase::_formatRect(const Rect2f &pt)
{
    char chArray[200];
    _snprintf(chArray, sizeof(chArray), "%.4f, %.4f, %.4f, %.4f", pt.tl().x, pt.tl().y, pt.br().x, pt.br().y);
    return String(chArray);
}

Rect2f LogCase::_parseRect(const String &strCoordinate)
{
    Point2f pt1(0.f, 0.f), pt2(0.f, 0.f);
    StringVector vecStrCoordinate = split(strCoordinate, ',');
    if ( vecStrCoordinate.size() == 4 ) {
        pt1.x = std::stof ( vecStrCoordinate[0] );
        pt1.y = std::stof ( vecStrCoordinate[1] );
        pt2.x = std::stof ( vecStrCoordinate[2] );
        pt2.y = std::stof ( vecStrCoordinate[3] );
    }
    return Rect2f(pt1, pt2);
}

String LogCase::_generateLogCaseName(const String &strFolderPrefix)
{
    auto timeT = std::time(nullptr);
    auto stTM = std::localtime(&timeT);
    CStopWatch stopWatch;
    auto nMilliseconds = stopWatch.AbsNow () - timeT * 1000;
    String strFmt = "_%s_%s_%s_%s_%s_%s_%s";
    String strTime = (boost::format(strFmt) % (stTM->tm_year + 1900) % (stTM->tm_mon + 1) % stTM->tm_mday % stTM->tm_hour % stTM->tm_min % stTM->tm_sec % nMilliseconds).str();
    String strLogCasePath = _strLogCasePath + strFolderPrefix + strTime + "\\";
    bfs::path dir(strLogCasePath);
    bfs::create_directories(dir);
    return strLogCasePath;
}

LogCaseLrnTmpl::LogCaseLrnTmpl(const String &strPath):LogCase(strPath)
{
    _strLogCasePath = _generateLogCaseName(_strFolderPrefix);
}

VisionStatus LogCaseLrnTmpl::WriteCmd(PR_LEARN_TMPL_CMD *pLrnTmplCmd)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _strCmdRpyFileName;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetLongValue(_strCmdSection.c_str(), _strKeyAlgorithm.c_str(), static_cast<long>(pLrnTmplCmd->enAlgorithm) );
    ini.SetValue(_strCmdSection.c_str(), _strKeyLrnWindow.c_str(), _formatRect(pLrnTmplCmd->rectLrn).c_str() );
    ini.SaveFile( cmdRpyFilePath.c_str() );
    cv::imwrite( _strLogCasePath + "image.jpg", pLrnTmplCmd->mat );
    if ( ! pLrnTmplCmd->mask.empty() )
        cv::imwrite( _strLogCasePath + "mask.jpg", pLrnTmplCmd->mask );
    return VisionStatus::OK;
}

VisionStatus LogCaseLrnTmpl::WriteRpy(PR_LEARN_TMPL_RPY *pLrnTmplRpy)
{
    CSimpleIni ini(false, false, false);
    auto cmdRpyFilePath = _strLogCasePath + _strCmdRpyFileName;
    ini.LoadFile( cmdRpyFilePath.c_str() );
    ini.SetLongValue(_strRpySection.c_str(), _strKeyStatus.c_str(), static_cast<long>(pLrnTmplRpy->nStatus) );
    ini.SetValue(_strRpySection.c_str(), _strKeyCenterPos.c_str(), _formatCoordinate (pLrnTmplRpy->ptCenter ).c_str() );

    ini.SaveFile( cmdRpyFilePath.c_str() );
    //cv::imwrite( _strLogCasePath + "image.jpg", pLrnTmplCmd->mat );
    //if ( ! pLrnTmplCmd->mask.empty() )
    //    cv::imwrite( _strLogCasePath + "mask.jpg", pLrnTmplCmd->mask );
    return VisionStatus::OK;
}

VisionStatus LogCaseLrnTmpl::RunLogCase()
{
    return VisionStatus::OK;
}

}
}