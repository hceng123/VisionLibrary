#ifndef _LOG_CASE_H_
#define _LOG_CASE_H_

#include "BaseType.h"
#include "VisionHeader.h"

namespace AOI
{
namespace Vision
{

class LogCase : private Uncopyable
{
public:
    explicit LogCase(const String &strPath):_strLogCasePath(strPath)  {}
    virtual VisionStatus RunLogCase()  = 0;
protected:
    String _formatCoordinate(const Point2f &pt);
    Point2f _parseCoordinate(const String &strCoordinate);
    String _formatRect(const Rect2f &pt);
    Rect2f _parseRect(const String &strCoordinate);
    String _generateLogCaseName(const String &strFolderPrefix);
    String      _strLogCasePath;
    const String _strCmdRpyFileName = "cmdrpy.log";
    const String _strCmdSection = "CMD";
    const String _strRpySection = "RPY";
};

class LogCaseLrnTmpl:public LogCase
{
public:
    explicit LogCaseLrnTmpl(const String &strPath);
    VisionStatus WriteCmd(PR_LEARN_TMPL_CMD *pLrnTmplCmd);
    VisionStatus WriteRpy(PR_LEARN_TMPL_RPY *pLrnTmplRpy);
    virtual VisionStatus RunLogCase() override;
private:
    const String _strFolderPrefix   =   "LrnTmpl";
    const String _strKeyAlgorithm   =   "Algorithm";
    const String _strKeyLrnWindow   =   "LrnWindow";
    const String _strKeyStatus      =   "Status";
    const String _strKeyCenterPos   =   "Center";
};

}
}
#endif