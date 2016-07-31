#ifndef _LOG_CASE_H_
#define _LOG_CASE_H_

#include "BaseType.h"
#include "VisionHeader.h"

namespace AOI
{
namespace Vision
{

class LogCase
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
    const String _CMD_RPY_FILE_NAME = "cmdrpy.log";
    const String _CMD_SECTION = "CMD";
    const String _RPY_SECTION = "RPY";
};

class LogCaseLrnTmpl:public LogCase
{
public:
    explicit LogCaseLrnTmpl(const String &strPath, bool bReplay = false);
    VisionStatus WriteCmd(PR_LRN_TMPL_CMD *pLrnTmplCmd);
    VisionStatus WriteRpy(PR_LRN_TMPL_RPY *pLrnTmplRpy);
    virtual VisionStatus RunLogCase() override;
    const static String FOLDER_PREFIX;
private:    
    const String _strKeyAlgorithm   =   "Algorithm";
    const String _strKeyLrnWindow   =   "LrnWindow";
    const String _strKeyStatus      =   "Status";
    const String _strKeyCenterPos   =   "Center";
};

class LogCaseLrnDevice:public LogCase
{
public:
    explicit LogCaseLrnDevice(const String &strPath);
    VisionStatus WriteCmd(PR_LRN_DEVICE_CMD *pLrnTmplCmd);
    VisionStatus WriteRpy(PR_LRN_DEVICE_RPY *pLrnTmplRpy);
    virtual VisionStatus RunLogCase() override;
    const static String FOLDER_PREFIX;
private:
};

}
}
#endif