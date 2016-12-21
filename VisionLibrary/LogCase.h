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
    explicit LogCase(const String &strPath, bool bReplay = false);
    virtual String GetFolderPrefix() const = 0;
    virtual VisionStatus RunLogCase()  = 0;
protected:
    String _formatCoordinate(const cv::Point2f &pt);
    cv::Point2f _parseCoordinate(const String &strCoordinate);
    String _formatRect(const cv::Rect2f &pt);
    cv::Rect2f _parseRect(const String &strCoordinate);
    String _generateLogCaseName(const String &strFolderPrefix);
    String      _strLogCasePath;
    const String _CMD_RPY_FILE_NAME = "cmdrpy.log";
    const String _CMD_SECTION = "CMD";
    const String _RPY_SECTION = "RPY";
    const String _DEFAULT_COORD = "0, 0";
    const String _DEFAULT_RECT = "0, 0, 0, 0";
    bool         _bReplay;
};

class LogCaseLrnTmpl:public LogCase
{
public:
    explicit LogCaseLrnTmpl(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_LRN_TMPL_CMD *pCmd);
    VisionStatus WriteRpy(PR_LRN_TMPL_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:    
    const String _strKeyAlgorithm   =   "Algorithm";
    const String _strKeyLrnWindow   =   "LrnWindow";
    const String _strKeyStatus      =   "Status";
    const String _strKeyCenterPos   =   "Center";
};

class LogCaseLrnDevice:public LogCase
{
public:
    explicit LogCaseLrnDevice(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_LRN_DEVICE_CMD *pLrnTmplCmd);
    VisionStatus WriteRpy(PR_LRN_DEVICE_RPY *pLrnTmplRpy);
    virtual VisionStatus RunLogCase() override;
    const static String FOLDER_PREFIX;
private:
};

class LogCaseFitCircle : public LogCase
{
public:
    explicit LogCaseFitCircle(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_FIT_CIRCLE_CMD *pCmd);
    VisionStatus WriteRpy(PR_FIT_CIRCLE_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyMethod      = "Method";
    const String _strKeyExpectedCtr = "ExpectedCtr";
    const String _strKeyInnerRadius = "InnerRadius";
    const String _strKeyOuterRadius = "OuterRadius";
    const String _strKeyErrorTol    = "ErrorTolerance";
    const String _strKeyThreshold   = "Threshold";
    const String _strKeyStatus      = "Status";
    const String _strKeyResultCtr   = "ResultCtr";
    const String _strKeyRadius      = "Radius";
};

class LogCaseFitLine : public LogCase
{
public:
    explicit LogCaseFitLine(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_FIT_LINE_CMD *pCmd);
    VisionStatus WriteRpy(PR_FIT_LINE_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyMethod      = "Method";
    const String _strKeySrchWindow  = "SrchWindow";
    const String _strKeyErrorTol    = "ErrorTolerance";
    const String _strKeyThreshold   = "Threshold";
    const String _strKeyStatus      = "Status";
    const String _strKeySlope       = "Slope";
    const String _strKeyIntercept   = "Intercept";
    const String _strKeyPoint1      = "Point1";
    const String _strKeyPoint2      = "Point2";
};

class LogCaseFitParallelLine : public LogCase
{
public:
    explicit LogCaseFitParallelLine(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_FIT_PARALLEL_LINE_CMD *pCmd);
    VisionStatus WriteRpy(PR_FIT_PARALLEL_LINE_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyMethod = "Method";
    const String _strKeySrchWindow1 = "SrchWindow1";
    const String _strKeySrchWindow2 = "SrchWindow2";
    const String _strKeyErrorTol = "ErrorTolerance";
    const String _strKeyThreshold = "Threshold";
    const String _strKeyStatus = "Status";
    const String _strKeySlope = "Slope";
    const String _strKeyIntercept1 = "Intercept1";
    const String _strKeyIntercept2 = "Intercept2";
    const String _strKeyLineOnePoint1 = "LineOnePoint1";
    const String _strKeyLineOnePoint2 = "LineOnePoint2";
    const String _strKeyLineTwoPoint1 = "LineTwoPoint1";
    const String _strKeyLineTwoPoint2 = "LineTwoPoint2";
};

class LogCaseSrchFiducial : public LogCase
{
public:
    explicit LogCaseSrchFiducial(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_SRCH_FIDUCIAL_MARK_CMD *pCmd);
    VisionStatus WriteRpy(PR_SRCH_FIDUCIAL_MARK_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeySrchWindow = "SrchWindow";
    const String _strKeyType        = "Type";
    const String _strKeySize        = "Size";
    const String _strKeyMargin      = "Margin";
    const String _strKeyStatus      = "Status";
    const String _strKeyPos         = "Position";
};

}
}
#endif