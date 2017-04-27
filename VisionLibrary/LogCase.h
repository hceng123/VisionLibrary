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
    String _formatSize(const cv::Size &size);
    cv::Size _parseSize(const String &strSize);
    template<typename _tp>
    String _formatVector(const std::vector<_tp> &vecValue)  {
        String strValue;
        for ( const auto &value : vecValue )
            strValue += std::to_string(value) + ", ";
        strValue = strValue.substr(0, strValue.length() - 2 );
        return strValue;
    }
    String _generateLogCaseName(const String &strFolderPrefix);
    String      _strLogCasePath;
    const String _CMD_RPY_FILE_NAME = "cmdrpy.log";
    const String _CMD_SECTION       = "CMD";
    const String _RPY_SECTION       = "RPY";
    const String _IMAGE_NAME        = "image.png";
    const String _MASK_NAME         = "mask.png";
    const String _RESULT_IMAGE_NAME = "result.png";
    const String _DEFAULT_COORD     = "0, 0";
    const String _DEFAULT_RECT      = "0, 0, 0, 0";
    const String _DEFAULT_SIZE      = "0, 0";
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

class LogCaseLrnDevice : public LogCase
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
    const String _strKeyROI             = "ROI";
    const String _strKeyMethod          = "Method";
    const String _strKeyRmNoiseMethod   = "RmNoiseMethod";
    const String _strKeyPreprocessed    = "Preprocessed";
    const String _strKeyErrorTol        = "ErrorTolerance";
    const String _strKeyAutoThreshold   = "AutoThreshold";
    const String _strKeyThreshold       = "Threshold";
    const String _strKeyAttribute       = "Attribute";
    

    const String _strKeyStatus          = "Status";
    const String _strKeyResultCtr       = "ResultCtr";
    const String _strKeyRadius          = "Radius";
};

class LogCaseCircleRoundness : public LogCase
{
public:
    explicit LogCaseCircleRoundness(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_CIRCLE_ROUNDNESS_CMD *pCmd);
    VisionStatus WriteRpy(PR_CIRCLE_ROUNDNESS_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyROI             = "ROI";

    const String _strKeyStatus          = "Status";
    const String _strKeyResultCtr       = "ResultCtr";
    const String _strKeyRadius          = "Radius";
    const String _strKeyRoundness       = "Roundness";
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
    const String _strKeyMethod          = "Method";
    const String _strKeySrchWindow      = "SrchWindow";
    const String _strKeyErrorTol        = "ErrorTolerance";
    const String _strKeyPreprocessed    = "Preprocessed";
    const String _strKeyThreshold       = "Threshold";
    const String _strKeyAttribute       = "Attribute";

    const String _strKeyStatus          = "Status";
    const String _strKeyReversedFit     = "ReversedFit";
    const String _strKeySlope           = "Slope";
    const String _strKeyIntercept       = "Intercept";
    const String _strKeyPoint1          = "Point1";
    const String _strKeyPoint2          = "Point2";
};

class LogCaseDetectLine : public LogCase
{
public:
    explicit LogCaseDetectLine(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_DETECT_LINE_CMD *pCmd);
    VisionStatus WriteRpy(PR_DETECT_LINE_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyROI         = "ROI";
    const String _strKeyDir         = "Direction";

    const String _strKeyStatus      = "Status";
    const String _strKeyReversedFit = "ReversedFit";
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
    const String _strKeyMethod          = "Method";
    const String _strKeySrchWindow1     = "SrchWindow1";
    const String _strKeySrchWindow2     = "SrchWindow2";
    const String _strKeyErrorTol        = "ErrorTolerance";
    const String _strKeyThreshold       = "Threshold";
    const String _strKeyAttribute       = "Attribute";

    const String _strKeyStatus          = "Status";
    const String _strKeyReversedFit     = "ReversedFit";
    const String _strKeySlope           = "Slope";
    const String _strKeyIntercept1      = "Intercept1";
    const String _strKeyIntercept2      = "Intercept2";
    const String _strKeyLineOnePoint1   = "LineOnePoint1";
    const String _strKeyLineOnePoint2   = "LineOnePoint2";
    const String _strKeyLineTwoPoint1   = "LineTwoPoint1";
    const String _strKeyLineTwoPoint2   = "LineTwoPoint2";
};

class LogCaseFitRect : public LogCase
{
public:
    explicit LogCaseFitRect(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_FIT_RECT_CMD *pCmd);
    VisionStatus WriteRpy(PR_FIT_RECT_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyMethod          = "Method";
    const String _strKeySrchWindow1     = "SrchWindow1";
    const String _strKeySrchWindow2     = "SrchWindow2";
    const String _strKeySrchWindow3     = "SrchWindow3";
    const String _strKeySrchWindow4     = "SrchWindow4";
    const String _strKeyErrorTol        = "ErrorTolerance";
    const String _strKeyThreshold       = "Threshold";
    const String _strKeyAttribute       = "Attribute";

    const String _strKeyStatus          = "Status";
    const String _strKeyReversedFit1    = "ReversedFit1";
    const String _strKeyReversedFit2    = "ReversedFit2";
    const String _strKeySlope1          = "Slope1";    
    const String _strKeySlope2          = "Slope2";
    const String _strKeyIntercept1      = "Intercept1";
    const String _strKeyIntercept2      = "Intercept2";
    const String _strKeyIntercept3      = "Intercept3";
    const String _strKeyIntercept4      = "Intercept4";    
    const String _strKeyLineOnePoint1   = "LineOnePoint1";
    const String _strKeyLineOnePoint2   = "LineOnePoint2";
    const String _strKeyLineTwoPoint1   = "LineTwoPoint1";
    const String _strKeyLineTwoPoint2   = "LineTwoPoint2";
    const String _strKeyLineThreePoint1 = "LineThreePoint1";
    const String _strKeyLineThreePoint2 = "LineThreePoint2";
    const String _strKeyLineFourPoint1  = "LineFourPoint1";
    const String _strKeyLineFourPoint2  = "LineFourPoint2";
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
    const String _strKeySrchWindow  = "SrchWindow";
    const String _strKeyType        = "Type";
    const String _strKeySize        = "Size";
    const String _strKeyMargin      = "Margin";
    const String _strKeyStatus      = "Status";
    const String _strKeyPos         = "Position";
};

class LogCaseOcr : public LogCase
{
public:
    explicit LogCaseOcr(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_OCR_CMD *pCmd);
    VisionStatus WriteRpy(PR_OCR_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyDirection   = "Direction";
    const String _strKeySrchWindow  = "SrchWindow";    
    const String _strKeyStatus      = "Status";
    const String _strKeyResultStr   = "ResultStr";
};

class LogCaseRemoveCC : public LogCase
{
public:
    explicit LogCaseRemoveCC(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_REMOVE_CC_CMD *pCmd);
    VisionStatus WriteRpy(PR_REMOVE_CC_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyROI             = "ROI";
    const String _strKeyConnectivity    = "Connectivity";
    const String _strKeyCompareType     = "CompareType";
    const String _strKeyAreaThreshold   = "AreaThreshold";

    const String _strKeyStatus          = "Status";
    const String _strKeyTotalCC         = "TotalCC";
    const String _strKeyRemovedCC       = "RemovedCC";
};

class LogCaseDetectEdge : public LogCase
{
public:
    explicit LogCaseDetectEdge(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_DETECT_EDGE_CMD *pCmd);
    VisionStatus WriteRpy(PR_DETECT_EDGE_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyROI             = "ROI";
    const String _strKeyThreshold1      = "Threshold1";
    const String _strKeyThreshold2      = "Threshold2";
    const String _strKeyApertureSize    = "ApertureSize";

    const String _strKeyStatus          = "Status";
};

class LogCaseAutoThreshold : public LogCase
{
public:
    explicit LogCaseAutoThreshold(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_AUTO_THRESHOLD_CMD *pCmd);
    VisionStatus WriteRpy(PR_AUTO_THRESHOLD_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyROI             = "ROI";
    const String _strKeyThresholdNum    = "ThresholdNum";

    const String _strKeyStatus          = "Status";
    const String _strKeyThreshold       = "Threshold";
};

class LogCaseFillHole : public LogCase
{
public:
    explicit LogCaseFillHole(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_FILL_HOLE_CMD *pCmd);
    VisionStatus WriteRpy(PR_FILL_HOLE_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyROI             = "ROI";
    const String _strKeyAttribute       = "Attribute";
    const String _strKeyMethod          = "Method";
    const String _strKeyMorphShape      = "MorphShape";
    const String _strKeyMorphKernelX    = "MorphKernelX";
    const String _strKeyMorphKernelY    = "MorphKernelY";
    const String _strKeyMorphIteration  = "MorphIteration";

    const String _strKeyStatus          = "Status";
};

class LogCaseMatchTmpl : public LogCase
{
public:
    explicit LogCaseMatchTmpl(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_MATCH_TEMPLATE_CMD *pCmd);
    VisionStatus WriteRpy(PR_MATCH_TEMPLATE_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _TMPL_FILE_NAME        = "Template.png";
    const String _strKeySrchWindow      = "SrchWindow";
    const String _strKeyMotion          = "Motion";

    const String _strKeyStatus          = "Status";
};

class LogCasePickColor : public LogCase
{
public:
    explicit LogCasePickColor(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_PICK_COLOR_CMD *pCmd);
    VisionStatus WriteRpy(PR_PICK_COLOR_RPY *pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyROI             = "ROI";
    const String _strKeyPickPoint       = "PickPoint";
    const String _strKeyColorDiff       = "ColorDiff";
    const String _strKeyGrayDiff        = "GrayDiff";

    const String _strKeyStatus          = "Status";
};

class LogCaseCalibrateCamera : public LogCase
{
public:
    explicit LogCaseCalibrateCamera(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_CALIBRATE_CAMERA_CMD *const pCmd);
    VisionStatus WriteRpy(const PR_CALIBRATE_CAMERA_RPY *const pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyBoardPattern    = "BoardPattern";
    const String _strKeyObjectSize      = "ObjectSize";

    const String _strKeyStatus          = "Status";
    const String _strKeyResolutionX     = "ResolutionX";
    const String _strKeyResolutionY     = "ResolutionY";
};

class LogCaseRestoreImg : public LogCase
{
public:
    explicit LogCaseRestoreImg(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_RESTORE_IMG_CMD *const pCmd);
    VisionStatus WriteRpy(PR_RESTORE_IMG_RPY *const pRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strYmlFile            = "InputMat.yml";
    const String _strKeyMapX            = "MapX";
    const String _strKeyMapY            = "MapY";

    const String _strKeyStatus          = "Status";
    const String _strDiffImg            = "diff.png";
};

}
}
#endif