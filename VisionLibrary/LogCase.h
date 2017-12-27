#ifndef _LOG_CASE_H_
#define _LOG_CASE_H_

#include "BaseType.h"
#include "VisionHeader.h"
#include <memory>

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
    static String unzip(const String &strFilePath);
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
    void _zip();
    String      _strLogCasePath;
    const String _CMD_RPY_FILE_NAME = "cmdrpy.log";
    const String _CMD_SECTION       = "CMD";
    const String _RPY_SECTION       = "RPY";
    const String _IMAGE_NAME        = "image.png";
    const String _MASK_NAME         = "mask.png";
    const String _RESULT_IMAGE_NAME = "result.png";
    const String _IMAGE_FILE_PREFIX = "image_";
    const String _IMAGE_FILE_EXT    = ".png";
    const String _DEFAULT_COORD     = "0, 0";
    const String _DEFAULT_RECT      = "0, 0, 0, 0";
    const String _DEFAULT_SIZE      = "0, 0";
    const String _EXTENSION         = ".logcase";
    bool         _bReplay;
};
using LogCasePtr = std::unique_ptr<LogCase>;

class LogCaseLrnObj : public LogCase
{
public:
    explicit LogCaseLrnObj(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_LRN_OBJ_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_LRN_OBJ_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:    
    const String _strKeyAlgorithm   = "Algorithm";
    const String _strKeyLrnWindow   = "LrnWindow";

    const String _strKeyStatus      = "Status";
    const String _strKeyCenterPos   = "Center";
    const String _strKeyRecordId    = "RecordId"; 
};

class LogCaseSrchObj : public LogCase
{
public:
    explicit LogCaseSrchObj(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_SRCH_OBJ_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_SRCH_OBJ_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:    
    const String _strKeyAlgorithm   = "Algorithm";
    const String _strKeySrchWindow  = "SrchWindow";
    const String _strKeyRecordId    = "RecordId";
    const String _strKeyExpectedPos = "ExpectedPos";

    const String _strKeyStatus      = "Status";
    const String _strKeyObjPos      = "ObjPos";
    const String _strKeyRotation    = "Rotation";
    const String _strKeyOffset      = "Offset";
    const String _strKeyMatchScore  = "MatchScore";
};

class LogCaseLrnDevice : public LogCase
{
public:
    explicit LogCaseLrnDevice(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_LRN_DEVICE_CMD *pLrnTmplCmd);
    VisionStatus WriteRpy(PR_LRN_DEVICE_RPY *pLrnTmplRpy);
    virtual VisionStatus RunLogCase() override;
private:
};

class LogCaseFitCircle : public LogCase
{
public:
    explicit LogCaseFitCircle(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_FIT_CIRCLE_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_FIT_CIRCLE_RPY *const pstRpy);
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

class LogCaseFindCircle : public LogCase
{
public:
    explicit LogCaseFindCircle(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_FIND_CIRCLE_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_FIND_CIRCLE_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:    
    const String _strKeyInnerAttribute  = "InnerAttribute";
    const String _strKeyExpCircleCtr    = "ExpectedCircleCtr";
    const String _strKeyFindCirclePair  = "FindCirclePair";
    const String _strKeyMinSrchRadius   = "MinSrchRadius";
    const String _strKeyMaxSrchRadius   = "MaxSrchRadius";
    const String _strKeyStartSrchAngle  = "StartSrchAngle";
    const String _strKeyEndSrchAngle    = "EndSrchAngle";
    const String _strKeyRmNoiseMethod   = "RmNoiseMethod";
    const String _strKeyErrorTol        = "ErrorTolerance";
    const String _strKeyCaliperCount    = "CaliperCount";
    const String _strKeyCaliperWidth    = "CaliperWidth";
    const String _strKeyDiffFilterHalfW = "DiffFilterHalfW";
    const String _strKeyDiffFilterSigma = "DiffFilterSigma";
    const String _strKeyEdgeThreshold   = "EdgeThreshold";
    const String _strKeySelectEdge      = "SelectEdge";
    const String _strKeyRmStrayPtRatio  = "RmStrayPtRatio";

    const String _strKeyStatus          = "Status";
    const String _strKeyResultCtr       = "ResultCtr";
    const String _strKeyRadius          = "Radius";
    const String _strKeyRadius2         = "Radius2";
};

class LogCaseInspCircle : public LogCase
{
public:
    explicit LogCaseInspCircle(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_INSP_CIRCLE_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_INSP_CIRCLE_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyROI             = "ROI";

    const String _strKeyStatus          = "Status";
    const String _strKeyResultCtr       = "ResultCtr";
    const String _strKeyDiameter        = "Diameter";
    const String _strKeyRoundness       = "Roundness";
};

class LogCaseFitLine : public LogCase
{
public:
    explicit LogCaseFitLine(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_FIT_LINE_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_FIT_LINE_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyMethod          = "Method";
    const String _strKeyROI             = "ROI";
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

class LogCaseFindLine : public LogCase
{
public:
    explicit LogCaseFindLine(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_FIND_LINE_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_FIND_LINE_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyRoiCenter       = "RoiCenter";
    const String _strKeyRoiSize         = "RoiSize";
    const String _strKeyRoiAngle        = "RoiAngle";
    const String _strKeyAlgorithm       = "Algorithm";
    const String _strKeyCaliperCount    = "CaliperCount";
    const String _strKeyCaliperWidth    = "CaliperWidth";
    const String _strKeyDir             = "Direction";
    const String _strKeyCheckLinerity   = "CheckLinerity";
    const String _strKeyPointMaxOffset  = "PointMaxOffset";
    const String _strKeyMinLinerity     = "MinLinerity";
    const String _strKeyCheckAngle      = "CheckAngle";
    const String _strKeyExpectedAngle   = "ExpectedAngle";
    const String _strKeyAngleDiffTol    = "AngleDiffTol";

    const String _strKeyStatus          = "Status";
    const String _strKeyReversedFit     = "ReversedFit";
    const String _strKeySlope           = "Slope";
    const String _strKeyIntercept       = "Intercept";
    const String _strKeyPoint1          = "Point1";
    const String _strKeyPoint2          = "Point2";
    const String _strKeyLinerityPass    = "LinerityPass";
    const String _strKeyLinerity        = "Linerity";
    const String _strKeyAngleCheckPass  = "AngleCheckPass";
    const String _strKeyAngle           = "Angle";
};

class LogCaseFitParallelLine : public LogCase
{
public:
    explicit LogCaseFitParallelLine(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_FIT_PARALLEL_LINE_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_FIT_PARALLEL_LINE_RPY *const pstRpy);
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
    VisionStatus WriteCmd(const PR_FIT_RECT_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_FIT_RECT_RPY *const pstRpy);
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

class LogCaseFindEdge : public LogCase
{
public:
    explicit LogCaseFindEdge(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_FIND_EDGE_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_FIND_EDGE_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyROI             = "ROI";
    const String _strKeyAutoThreshold   = "AutoThreshold";
    const String _strKeyThreshold       = "Threshold";
    const String _strKeyDirection       = "Direction";
    const String _strKeyMinLength       = "MinLength";

    const String _strKeyStatus          = "Status";
    const String _strKeyEdgeCount       = "EdgeCount";
};

class LogCaseSrchFiducial : public LogCase
{
public:
    explicit LogCaseSrchFiducial(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(PR_SRCH_FIDUCIAL_MARK_CMD *pstCmd);
    VisionStatus WriteRpy(PR_SRCH_FIDUCIAL_MARK_RPY *pstRpy);
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
    VisionStatus WriteCmd(PR_OCR_CMD *pstCmd);
    VisionStatus WriteRpy(PR_OCR_RPY *pstRpy);
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
    VisionStatus WriteCmd(PR_REMOVE_CC_CMD *pstCmd);
    VisionStatus WriteRpy(PR_REMOVE_CC_RPY *pstRpy);
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
    VisionStatus WriteCmd(PR_DETECT_EDGE_CMD *pstCmd);
    VisionStatus WriteRpy(PR_DETECT_EDGE_RPY *pstRpy);
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
    VisionStatus WriteCmd(PR_AUTO_THRESHOLD_CMD *pstCmd);
    VisionStatus WriteRpy(PR_AUTO_THRESHOLD_RPY *pstRpy);
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
    VisionStatus WriteCmd(PR_FILL_HOLE_CMD *pstCmd);
    VisionStatus WriteRpy(PR_FILL_HOLE_RPY *pstRpy);
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

class LogCaseLrnTmpl : public LogCase
{
public:
    explicit LogCaseLrnTmpl(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_LRN_TEMPLATE_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_LRN_TEMPLATE_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyROI             = "ROI";
    const String _strKeyAlgorithm       = "Algorithm";

    const String _strKeyStatus          = "Status";
    const String _strKeyRecordId        = "RecordId";
    const String _strTmplFileName       = "Tmpl.png";
};

class LogCaseMatchTmpl : public LogCase
{
public:
    explicit LogCaseMatchTmpl(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_MATCH_TEMPLATE_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_MATCH_TEMPLATE_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyRecordId        = "RecordId";
    const String _strKeySrchWindow      = "SrchWindow";
    const String _strKeyMotion          = "Motion";
    const String _strKeyAlgorithm       = "Algorithm";

    const String _strKeyStatus          = "Status";
    const String _strKeyObjPos          = "ObjPos";
    const String _strKeyRotation        = "Rotation";
    const String _strKeyMatchScore      = "MatchScore";
};

class LogCasePickColor : public LogCase
{
public:
    explicit LogCasePickColor(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_PICK_COLOR_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_PICK_COLOR_RPY *const pstRpy);
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
    VisionStatus WriteCmd(const PR_CALIBRATE_CAMERA_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_CALIBRATE_CAMERA_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strCornerPointsImage  = "CorerPoints.png";

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
    VisionStatus WriteCmd(const PR_RESTORE_IMG_CMD *const pstCmd);
    VisionStatus WriteRpy(PR_RESTORE_IMG_RPY *const pstRpy);
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

class LogCaseAutoLocateLead : public LogCase
{
public:
    explicit LogCaseAutoLocateLead(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_AUTO_LOCATE_LEAD_CMD *const pstCmd);
    VisionStatus WriteRpy(PR_AUTO_LOCATE_LEAD_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeySrchWindow      = "SrchWindow";
    const String _strKeyChipWindow      = "ChipWindow";

    const String _strKeyStatus          = "Status";
    const String _strLeadLocation       = "LeadLocation_";  //End with _ because it will concatenate with lead number.
};

class LogCaseInspBridge : public LogCase
{
public:
    explicit LogCaseInspBridge(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_INSP_BRIDGE_CMD *const pstCmd);
    VisionStatus WriteRpy(PR_INSP_BRIDGE_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyInspItemCount   = "InspItemCount";
    const String _strKeyMode            = "Mode";
    const String _strKeyInnerWindow     = "InnerWindow";
    const String _strKeyOuterWindow     = "OuterWindow";    
    const String _strKeyDirection       = "Direction";
    const String _strKeyMaxLengthX      = "MaxLengthX";
    const String _strKeyMaxLengthY      = "MaxLengthY";

    const String _strKeyStatus          = "Status";
    const String _strKeyWithBridge      = "WithBridge";
    const String _strKeyBridgeWindow    = "BridgeWindow";
};

class LogCaseInspChip : public LogCase
{
public:
    explicit LogCaseInspChip(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_INSP_CHIP_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_INSP_CHIP_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeySrchWindow      = "SrchWindow";
    const String _strKeyInspMode        = "InspMode";
    const String _strKeyRecordId        = "RecordId";

    const String _strKeyStatus          = "Status";
    const String _strKeyCenter          = "Center";
    const String _strKeySize            = "Size";
    const String _strKeyAngle           = "Angle";
};

class LogCaseLrnContour : public LogCase
{
public:
    explicit LogCaseLrnContour(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_LRN_CONTOUR_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_LRN_CONTOUR_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyROI             = "ROI";
    const String _strKeyAutoThreshold   = "AutoThreshold";
    const String _strKeyThreshold       = "Threshold";

    const String _strKeyStatus          = "Status";
    const String _strKeyRecordId        = "RecordId";
};

class LogCaseInspContour : public LogCase
{
public:
    explicit LogCaseInspContour(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_INSP_CONTOUR_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_INSP_CONTOUR_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyROI             = "ROI";
    const String _strKeyRecordId        = "RecordId";
    const String _strKeyDefectThreshold = "DefectThreshold";
    const String _strKeyMinDefectArea   = "MinThresholdArea";
    const String _strKeyInnerLengthTol  = "InnerLengthTol";
    const String _strKeyOuterLengthTol  = "OuterLengthTol";
    const String _strKeyInnerMaskDepth  = "InnerMaskDepth";
    const String _strKeyOuterMaskDepth  = "OuterMaskDepth";

    const String _strKeyStatus          = "Status";
};

class LogCaseInspHole : public LogCase
{
public:
    explicit LogCaseInspHole(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_INSP_HOLE_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_INSP_HOLE_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyROI             = "ROI";
    const String _strKeyRecordId        = "RecordId";
    const String _strKeySegmentMethod   = "SegmentMethod";
    const String _strKeyInspMode        = "InspMode";
    const String _strKeyGrayRangeStart  = "GrayRangeStart";
    const String _strKeyGrayRangeEnd    = "GrayRangeEnd";
    const String _strKeyBlueRangeStart  = "BlueRangeStart";
    const String _strKeyBlueRangeEnd    = "BlueRangeEnd";
    const String _strKeyGreenRangeStart = "GreenRangeStart";
    const String _strKeyGreenRangeEnd   = "GreenRangeEnd";
    const String _strKeyRedRangeStart   = "RedRangeStart";
    const String _strKeyRedRangeEnd     = "RedRangeEnd";
    const String _strKeyMaxRatio        = "MaxRatio";
    const String _strKeyMinRatio        = "MinRatio";
    const String _strKeyMaxArea         = "MaxArea";
    const String _strKeyMinArea         = "MinArea";
    const String _strKeyMaxBlobCount    = "MaxBlobCount";
    const String _strKeyMinBlobCount    = "MinBlobCount";
    const String _strKeyAdvBlobCriteria = "EnableAdvBlobCriteria";
    const String _strKeyMaxLWRatio      = "MaxLengthWidthRatio";
    const String _strKeyMinLWRatio      = "MinLengthWidthRatio";
    const String _strKeyMaxCircularity  = "MaxCircularity";
    const String _strKeyMinCircularity  = "MinCircularity";

    const String _strKeyStatus          = "Status";
    const String _strKeyRatio           = "Ratio";
    const String _strKeyBlobCount       = "BlobCount";
};

class LogCaseInspLead : public LogCase
{
public:
    explicit LogCaseInspLead(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_INSP_LEAD_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_INSP_LEAD_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyChipCenter          = "RoiCenter";
    const String _strKeyChipSize            = "RoiSize";
    const String _strKeyChipAngle           = "RoiAngle";
    const String _strKeyLeadCount           = "LeadCount";
    const String _strKeyLeadSrchWinCtr      = "LeadSrchWinCenter";
    const String _strKeyLeadSrchWinSize     = "LeadSrchWinSize";
    const String _strKeyLeadSrchWinAngle    = "LeadSrchWinAngle";
    const String _strKeyLeadExpWinCtr       = "LeadExpWinCenter";
    const String _strKeyLeadExpWinSize      = "LeadExpWinSize";
    const String _strKeyLeadExpWinAngle     = "LeadExpWinAngle";
    const String _strKeyLeadStartWidthRatio = "LeadStartWidthRatio";
    const String _strKeyLeadStartConLen     = "LeadStartConsecutiveLength";
    const String _strKeyLeadEndWidthRatio   = "LeadEndWidthRatio";
    const String _strKeyLeadEndConLen       = "LeadEndConsecutiveLength";
    const String _strKeyFindLeadEndMethod   = "FindLeadEndMethod";

    const String _strKeyStatus              = "Status";
    const String _strFound                  = "Found";
    const String _strKeyLeadWinCtr          = "LeadWinCenter";
    const String _strKeyLeadWinSize         = "LeadWinSize";
    const String _strKeyLeadWinAngle        = "LeadWinAngle";
};

class LogCaseCalib3DBase : public LogCase
{
public:
    explicit LogCaseCalib3DBase(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_CALIB_3D_BASE_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_CALIB_3D_BASE_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strResultYmlFileName      = "Result.yml";

    const String _strKeyEnableGF            = "EnableGaussianFilter";
    const String _strKeyReverseSeq          = "ReverseSeq";
    const String _strKeyRemoveHarmonicWaveK = "RemoveHarmonicWaveK";

    const String _strKeyStatus              = "Status";
    const String _strKeyThkToThinK          = "ThickToThinK";
    const String _strKeyThkToThinnestK      = "ThickToThinnestK";
};

class LogCaseCalib3DHeight : public LogCase
{
public:
    explicit LogCaseCalib3DHeight(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_CALIB_3D_HEIGHT_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_CALIB_3D_HEIGHT_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strInputYmlFileName       = "Input.yml";
    const String _strResultYmlFileName      = "Result.yml";

    const String _strKeyEnableGF            = "EnableGaussianFilter";
    const String _strKeyReverseSeq          = "ReverseSeq";
    const String _strKeyReverseHeight       = "ReverseHeight";
    const String _strKeyUseThinnestPattern  = "UseThinnestPattern";
    const String _strKeyRemoveHarmonicWaveK = "RemoveHarmonicWaveK";
    const String _strKeyMinIntensityDiff    = "MinIntensityDiff";
    const String _strKeyMinAmplitude        = "MinAmplitude";
    const String _strKeyThickToThinK        = "ThickToThinK";
    const String _strKeyThickToThinnestK    = "ThickToThinnestK";
    const String _strKeyBlockStepCount      = "BlockStepCount";
    const String _strKeyBlockStepHeight     = "BlockStepHeight";
    const String _strKeyResultImgGridRow    = "ResultImgGridRow";
    const String _strKeyResultImgGridCol    = "ResultImgGridCol";

    const String _strKeyStatus              = "Status";
};

class LogCaseCalc3DHeight : public LogCase
{
public:
    explicit LogCaseCalc3DHeight(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_CALC_3D_HEIGHT_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_CALC_3D_HEIGHT_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strInputYmlFileName       = "Input.yml";

    const String _strKeyEnableGF            = "EnableGaussianFilter";
    const String _strKeyReverseSeq          = "ReverseSeq";
    const String _strKeyUseThinnestPattern  = "UseThinnestPattern";
    const String _strKeyRemoveHarmonicWaveK = "RemoveHarmonicWaveK";
    const String _strKeyMinAmplitude        = "MinAmplitude";
    const String _strKeyThickToThinK        = "ThickToThinK";
    const String _strKeyThickToThinnestK    = "ThickToThinnestK";
    const String _strKeyRmBetaJumpSpanX     = "RemoveBetaJumpSpanX";
    const String _strKeyRmBetaJumpSpanY     = "RemoveBetaJumpSpanY";
    const String _strKeyRmGammaJumpSpanX    = "RemoveGammaJumpSpanX";
    const String _strKeyRmGammaJumpSpanY    = "RemoveGammaJumpSpanY";

    const String _strKeyStatus              = "Status";
};

class LogCaseCalcCameraMTF : public LogCase
{
public:
    explicit LogCaseCalcCameraMTF(const String &strPath, bool bReplay = false) : LogCase(strPath, bReplay) {}
    VisionStatus WriteCmd(const PR_CALC_CAMERA_MTF_CMD *const pstCmd);
    VisionStatus WriteRpy(const PR_CALC_CAMERA_MTF_RPY *const pstRpy);
    virtual VisionStatus RunLogCase() override;
    virtual String GetFolderPrefix()    const { return StaticGetFolderPrefix(); }
    static String StaticGetFolderPrefix();
private:
    const String _strKeyVBigPatternROI      = "VBigPatternROI";
    const String _strKeyHBigPatternROI      = "HBigPatternROI";
    const String _strKeyVSmallPatternROI    = "VSmallPatternROI";
    const String _strKeyHSmallPatternROI    = "HSmallPatternROI";

    const String _strKeyStatus              = "Status";
    const String _strKeyBigPatternAbsMtfV   = "BigPatternAbsMtfV";
    const String _strKeyBigPatternAbsMtfH   = "BigPatternAbsMtfH";
    const String _strKeySmallPatternAbsMtfV = "SmallPatternAbsMtfV";
    const String _strKeySmallPatternAbsMtfH = "SmallPatternAbsMtfH";
    const String _strKeySmallPatternRelMtfV = "SmallPatternRelMtfV";
    const String _strKeySmallPatternRelMtfH = "SmallPatternRelMtfH";
};

}
}
#endif