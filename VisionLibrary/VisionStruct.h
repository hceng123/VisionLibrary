#ifndef _AOI_VISION_STRUCT_H_
#define _AOI_VISION_STRUCT_H_

#include "VisionType.h"
#include "VisionStatus.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc.hpp"
#include "BaseType.h"
#include <list>

namespace AOI
{
namespace Vision
{

using VectorOfDMatch = std::vector<cv::DMatch>;
using VectorOfVectorOfDMatch = std::vector<VectorOfDMatch>;
using VectorOfKeyPoint =  std::vector<cv::KeyPoint>;
using VectorOfVectorKeyPoint = std::vector<VectorOfKeyPoint>;
using VectorOfPoint = std::vector<cv::Point>;
using VectorOfVectorOfPoint = std::vector<VectorOfPoint>;
using VectorOfPoint2f = std::vector<cv::Point2f>;
using VectorOfPoint3f = std::vector<cv::Point3f>;
using VectorOfVectorOfPoint2f = std::vector<VectorOfPoint2f>;
using VectorOfRect = std::vector<cv::Rect>;
using ListOfPoint = std::list<cv::Point>;
using ListOfPoint2f = std::list<cv::Point2f>;
using VectorOfListOfPoint = std::vector<ListOfPoint>;
using VectorOfSize2f = std::vector<cv::Size2f>;
using VectorOfMat = std::vector<cv::Mat>;
using VectorOfVectorOfMat = std::vector<VectorOfMat>;
using VectorOfGpuMat = std::vector<cv::cuda::GpuMat>;
using VectorOfFloat = std::vector<float>;
using VectorOfVectorOfFloat = std::vector<VectorOfFloat>;
using VectorOfDouble = std::vector<double>;
using VectorOfDirection = std::vector<PR_DIRECTION>;
using VectorOfInt = std::vector<int>;
using VectorOfVectorOfInt = std::vector<VectorOfInt>;
using VectorOfRtRect = std::vector<cv::RotatedRect>;

template <typename Tp> inline Int32 ToInt32(Tp param) { return static_cast<Int32>(param); }
template <typename Tp> inline Int16 ToInt16(Tp param) { return static_cast<Int16>(param); }
template <typename Tp> inline float ToFloat(Tp param) { return static_cast<float>(param); }

struct PR_VERSION_INFO {
    char                    chArrVersion[100];
};

struct PR_GET_ERROR_INFO_RPY {
    PR_STATUS_ERROR_LEVEL    enErrorLevel;
    char                    achErrorStr[PR_MAX_ERR_STR_LEN];
};

struct PR_GET_RECORD_INFO_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matImage;
};

struct PR_LRN_OBJ_CMD {
    cv::Mat                 matInputImg;
    PR_SRCH_OBJ_ALGORITHM   enAlgorithm;
    cv::Mat                 matMask;
    cv::Rect2f              rectLrn;
};

struct PR_LRN_OBJ_RPY {
    VisionStatus                enStatus;
    std::vector<cv::KeyPoint>   vecKeyPoint;
    cv::Mat                     matDescritor;
    cv::Mat                     matTmpl;
    cv::Point2f                 ptCenter;
    Int32                       nRecordId;
    cv::Mat                     matResultImg;
};

struct PR_SRCH_OBJ_CMD {
    PR_SRCH_OBJ_ALGORITHM  enAlgorithm;
    cv::Mat                matInputImg;
    cv::Rect               rectSrchWindow;
    cv::Point2f            ptExpectedPos;
    Int32                  nRecordId;
};

struct PR_SRCH_OBJ_RPY {
    VisionStatus            enStatus;
    float                   fMatchScore;
    cv::Mat                 matHomography;
    cv::Point2f             ptObjPos;
    cv::Size2f              szOffset;
    float                   fRotation;
    cv::Mat                 matResultImg;
};

/******************************************
* Match Template Section *
******************************************/
struct PR_LRN_TEMPLATE_CMD {
    PR_LRN_TEMPLATE_CMD() : enAlgorithm (PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF) {}
    cv::Mat                 matInputImg;
    cv::Mat                 matMask;
    cv::Rect                rectROI;
    PR_MATCH_TMPL_ALGORITHM enAlgorithm;
};

struct PR_LRN_TEMPLATE_RPY {
    VisionStatus            enStatus;
    Int32                   nRecordId;
    cv::Mat                 matTmpl;
};

struct PR_MATCH_TEMPLATE_CMD {
    PR_MATCH_TEMPLATE_CMD() :
        enAlgorithm         (PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF),
        nRecordId           (-1),
        bSubPixelRefine     (false),
        enMotion            (PR_OBJECT_MOTION::TRANSLATION),
        fMinMatchScore      (60) {}
    cv::Mat                 matInputImg;
    PR_MATCH_TMPL_ALGORITHM enAlgorithm;
    Int32                   nRecordId;
    cv::Rect                rectSrchWindow;
    bool                    bSubPixelRefine;
    PR_OBJECT_MOTION        enMotion;
    float                   fMinMatchScore;
};

struct PR_MATCH_TEMPLATE_RPY {
    VisionStatus            enStatus;
    cv::Point2f             ptObjPos;
    float                   fRotation;
    float                   fMatchScore;
    cv::Mat                 matResultImg;
};
/******************************************
* End of Match Template Section *
******************************************/

struct PR_DefectCriteria {
    PR_DEFECT_ATTRIBUTE     enAttribute;
    Byte                    ubContrast;
    float                   fArea;    // For general defect.
    float                   fLength;// For line defect only. If != 0, will regard as line defect detecton and rAreaThreshold will be ignored
    float                   fWidth;    // For noise removal. Note: unit is in 8192 coord sys (in old convention is phy coord sys, which is no good)
};

struct PR_INSP_SURFACE_CMD {
    cv::Mat                 matInsp;
    cv::Mat                 matTmpl;
    cv::Mat                 matMask;
    cv::Rect2f              rectLrn;
    cv::Point2f             ptObjPos;
    float                   fRotation;
    UInt16                  u16NumOfDefectCriteria;
    PR_DefectCriteria       astDefectCriteria[MAX_NUM_OF_DEFECT_CRITERIA];
};

struct PR_INSP_SURFACE_RPY {
    struct PR_SURFACE_DEFECT {
        PR_DEFECT_TYPE      enType;
        float               fArea;
        float               fRadius;
        float               fLength;
        Int16               n16Constrast;
    };
    Int16                   nStatus;
    Int16                   n16NDefect;
    PR_SURFACE_DEFECT       astDefect[MAX_NUM_OF_DEFECT_RESULT];
};

template<typename _Tp> class PR_Line_
{
public:
    PR_Line_ ()
    {
        pt1.x = 0;
        pt1.y = 0;
        pt2.x = 0;
        pt2.y = 0;
    }
    PR_Line_ ( cv::Point_<_Tp> inPt1, cv::Point_<_Tp> inPt2 ) : pt1 ( inPt1 ), pt2 ( inPt2 )
    {
    }
    cv::Point_<_Tp> pt1;
    cv::Point_<_Tp> pt2;
};

using PR_Line = PR_Line_<int>;
using PR_Line2f = PR_Line_<float>;

/******************************************
* Device Inspection Section *
******************************************/
#define PR_MAX_DEVICE_COUNT         (100)
#define PR_MAX_CRITERIA_COUNT       (5)

struct PR_INSP_DEVICE_ITEM {
    bool        bCheckMissed;           //check if the device is missed
    bool        bCheckShift;            //check if the device shift exceed tolerance
    bool        bCheckRotation;         //check if rotation exceed tolerance
    bool        bCheckScale;
};

struct PR_SINGLE_DEVICE_INFO {
    cv::Point2f             stCtrPos;
    cv::Size2f              stSize;
    cv::Rect2f              rectSrchWindow;
    PR_INSP_DEVICE_ITEM     stInspItem;
    Int16                   nCriteriaNo;
};

struct PR_INSP_DEVICE_CRITERIA {
    float                   fMaxOffsetX;
    float                   fMaxOffsetY;
    float                   fMaxRotate;
    float                   fMaxScale;
    float                   fMinScale;
};

struct PR_LRN_DEVICE_CMD {
    cv::Mat                 matInputImg;
    cv::Rect2f              rectDevice;
    bool                    bAutoThreshold;
    Int16                   nElectrodeThreshold;
};

struct PR_LRN_DEVICE_RPY {
    Int32                   nStatus;
    Int32                   nRecordId;
    cv::Size2f              sizeDevice;
    Int16                   nElectrodeThreshold;
};

struct PR_INSP_DEVICE_CMD {
    cv::Mat                 matInputImg;
    Int32                   nRecordId;
    Int32                   nDeviceCount;
    Int16                   nElectrodeThreshold;
    PR_SINGLE_DEVICE_INFO   astDeviceInfo[PR_MAX_DEVICE_COUNT];
    PR_INSP_DEVICE_CRITERIA astCriteria[PR_MAX_CRITERIA_COUNT];
};

struct PR_DEVICE_INSP_RESULT {
    Int32                   nStatus;
    cv::Point2f             ptPos;
    float                   fRotation;
    float                   fOffsetX;
    float                   fOffsetY;
    float                   fScale;
};

struct PR_INSP_DEVICE_RPY {
    Int32                   nStatus;
    Int32                   nDeviceCount;
    PR_DEVICE_INSP_RESULT   astDeviceResult[PR_MAX_DEVICE_COUNT];
};
/******************************************
* End of Device Inspection Section
******************************************/

struct PR_SRCH_FIDUCIAL_MARK_CMD {
    PR_SRCH_FIDUCIAL_MARK_CMD() :
        enType          (PR_FIDUCIAL_MARK_TYPE::SQUARE),
        fSize           (0.f),
        fMargin         (10.f),
        fMinMatchScore  (60.f) {}
    cv::Mat                 matInputImg;
    cv::Rect                rectSrchWindow;
    PR_FIDUCIAL_MARK_TYPE   enType;
    float                   fSize;          //The white part size of the fiducial mark.
    float                   fMargin;        //The dark part width outside of the fiducial mark.
    float                   fMinMatchScore;
};

struct PR_SRCH_FIDUCIAL_MARK_RPY {
    VisionStatus            enStatus;
    cv::Point2f             ptPos;
    float                   fMatchScore;
    cv::Mat                 matResultImg;
};

//Fit line is for accurately fit line in the preprocessed image, the line should be obvious compared to the background.
struct PR_FIT_LINE_CMD {
    PR_FIT_LINE_CMD() : 
        enMethod        ( PR_FIT_METHOD::LEAST_SQUARE_REFINE ),
        bPreprocessed   ( false ),
        nMaxRansacTime  (20) {}
    cv::Mat                 matInputImg;
    cv::Mat                 matMask;
    cv::Rect                rectROI;
    PR_FIT_METHOD           enMethod;
    bool                    bPreprocessed;
    Int32                   nThreshold;
    PR_OBJECT_ATTRIBUTE     enAttribute;
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    float                   fErrTol;
    int                     nMaxRansacTime;             //Only used when fitting method is ransac.
    int                     nNumOfPtToFinishRansac;     //Only used when fitting method is ransac.
};

struct PR_FIT_LINE_RPY {
    VisionStatus            enStatus;
    bool                    bReversedFit;   //If it is true, then the result is x = fSlope * y + fIntercept. Otherwise the line is y = fSlope * x + fIntercept.
    float                   fSlope;
    float                   fIntercept;
    PR_Line2f               stLine;
    cv::Mat                 matResultImg;
};

struct PR_FIT_LINE_BY_POINT_CMD {
    PR_FIT_LINE_BY_POINT_CMD() : 
        enMethod        ( PR_FIT_METHOD::LEAST_SQUARE_REFINE ),
        bPreprocessed   ( false ),
        nMaxRansacTime  (20) {}
    VectorOfPoint2f         vecPoints;
    PR_FIT_METHOD           enMethod;
    bool                    bPreprocessed;
    Int32                   nThreshold;
    PR_OBJECT_ATTRIBUTE     enAttribute;
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    float                   fErrTol;
    int                     nMaxRansacTime;             //Only used when fitting method is ransac.
    int                     nNumOfPtToFinishRansac;     //Only used when fitting method is ransac.
};

struct PR_FIT_LINE_BY_POINT_RPY {
    VisionStatus            enStatus;
    bool                    bReversedFit;   //If it is true, then the result is x = fSlope * y + fIntercept. Otherwise the line is y = fSlope * x + fIntercept.
    float                   fSlope;
    float                   fIntercept;
    PR_Line2f               stLine;
};

// Find the lines in the image.
// The cv::RotatedRect, the angle is the direction of width. Facing right is 0. Clock-Wise is positive, Anti-Clock_Wise is negtive.
// The angle unit is degree.
//                |   /
//                |  /
//                | /
//                |/ -60 degree
// -------------------------------------
//               /|\ ) 60 degree
//              / | \
//             /  |  \
//            /   |   \
//           /120 |    \

struct PR_FIND_LINE_CMD {
    PR_FIND_LINE_CMD() :
        enAlgorithm         ( PR_FIND_LINE_ALGORITHM::CALIPER ),
        bFindPair           (false),
        nCaliperCount       (20),
        fCaliperWidth       (30.f),
        nDiffFilterHalfW    (2),
        fDiffFilterSigma    (1.f),
        nEdgeThreshold      (50),
        enSelectEdge        (PR_CALIPER_SELECT_EDGE::MAX_EDGE),
        fRmStrayPointRatio  (0.2f),
        bCheckLinearity     (false),
        fPointMaxOffset     (0.f),
        fMinLinearity       (0.f),
        bCheckAngle         (false),
        fExpectedAngle      (0.f),
        fAngleDiffTolerance (0.f) {}
    cv::Mat                 matInputImg;
    cv::Mat                 matMask;
    cv::RotatedRect         rectRotatedROI;
    PR_FIND_LINE_ALGORITHM  enAlgorithm;
    bool                    bFindPair;          //Used only when find line algorithm is caliper, if it is true, then will find a pair of lines.
    PR_CALIPER_DIR          enDetectDir;        //The explaination can be find in definition of PR_CALIPER_DIR.
    Int32                   nCaliperCount;      //How many calipers will be used to find line. It is used when the PR_FIND_LINE_ALGORITHM is CALIPER.
    float                   fCaliperWidth;      //The width of caliper. It is used when the PR_FIND_LINE_ALGORITHM is CALIPER.
    int                     nDiffFilterHalfW;   //The half width of gaussian diff filter. Details is in caliper tool document.
    float                   fDiffFilterSigma;   //The Sigma of gaussian diff filter. Sigma is the standard deviation.
    int                     nEdgeThreshold;     //The gray scale difference threshold of the edge. Over this threshold consider as an edge candidate.
    PR_CALIPER_SELECT_EDGE  enSelectEdge;       //Used only when find line algorithm is caliper.
    float                   fRmStrayPointRatio; //The ratio to remove the stray point(Fit for one time, then remove the points with largest error and fit again).
    bool                    bCheckLinearity;
    float                   fPointMaxOffset;    //If a point offset to the line over this tolerance, it is a unlinear point. 
    float                   fMinLinearity;
    bool                    bCheckAngle;
    float                   fExpectedAngle;
    float                   fAngleDiffTolerance;
};

struct PR_FIND_LINE_RPY {
    VisionStatus            enStatus;    
    bool                    bReversedFit;   //If it is true, then the result is x = fSlope * y + fIntercept. Otherwise the line is y = fSlope * x + fIntercept.
    float                   fSlope;
    float                   fIntercept;
    PR_Line2f               stLine;
    float                   fIntercept2;    //When the find pair is enabled, it will return the second line intercept.
    PR_Line2f               stLine2;        //When the find pair is enabled, it will return the second line.
    float                   fDistance;      //When the find pair is enabled, it will return the distance of two lines.
    float                   fLinearity;
    float                   fAngle;
    cv::Mat                 matResultImg;
};

struct PR_FIT_PARALLEL_LINE_CMD {
    cv::Mat                 matInputImg;
    cv::Rect                rectArrROI[2];
    Int32                   nThreshold;
    PR_OBJECT_ATTRIBUTE     enAttribute;
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    float                   fErrTol;
};

struct PR_FIT_PARALLEL_LINE_RPY {
    VisionStatus            enStatus;
    bool                    bReversedFit;
    float                   fSlope;
    float                   fIntercept1;
    float                   fIntercept2;
    PR_Line2f               stLine1;
    PR_Line2f               stLine2;
    cv::Mat                 matResultImg;
};

//The rectArrROI is the ROI of the rect edge. the 1st and 2nd should be parallel, and the 3rd and 4th is parallel.
struct PR_FIT_RECT_CMD {
    cv::Mat                 matInputImg;
    cv::Rect                rectArrROI[PR_RECT_EDGE_COUNT];
    Int32                   nThreshold;
    PR_OBJECT_ATTRIBUTE     enAttribute;
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    float                   fErrTol;
};

struct PR_FIT_RECT_RPY {
    VisionStatus            enStatus;
    bool                    bLineOneReversedFit;
    float                   fSlope1;
    bool                    bLineTwoReversedFit;
    float                   fSlope2;
    float                   fArrIntercept[PR_RECT_EDGE_COUNT];
    PR_Line2f               arrLines[PR_RECT_EDGE_COUNT];
    cv::Mat                 matResultImg;
};

struct PR_FIND_EDGE_CMD {
    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    bool                    bAutoThreshold;
    Int32                   nThreshold;
    PR_EDGE_DIRECTION       enDirection;
    float                   fMinLength;
};

struct PR_FIND_EDGE_RPY {
    VisionStatus            enStatus;
    Int32                   nEdgeCount;
    cv::Mat                 matResultImg;
};

//The PR_FitCircle command suitable to fit the circle after use Canny or Sobel edge detector find the rough circle.
struct PR_FIT_CIRCLE_CMD {
    cv::Mat                 matInputImg;
    cv::Mat                 matMask;
    cv::Rect                rectROI;
    PR_FIT_METHOD           enMethod;
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    float                   fErrTol;
    bool                    bPreprocessed;
    bool                    bAutoThreshold;
    Int32                   nThreshold;
    PR_OBJECT_ATTRIBUTE     enAttribute;
    int                     nMaxRansacTime;             //Only used when fitting method is ransac.
    int                     nNumOfPtToFinishRansac;     //Only used when fitting method is ransac.
};

struct PR_FIT_CIRCLE_RPY {
    VisionStatus            enStatus;
    cv::Point2f             ptCircleCtr;
    float                   fRadius;
    cv::Mat                 matResultImg;
};

//The PR_FitCircle command suitable to fit the circle after use Canny or Sobel edge detector find the rough circle.
struct PR_FIT_CIRCLE_BY_POINT_CMD {
    VectorOfPoint2f         vecPoints;
    PR_FIT_METHOD           enMethod;
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    float                   fErrTol;
    bool                    bPreprocessed;
    bool                    bAutoThreshold;
    Int32                   nThreshold;
    PR_OBJECT_ATTRIBUTE     enAttribute;
    int                     nMaxRansacTime;             //Only used when fitting method is ransac.
    int                     nNumOfPtToFinishRansac;     //Only used when fitting method is ransac.
};

struct PR_FIT_CIRCLE_BY_POINT_RPY {
    VisionStatus            enStatus;
    cv::Point2f             ptCircleCtr;
    float                   fRadius;
};

//Use caliper method to find the circle. Caliper introduction https://drive.google.com/open?id=16hrbMJ6gwkz2ErD12aIYPw5oAkq9y045
struct PR_FIND_CIRCLE_CMD {
    PR_FIND_CIRCLE_CMD() :
        bFindCirclePair     (false),
        fStartSrchAngle     (0.f),
        fEndSrchAngle       (0.f),
        nCaliperCount       (20),
        fCaliperWidth       (30.f),
        nDiffFilterHalfW    (2),
        fDiffFilterSigma    (1.f),
        nEdgeThreshold      (50),
        enSelectEdge        (PR_CALIPER_SELECT_EDGE::MAX_EDGE),
        fRmStrayPointRatio (0.2f) {}
    cv::Mat                 matInputImg;
    cv::Mat                 matMask;
    PR_OBJECT_ATTRIBUTE     enInnerAttribute;   //The inner circle attribute, is it darker or brighter than outside.
    cv::Point2f             ptExpectedCircleCtr;
    bool                    bFindCirclePair;    //Find two circles with the same center at the same time.
    float                   fMinSrchRadius;
    float                   fMaxSrchRadius;
    float                   fStartSrchAngle;    //Start search angle, unit is degree, clockwise is positive, anticlockwise is negative.
    float                   fEndSrchAngle;      //End search angle, unit is degree, clockwise is positive, anticlockwise is negative.
    Int32                   nCaliperCount;      //How many calipers will be used to detect circle.
    float                   fCaliperWidth;      //The width of caliper.
    int                     nDiffFilterHalfW;   //The half width of gaussian diff filter. Details is in caliper tool document.
    float                   fDiffFilterSigma;   //The Sigma of gaussian diff filter. Sigma is the standard deviation.
    int                     nEdgeThreshold;     //The gray scale difference threshold of the edge. Over this threshold consider as an edge candidate.
    PR_CALIPER_SELECT_EDGE  enSelectEdge;
    float                   fRmStrayPointRatio; //The ratio to remove the stray point(Fit for one time, then remove the points with largest error and fit again).
};

struct PR_FIND_CIRCLE_RPY {
    VisionStatus            enStatus;
    cv::Point2f             ptCircleCtr;
    float                   fRadius;
    float                   fRadius2;
    cv::Mat                 matResultImg;
};

//It use the basic circle fitting and check the fitted circle.
struct PR_INSP_CIRCLE_CMD {
    cv::Mat                 matInputImg;
    cv::Mat                 matMask;
    cv::Rect                rectROI;
};

struct PR_INSP_CIRCLE_RPY {
    VisionStatus            enStatus;
    float                   fRoundness;
    cv::Point2f             ptCircleCtr;
    float                   fDiameter;
    cv::Mat                 matResultImg;
};

struct PR_OCR_CMD {
    cv::Mat                 matInputImg;
    PR_DIRECTION            enDirection;
    cv::Rect                rectROI;
};

struct PR_OCR_RPY {
    VisionStatus            enStatus;
    String                  strResult;
};

struct PR_POINT_LINE_DISTANCE_CMD {
    PR_POINT_LINE_DISTANCE_CMD() :
        bReversedFit(false) {}
    cv::Point2f             ptInput;
    bool                    bReversedFit;   // If bReversedFit is false, then the line is y = fSlope * x + fIntercept. If bReversedFit is true, then the line is x = fSlope * y + fIntercept.
    float                   fSlope;
    float                   fIntercept;
};

struct PR_POINT_LINE_DISTANCE_RPY {
    float                   fDistance;
};

struct PR_TWO_LINE_ANGLE_CMD {
    PR_Line2f               line1;
    PR_Line2f               line2;
};

struct PR_TWO_LINE_ANGLE_RPY {
    float                   fAngle;
};

struct PR_TWO_LINE_INTERSECT_CMD {
    PR_Line2f               line1;
    PR_Line2f               line2;
};

struct PR_TWO_LINE_INTERSECT_RPY {
    VisionStatus            enStatus;
    cv::Point2f             ptIntersect;    //The intersect point of two lines.
};

struct PR_PARALLEL_LINE_DIST_CMD {
    PR_Line2f               line1;
    PR_Line2f               line2;
};

struct PR_PARALLEL_LINE_DIST_RPY {
    VisionStatus            enStatus;
    float                   fDistance;
};

struct PR_MEASURE_DIST_CMD {
    enum class MODE {
        DIRECT_LINE,
        X_DIRECTION,
        Y_DIRECTION,
    };
    cv::Point2f             ptStart;
    cv::Point2f             ptEnd;
    MODE                    enMeasureMode;
    float                   fFiducialSlope;
};

struct PR_MEASURE_DIST_RPY {
    VisionStatus            enStatus;
    float                   fDistance;
    cv::Point2f             ptCross;
    bool                    bMeasureWithStart;
};

//Introduction of cross section: https://en.wikipedia.org/wiki/Cross_section_(geometry)
struct PR_CROSS_SECTION_AREA_CMD {
    PR_CROSS_SECTION_AREA_CMD() : bClosed(false) {}
    VectorOfPoint2f         vecContourPoints;
    bool                    bClosed;    //The contour is closed
};

struct PR_CROSS_SECTION_AREA_RPY {
    VisionStatus            enStatus;
    float                   fArea;
};

struct PR_RGB_RATIO {
    PR_RGB_RATIO() : fRatioR(1.f), fRatioG(1.f), fRatioB(1.f) {}
    PR_RGB_RATIO(float fRatioR, float fRatioG, float fRatioB ) : fRatioR(fRatioR), fRatioG(fRatioG), fRatioB(fRatioB)   {}
    float                   fRatioR;
    float                   fRatioG;
    float                   fRatioB;
};

struct PR_COLOR_TO_GRAY_CMD {
    cv::Mat                 matInputImg;
    PR_RGB_RATIO            stRatio;
};

struct PR_COLOR_TO_GRAY_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matResultImg;
};

struct PR_FILTER_CMD {
    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    PR_FILTER_TYPE          enType;
    cv::Size                szKernel;
    double                  dSigmaX;        //Only used when filter type is Guassian.
    double                  dSigmaY;        //Only used when filter type is Guassian.
    Int16                   nDiameter;      //Only used when filter type is Bilaterial.
    double                  dSigmaColor;    //Only used when filter type is Bilaterial.
    double                  dSigmaSpace;    //Only used when filter type is Bilaterial.
};

struct PR_FILTER_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matResultImg;
};

struct PR_AUTO_THRESHOLD_CMD {
    cv::Mat                 matInputImg;
    cv::Mat                 matMask;
    cv::Rect                rectROI;
    Int16                   nThresholdNum;
};

struct PR_AUTO_THRESHOLD_RPY {
    VisionStatus            enStatus;
    std::vector<Int16>      vecThreshold;
};

//CC is acronym of of connected components.
struct PR_REMOVE_CC_CMD {
    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    Int16                   nConnectivity;      //connect by 4 or 8 points.
    PR_COMPARE_TYPE         enCompareType;
    float                   fAreaThreshold;
};

struct PR_REMOVE_CC_RPY {
    VisionStatus            enStatus;
    Int32                   nTotalCC;
    Int32                   nRemovedCC;
    cv::Mat                 matResultImg;
};

// Hysteresis: The final step. Canny does use two thresholds (upper and lower):
// If a pixel gradient is higher than the upper threshold, the pixel is accepted as an edge
// If a pixel gradient value is below the lower threshold, then it is rejected.
// If the pixel gradient is between the two thresholds, then it will be accepted only if it is connected to a pixel that is above the upper threshold.
struct PR_DETECT_EDGE_CMD {
    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    Int16                   nThreshold1;        // first threshold for the hysteresis procedure.
    Int16                   nThreshold2;        // second threshold for the hysteresis procedure.
    Int16                   nApertureSize;      // aperture size for the Sobel operator.
};

struct PR_DETECT_EDGE_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matResultImg;
};

struct PR_FILL_HOLE_CMD {
    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    PR_OBJECT_ATTRIBUTE     enAttribute;
    PR_FILL_HOLE_METHOD     enMethod;
    cv::MorphShapes         enMorphShape;
    cv::Size                szMorphKernel;
    Int16                   nMorphIteration;
};

struct PR_FILL_HOLE_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matResultImg;
};

struct PR_PICK_COLOR_CMD {
    PR_PICK_COLOR_CMD() :
        enMethod    (PR_PICK_COLOR_METHOD::SELECT_POINT) {}
    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    PR_PICK_COLOR_METHOD    enMethod;
    cv::Point               ptPick;
    cv::Scalar              scalarSelect;
    Int16                   nColorDiff;
    Int16                   nGrayDiff;
};

struct PR_PICK_COLOR_RPY {
    VisionStatus            enStatus;
    UInt32                  nPickPointCount;
    cv::Mat                 matResultMask;
    cv::Mat                 matResultImg;
};

struct PR_CALIBRATE_CAMERA_CMD {
    PR_CALIBRATE_CAMERA_CMD() : fPatternDist(1.f), fMinTmplMatchScore(90.f) {}
    cv::Mat                 matInputImg;
    cv::Size                szBoardPattern;             //Number of corners per chessboard row and col. szBoardPattern = cv::Size(points_per_row, points_per_col) = cv::Size(columns, rows).
    float                   fPatternDist;               //The real chess board corner to corner distance. Unit: mm.
    float                   fMinTmplMatchScore;
};

struct PR_CALIBRATE_CAMERA_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matIntrinsicMatrix;         //type: CV_64FC1.
    cv::Mat                 matExtrinsicMatrix;         //type: CV_64FC1.
    cv::Mat                 matDistCoeffs;              //type: CV_64FC1.
    double                  dResolutionX;               //Unit: um/pixel.
    double                  dResolutionY;               //Unit: um/pixel.
    VectorOfMat             vecMatRestoreMap;           //The remap matrix to restore image. vector size is 2, the matrix dimension is same as input image.
    //Intermediate result.
    cv::Mat                 matCornerPointsImg;
    VectorOfPoint2f         vecImagePoints;
    VectorOfPoint3f         vecObjectPoints;
    cv::Mat                 matInitialIntrinsicMatrix;  //type: CV_64FC1.
    cv::Mat                 matInitialExtrinsicMatrix;  //type: CV_64FC1.
};

struct PR_RESTORE_IMG_CMD {
    cv::Mat                 matInputImg;
    VectorOfMat             vecMatRestoreMap;
};

struct PR_RESTORE_IMG_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matResultImg;
};

struct PR_CALC_UNDISTORT_RECTIFY_MAP_CMD {
    cv::Mat                 matIntrinsicMatrix; //type: CV_64FC1.
    cv::Mat                 matDistCoeffs;
    cv::Size                szImage;            //The pixel width and height of the image camptured by camera.
};

struct PR_CALC_UNDISTORT_RECTIFY_MAP_RPY {
    VisionStatus            enStatus;
    std::vector<cv::Mat>    vecMatRestoreMap;
};

struct PR_AUTO_LOCATE_LEAD_CMD {
    cv::Mat                     matInputImg;
    PR_AUTO_LOCATE_LEAD_METHOD  enMethod;
    cv::Rect                    rectSrchWindow;
    cv::Rect                    rectChipBody;
    PR_OBJECT_ATTRIBUTE         enLeadAttribute;
    cv::Rect                    rectPadWindow;
    cv::Rect                    rectLeadWindow;
    VectorOfDirection           vecSrchLeadDirections;
};

struct PR_AUTO_LOCATE_LEAD_RPY {
    struct PR_LEAD_INFO {
        PR_DIRECTION        enDir;
        cv::Rect            rectPadWindow;
        cv::Rect            rectLeadWindow;
        cv::Rect            rectSrchWindow;
        Int32               nPadRecordId;
        Int32               nLeadRecordId;
    };
    using VectorOfLeadInfo = std::vector<PR_LEAD_INFO>;

    VisionStatus            enStatus;
    cv::Mat                 matResultImg;
    VectorOfRect            vecLeadLocation;
    VectorOfLeadInfo        vecLeadInfo;
    std::vector<Int32>      vecLeadRecordId;
    std::vector<Int32>      vecPadRecordId;
};

struct PR_INSP_BRIDGE_CMD {
    struct INNER_INSP_CRITERIA {
        INNER_INSP_CRITERIA() : fMaxLengthX(0.f), fMaxLengthY(0.f) {}
        float                   fMaxLengthX;
        float                   fMaxLengthY;
    };

    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    PR_INSP_BRIDGE_MODE     enInspMode;
    VectorOfDirection       vecOuterInspDirection;
    cv::Rect                rectOuterSrchWindow;
    INNER_INSP_CRITERIA     stInnerInspCriteria;
};

struct PR_INSP_BRIDGE_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matResultImg;
    VectorOfRect            vecBridgeWindow;
};

/******************************************
* Chip Inspection Section *
******************************************/
struct PR_LRN_CHIP_CMD {
    PR_LRN_CHIP_CMD() : 
        enInspMode      (PR_INSP_CHIP_MODE::HEAD),
        bAutoThreshold  (true),
        nThreshold      (0) {}
    cv::Mat                 matInputImg;
    cv::Rect2f              rectChip;
    PR_INSP_CHIP_MODE       enInspMode;
    bool                    bAutoThreshold;
    Int16                   nThreshold;
};

struct PR_LRN_CHIP_RPY {
    VisionStatus            enStatus;
    Int32                   nRecordId;
    cv::Size2f              sizeDevice;
    cv::Size2f              arrSizeElectrode[PR_ELECTRODE_COUNT];
    Int16                   nThreshold;
    cv::Mat                 matResultImg;
};

struct PR_INSP_CHIP_CMD {
    PR_INSP_CHIP_CMD() : nRecordId(-1) {}
    cv::Mat                 matInputImg;
    cv::Rect                rectSrchWindow;
    PR_INSP_CHIP_MODE       enInspMode;
    Int32                   nRecordId;
};

struct PR_INSP_CHIP_RPY {
    VisionStatus            enStatus;
    cv::RotatedRect         rotatedRectResult;  //The angle start from left to right, and rotate by anti-clockwise.
    cv::Mat                 matResultImg;
};
/******************************************
* End of Chip Inspection Section
******************************************/

/******************************************
* Inspect Contour Section *
******************************************/
struct PR_LRN_CONTOUR_CMD {
    PR_LRN_CONTOUR_CMD() :
        bAutoThreshold  (true),
        nThreshold      (100) {}
    cv::Mat                 matInputImg;
    cv::Mat                 matMask;
    cv::Rect                rectROI;
    bool                    bAutoThreshold;
    Int16                   nThreshold;
};

struct PR_LRN_CONTOUR_RPY {
    VisionStatus            enStatus;
    Int16                   nThreshold;
    VectorOfVectorOfPoint   vecContours;
    cv::Mat                 matResultImg;
    Int32                   nRecordId;
};

struct PR_INSP_CONTOUR_CMD {
    PR_INSP_CONTOUR_CMD() : 
        nRecordId             (0), 
        nDefectThreshold      (30),
        fMinDefectArea        (100.f),
        fDefectInnerLengthTol (20),
        fDefectOuterLengthTol (20),
        fInnerMaskDepth       (5),
        fOuterMaskDepth       (5) {}
    cv::Mat                 matInputImg;
    cv::Mat                 matMask;
    cv::Rect                rectROI;
    Int32                   nRecordId;
    Int16                   nDefectThreshold;
    float                   fMinDefectArea;
    float                   fDefectInnerLengthTol;   // If the defect's inner length exceed the tolerance, then it is a true defect.
    float                   fDefectOuterLengthTol;   // If the defect's outer length exceed the tolerance, then it is a true defect.
    float                   fInnerMaskDepth;
    float                   fOuterMaskDepth;
};

struct PR_INSP_CONTOUR_RPY {
    VisionStatus            enStatus;
    VectorOfVectorOfPoint   vecDefectContour;
    cv::Mat                 matResultImg;
};

/******************************************
* End of Inspect Contour Section *
******************************************/

/******************************************
* Inspect Hole Section *
******************************************/
struct PR_INSP_HOLE_CMD {
    struct GRAY_SCALE_RANGE {
        GRAY_SCALE_RANGE() : nStart(0), nEnd(PR_MAX_GRAY_LEVEL) {}
        Int16               nStart;
        Int16               nEnd;
    };

    struct COLOR_RANGE {
        COLOR_RANGE() : nStartB(0), nEndB(PR_MAX_GRAY_LEVEL), nStartG(0), nEndG(PR_MAX_GRAY_LEVEL), nStartR(0), nEndR(PR_MAX_GRAY_LEVEL) {}
        Int16               nStartB;    //Blue range start.
        Int16               nEndB;      //Blue range end.
        Int16               nStartG;    //Green range start.
        Int16               nEndG;      //Green range end.
        Int16               nStartR;    //Red range start.
        Int16               nEndR;      //Red range end.
    };

    struct RATIO_MODE_CRITERIA {
        RATIO_MODE_CRITERIA() : fMinRatio(0.1f), fMaxRatio(1.f) {}
        float               fMaxRatio;
        float               fMinRatio;
    };

    struct BLOB_MODE_CRITERIA {
        BLOB_MODE_CRITERIA() : fMaxArea(1000000.f), fMinArea(500.f), nMinBlobCount(0), nMaxBlobCount(10), bEnableAdvancedCriteria(false) {}
        float               fMaxArea;
        float               fMinArea;
        Int16               nMaxBlobCount;
        Int16               nMinBlobCount;
        bool                bEnableAdvancedCriteria;
        struct ADVANCED_CRITERIA {
            ADVANCED_CRITERIA() : fMaxLengthWidthRatio(1.f), fMinLengthWidthRatio(0.1f), fMaxCircularity(1.f), fMinCircularity(0.1f) {}
            float           fMaxLengthWidthRatio;
            float           fMinLengthWidthRatio;
            float           fMaxCircularity;
            float           fMinCircularity;
        };
        ADVANCED_CRITERIA   stAdvancedCriteria;
    };

    PR_INSP_HOLE_CMD() :
        bPreprocessedImg    (false),
        enSegmentMethod     (PR_IMG_SEGMENT_METHOD::GRAY_SCALE_RANGE) {}

    cv::Mat                 matInputImg;
    cv::Mat                 matMask;
    cv::Rect                rectROI;
    bool                    bPreprocessedImg;
    PR_IMG_SEGMENT_METHOD   enSegmentMethod;
    GRAY_SCALE_RANGE        stGrayScaleRange;   //It is used when enSegmentMethod is PR_IMG_SEGMENT_METHOD::GRAY_SCALE_RANGE.
    COLOR_RANGE             stColorRange;       //It is used when enSegmentMethod is PR_IMG_SEGMENT_METHOD::COLOR_RANGE.
    PR_INSP_HOLE_MODE       enInspMode;
    RATIO_MODE_CRITERIA     stRatioModeCriteria;
    BLOB_MODE_CRITERIA      stBlobModeCriteria;
};

struct PR_INSP_HOLE_RPY {
    struct RATIO_MODE_RESULT {
        float               fRatio;
    };

    struct BLOB_MODE_RESULT {
        VectorOfKeyPoint    vecBlobs;
    };

    VisionStatus            enStatus;
    RATIO_MODE_RESULT       stRatioModeResult;
    BLOB_MODE_RESULT        stBlobModeResult;
    cv::Mat                 matResultImg;
};
/******************************************
* End of Inspect Hole Section *
******************************************/

/******************************************
* Inspect Lead Section *
******************************************/
struct PR_INSP_LEAD_CMD {
    struct LEAD_INPUT_INFO {
        cv::RotatedRect     rectSrchWindow;
        cv::RotatedRect     rectExpectedWindow;
    };
    using VECTOR_LEAD_INPUT_INFO = std::vector<LEAD_INPUT_INFO>;
    PR_INSP_LEAD_CMD() : fLeadStartWidthRatio ( 0.5f ), nLeadStartConsecutiveLength(2), fLeadEndWidthRatio (0.5f), nLeadEndConsecutiveLength(2), enFindLeadEndMethod(PR_FIND_LEAD_END_METHOD::AVERAGE) {}
    cv::Mat                 matInputImg;
    cv::RotatedRect         rectChipWindow;
    VECTOR_LEAD_INPUT_INFO  vecLeads;
    float                   fLeadStartWidthRatio;
    Int16                   nLeadStartConsecutiveLength;
    float                   fLeadEndWidthRatio;
    Int16                   nLeadEndConsecutiveLength;
    PR_FIND_LEAD_END_METHOD enFindLeadEndMethod;
};

struct PR_INSP_LEAD_RPY {  
    struct LEAD_RESULT {
        bool                bFound;
        cv::RotatedRect     rectLead;
    };
    using VECTOR_LEAD_RESULT = std::vector<LEAD_RESULT>;
    VisionStatus            enStatus;
    VECTOR_LEAD_RESULT      vecLeadResult;
    cv::Mat                 matResultImg;
};

struct PR_INSP_LEAD_TMPL_CMD {
    PR_INSP_LEAD_TMPL_CMD() :
        nPadRecordId        (-1),
        nLeadRecordId       (-1),
        fMinMatchScore      (60),
        fLrnedPadLeadDist   (0) {}
    cv::Mat             matInputImg;
    cv::Rect            rectROI;
    Int32               nPadRecordId;
    Int32               nLeadRecordId;
    float               fMaxLeadOffsetX;
    float               fMaxLeadOffsetY;
    float               fLrnedPadLeadDist;
    float               fMinMatchScore;
};

struct PR_INSP_LEAD_TMPL_RPY {
    VisionStatus        enStatus;
    cv::Mat             matResultImg;
    cv::Point2f         ptPadPos;
    cv::Point2f         ptLeadPos;
    float               fLeadOffsetX;
    float               fLeadOffsetY;
};
/******************************************
* End of Inspect Lead Section *
******************************************/

/******************************************
* Inspect Polarity Section *
******************************************/
struct PR_INSP_POLARITY_CMD {
    PR_INSP_POLARITY_CMD() :
        enInspROIAttribute  (PR_OBJECT_ATTRIBUTE::BRIGHT),
        nGrayScaleDiffTol   (50) {}
    cv::Mat                 matInputImg;
    cv::Rect                rectInspROI;
    cv::Rect                rectCompareROI;
    PR_OBJECT_ATTRIBUTE     enInspROIAttribute;
    Int16                   nGrayScaleDiffTol;
};

struct PR_INSP_POLARITY_RPY {
    VisionStatus            enStatus;
    Int16                   nGrayScaleDiff;
    cv::Mat                 matResultImg;
};
/******************************************
* End of Inspect Polarity Section *
******************************************/

struct PR_GRID_AVG_GRAY_SCALE_CMD {
    PR_GRID_AVG_GRAY_SCALE_CMD() : nGridRow(5), nGridCol(5) {}
    VectorOfMat             vecInputImgs;
    Int16                   nGridRow;
    Int16                   nGridCol;
};

struct PR_GRID_AVG_GRAY_SCALE_RPY {    
    VisionStatus            enStatus;
    VectorOfVectorOfFloat   vecVecGrayScale;
    cv::Mat                 matResultImg;
};

struct PR_CALIB_3D_BASE_CMD {
    PR_CALIB_3D_BASE_CMD() : bEnableGaussianFilter(true), fRemoveHarmonicWaveK(0.f) {}
    VectorOfMat             vecInputImgs;           //Totally 12 images, including 4 thick, 4 thin, 4 extreme thin.
    bool                    bEnableGaussianFilter;    
    float                   fRemoveHarmonicWaveK;   //The factor to remove the harmonic wave in the thick stripe. If it is 0, then this procedure will be skipped.
};

struct PR_CALIB_3D_BASE_RPY {
    VisionStatus            enStatus;
    bool                    bReverseSeq;            //Change the image sequence.
    PR_DIRECTION            enProjectDir;           //The DLP project direction. UP means from top to bottom. Left means from left to right. To keep the same definition as Chen Lining's matlab code.
    PR_DIRECTION            enScanDir;              //The phase correction direction.
    cv::Mat                 matThickToThinK;        //The factor between thick stripe and thin stripe.
    cv::Mat                 matThickToThinnestK;    //The factor between thick stripe and thinnest stripe.
    cv::Mat                 matBaseWrappedAlpha;    //The wrapped thick stripe phase.
    cv::Mat                 matBaseWrappedBeta;     //The wrapped thin stripe phase.
    cv::Mat                 matBaseWrappedGamma;    //The wrapped thinnest stripe phase.
};

struct PR_CALC_3D_BASE_CMD {
    PR_CALC_3D_BASE_CMD() :
        nImageRows(2048),
        nImageCols(2040) {}
    cv::Mat                 matBaseSurfaceParam;
    int                     nImageRows;         //The rows of camera image.
    int                     nImageCols;         //The cols of camera image.
};

struct PR_CALC_3D_BASE_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matBaseSurface;
};

struct PR_CALIB_3D_HEIGHT_CMD {
    PR_CALIB_3D_HEIGHT_CMD() :
        bEnableGaussianFilter(true),
        bReverseSeq(true),
        bReverseHeight(false),
        bUseThinnestPattern(false),
        fRemoveHarmonicWaveK(0.f),
        fMinAmplitude(5.f),
        nRemoveBetaJumpMinSpan(25),
        nRemoveBetaJumpMaxSpan(80),
        nRemoveGammaJumpSpanX(23),
        nRemoveGammaJumpSpanY(4),
        nBlockStepCount(4),
        fBlockStepHeight(1.f),
        nResultImgGridRow(8),
        nResultImgGridCol(8),
        szMeasureWinSize(40, 40) {}
    VectorOfMat             vecInputImgs;
    bool                    bEnableGaussianFilter;
    bool                    bReverseSeq;            //Change the image sequence.    
    bool                    bReverseHeight;         //The calibration base is align to the top of calibration block.
    bool                    bUseThinnestPattern;    //Choose to use the thinnest stripe pattern. Otherwise just use the normal thin stripe.
    float                   fRemoveHarmonicWaveK;   //The factor to remove the harmonic wave in the thick stripe. If it is 0, then this procedure will be skipped.
    float                   fMinAmplitude;          //In a group of 4 images, if a pixel's gray scale amplitude less than this value, this pixel will be discarded.
    cv::Mat                 matThickToThinK;        //The factor between thick stripe and thin stripe.
    cv::Mat                 matThickToThinnestK;    //The factor between thick stripe and thinnest stripe.
    cv::Mat                 matBaseWrappedAlpha;    //The wrapped thick stripe phase.
    cv::Mat                 matBaseWrappedBeta;     //The wrapped thin stripe phase.
    cv::Mat                 matBaseWrappedGamma;    //The wrapped thin stripe phase.
    int                     nRemoveBetaJumpMinSpan; //The phase jump span in X direction under this value in beta phase(the thin pattern) will be removed.
    int                     nRemoveBetaJumpMaxSpan; //The phase jump span in Y direction under this value in beta phase(the thin pattern) will be removed.
    int                     nRemoveGammaJumpSpanX;  //The phase jump span in X direction under this value in gamma phase(the thinnest pattern) will be removed. It is used only when bUseThinnestPattern is true.
    int                     nRemoveGammaJumpSpanY;  //The phase jump span in Y direction under this value in gamma phase(the thinnest pattern) will be removed. It is used only when bUseThinnestPattern is true.
    //Below is the calibration related parameters
    Int16                   nBlockStepCount;        //How many steps on the calibration block.
    float                   fBlockStepHeight;       //The height of each step, unit mm. So the total block height is nBlockStepCount x fBlockStepHeight.
    Int32                   nResultImgGridRow;
    Int32                   nResultImgGridCol;
    cv::Size                szMeasureWinSize;       //The window size in the center of grid to measure the height.
};

struct PR_CALIB_3D_HEIGHT_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matPhaseToHeightK;      //The factor to convert phase to height.
    VectorOfVectorOfFloat   vecVecStepPhase;        //The result phase of 4 corners and the center, for application to draw the curves.
    VectorOfMat             vecMatStepSurface;      //The regression surface of the steps. Its size should be nBlockStepCount + 1.
    VectorOfFloat           vecStepPhaseSlope;      //5 slopes of the phase-step fitting lines.
    VectorOfVectorOfFloat   vecVecStepPhaseDiff;    //The actual phase and the fitting line difference.
    cv::Mat                 matDivideStepIndex;     //Denote each pixel belong to which step. The type is CV_8SC1, can be positive or negative.
    cv::Mat                 matPhase;               //The unwrapped phase.
    cv::Mat                 matDivideStepResultImg; //Use auto threshold to divide each step of the phase image. This result image can show to user confirm if the auto threshold is working correctly.
    cv::Mat                 matResultImg;
};

struct PR_INTEGRATE_3D_CALIB_CMD {
    PR_INTEGRATE_3D_CALIB_CMD() :
        nResultImgGridRow   (10),
        nResultImgGridCol   (10),
        szMeasureWinSize    (40, 40) {}
    struct SINGLE_CALIB_DATA {
        cv::Mat             matPhase;
        cv::Mat             matDivideStepIndex;
    };
    using CALIB_DATA_VECTOR = std::vector<SINGLE_CALIB_DATA>;
    CALIB_DATA_VECTOR       vecCalibData;
    cv::Mat                 matTopSurfacePhase;
    float                   fTopSurfaceHeight;
    Int32                   nResultImgGridRow;
    Int32                   nResultImgGridCol;
    cv::Size                szMeasureWinSize;       //The window size in the center of grid to measure the height.
};

struct PR_INTEGRATE_3D_CALIB_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matIntegratedK;         //The 12 parameters to calculate height. They are K1~K10 and P1, P2. H = (Phase + P1*Phase^2 + P2*Phase^3) ./ (K(1)*x1.^3 + K(2)*y1.^3 + K(3)*x1.^2.*y1 + K(4)*x1.*y1.^2 + K(5)*x1.^2 + K(6)*y1.^2 + K(7)*x1.*y1 + K(8)*x1 + K(9)*y1 + K(10))
    cv::Mat                 matOrder3CurveSurface;  //The regression 3 order curve surface to convert phase to height. It is calculated by K(1)*x1.^3 + K(2)*y1.^3 + K(3)*x1.^2.*y1 + K(4)*x1.*y1.^2 + K(5)*x1.^2 + K(6)*y1.^2 + K(7)*x1.*y1 + K(8)*x1 + K(9)*y1 + K(10)
    VectorOfMat             vecMatResultImg;
};

using PairHeightPhase = std::pair<float, cv::Mat>;
using VectorPairHeightPhase = std::vector<PairHeightPhase>;
using PairHeightCalibResult = std::pair<float, cv::Mat>;
using VectorHeightCalibResult = std::vector<PairHeightCalibResult>;
using PairHeightGridHeights = std::pair<float, VectorOfFloat>;
using VectorHeightGridHeights = std::vector<PairHeightGridHeights>;

struct PR_MOTOR_CALIB_3D_CMD {
    PR_MOTOR_CALIB_3D_CMD() :
        nResultImgGridRow(10),
        nResultImgGridCol(10),
        szMeasureWinSize(40, 40) {}
    VectorPairHeightPhase   vecPairHeightPhase;
    Int32                   nResultImgGridRow;
    Int32                   nResultImgGridCol;
    cv::Size                szMeasureWinSize;       //The window size in the center of grid to measure the height.
};

struct PR_MOTOR_CALIB_3D_RPY {
    VisionStatus            enStatus;
    float                   fMaxPhase;
    float                   fMinPhase;
    cv::Mat                 matIntegratedK;         //The 12 parameters to calculate height. They are K1~K10 and P1, P2. H = (Phase + P1*Phase^2 + P2*Phase^3) ./ (K(1)*x1.^3 + K(2)*y1.^3 + K(3)*x1.^2.*y1 + K(4)*x1.*y1.^2 + K(5)*x1.^2 + K(6)*y1.^2 + K(7)*x1.*y1 + K(8)*x1 + K(9)*y1 + K(10))
    cv::Mat                 matOrder3CurveSurface;  //The regression 3 order curve surface to convert phase to height. It is calculated by K(1)*x1.^3 + K(2)*y1.^3 + K(3)*x1.^2.*y1 + K(4)*x1.*y1.^2 + K(5)*x1.^2 + K(6)*y1.^2 + K(7)*x1.*y1 + K(8)*x1 + K(9)*y1 + K(10)
    VectorOfMat             vecMatResultImg;
    VectorHeightCalibResult vecHeightCalibResult;
    VectorHeightGridHeights vecHeightGridHeights;
};

struct PR_MOTOR_CALIB_3D_NEW_RPY {
    VisionStatus            enStatus;
    float                   fMaxPhase;
    float                   fMinPhase;
    cv::Mat                 mat3DBezierK;           // The 100 3D Bezier parameters. 5 x 5 x 4 = 100
    cv::Mat                 mat3DBezierSurface;     // The 3D bezier surface calculated from bezier parameters
    VectorOfMat             vecMatResultImg;
    VectorHeightCalibResult vecHeightCalibResult;
    VectorHeightGridHeights vecHeightGridHeights;
};

struct PR_CALIB_DLP_OFFSET_CMD {
    int                     nCalibPosRows;
    int                     nCalibPosCols;
    float                   fFrameDistX;            // The X distance between frame, unit in mm
    float                   fFrameDistY;            // The Y distance between frame, unit in mm
    float                   fResolution;            // Camera resolution
    VectorOfMat             arrVecDlpH[NUM_OF_DLP]; // The vector size of DLP height should equal to nCalibPosRows * nCalibPosCols
};

struct PR_CALIB_DLP_OFFSET_RPY {
    VisionStatus            enStatus;
    float                   arrOffset[NUM_OF_DLP];
    cv::Mat                 arrMatRotationSurface[NUM_OF_DLP];
};

struct PR_CALC_3D_HEIGHT_CMD {
    PR_CALC_3D_HEIGHT_CMD() :
        bEnableGaussianFilter(true),
        bReverseSeq(true),
        enProjectDir(PR_DIRECTION::LEFT),
        enScanDir(PR_DIRECTION::DOWN),
        bUseThinnestPattern(false),
        fRemoveHarmonicWaveK(0.f),
        fMinAmplitude(5.f),
        fPhaseShift(0.f),
        nRemoveBetaJumpMinSpan(25),
        nRemoveBetaJumpMaxSpan(80),
        nRemoveGammaJumpSpanX(23),
        nRemoveGammaJumpSpanY(4) {}
    VectorOfMat             vecInputImgs;
    bool                    bEnableGaussianFilter;
    bool                    bReverseSeq;             //Change the image sequence.
    PR_DIRECTION            enProjectDir;            //The DLP project direction. Get it from PR_CALIB_3D_BASE_RPY.
    PR_DIRECTION            enScanDir;               //The phase correction direction. Get it from PR_CALIB_3D_BASE_RPY.
    bool                    bUseThinnestPattern;     //Choose to use the thinnest stripe pattern. Otherwise just use the normal thin stripe.
    float                   fRemoveHarmonicWaveK;    //The factor to remove the harmonic wave in the thick stripe. If it is 0, then this procedure will be skipped.
    float                   fMinAmplitude;           //In a group of 4 images, if a pixel's gray scale amplitude less than this value, this pixel will be discarded.
    float                   fPhaseShift;             //Shift the phase measure range. Normal is -1~1, sometimes the H5 surface has corner over this range. So if shift is 0.1, the measure range is -0.9~1.1, so it can measure all the H5 surface.
    cv::Mat                 matThickToThinK;         //The factor between thick stripe and thin stripe.
    cv::Mat                 matThickToThinnestK;     //The factor between thick stripe and thinnest stripe.
    cv::Mat                 matBaseWrappedAlpha;     //The wrapped thick stripe phase.
    cv::Mat                 matBaseWrappedBeta;      //The wrapped thin stripe phase.
    cv::Mat                 matBaseWrappedGamma;     //The wrapped thin stripe phase.
    cv::Mat                 matPhaseToHeightK;       //The factor to convert phase to height. This is the single group of image calibration result.
    int                     nRemoveBetaJumpMinSpan;  //The phase jump span in the specified range of beta phase(the thin pattern) will be removed.
    int                     nRemoveBetaJumpMaxSpan;  //The phase jump span in the specified range of beta phase(the thin pattern) will be removed.
    int                     nRemoveGammaJumpSpanX;   //The phase jump span in X direction under this value in gamma phase(the thinnest pattern) will be removed. It is used only when bUseThinnestPattern is true.
    int                     nRemoveGammaJumpSpanY;   //The phase jump span in Y direction under this value in gamma phase(the thinnest pattern) will be removed. It is used only when bUseThinnestPattern is true.
    //Below 2 parameters are result of PR_MotorCalib3D, they are calibrated from positive, negative and H = 5mm surface phase. If these 2 parameters are used, then matPhaseToHeightK will be ignored.
    cv::Mat                 matIntegratedK;          //The 12 parameters to calculate height. They are K1~K10 and P1, P2. H = (Phase + P1*Phase^2 + P2*Phase^3) ./ (K(1)*x1.^3 + K(2)*y1.^3 + K(3)*x1.^2.*y1 + K(4)*x1.*y1.^2 + K(5)*x1.^2 + K(6)*y1.^2 + K(7)*x1.*y1 + K(8)*x1 + K(9)*y1 + K(10))
    cv::Mat                 matOrder3CurveSurface;   //The regression 3 order curve surface to convert phase to height. It is calculated by K(1)*x1.^3 + K(2)*y1.^3 + K(3)*x1.^2.*y1 + K(4)*x1.*y1.^2 + K(5)*x1.^2 + K(6)*y1.^2 + K(7)*x1.*y1 + K(8)*x1 + K(9)*y1 + K(10)
};

struct PR_CALC_3D_HEIGHT_NEW_CMD {
    PR_CALC_3D_HEIGHT_NEW_CMD() :
        bEnableGaussianFilter(true),
        bReverseSeq(true),
        enProjectDir(PR_DIRECTION::LEFT),
        enScanDir(PR_DIRECTION::DOWN),
        bUseThinnestPattern(false),
        fRemoveHarmonicWaveK(0.f),
        fMinAmplitude(3.f),
        fPhaseShift(0.f),
        nRemoveJumpSpan(7),
        nCompareRemoveJumpSpan(15) {}
    VectorOfMat             vecInputImgs;
    bool                    bEnableGaussianFilter;
    bool                    bReverseSeq;             //Change the image sequence.
    PR_DIRECTION            enProjectDir;            //The DLP project direction. Get it from PR_CALIB_3D_BASE_RPY.
    PR_DIRECTION            enScanDir;               //The phase correction direction. Get it from PR_CALIB_3D_BASE_RPY.
    bool                    bUseThinnestPattern;     //Choose to use the thinnest stripe pattern. Otherwise just use the normal thin stripe.
    float                   fRemoveHarmonicWaveK;    //The factor to remove the harmonic wave in the thick stripe. If it is 0, then this procedure will be skipped.
    float                   fMinAmplitude;           //In a group of 4 images, if a pixel's gray scale amplitude less than this value, this pixel will be discarded.
    float                   fPhaseShift;             //Shift the phase measure range. Normal is -1~1, sometimes the H5 surface has corner over this range. So if shift is 0.1, the measure range is -0.9~1.1, so it can measure all the H5 surface.
    cv::Mat                 matThickToThinK;         //The factor between thick stripe and thin stripe.
    cv::Mat                 matThickToThinnestK;     //The factor between thick stripe and thinnest stripe.
    cv::Mat                 matBaseWrappedAlpha;     //The wrapped thick stripe phase.
    cv::Mat                 matBaseWrappedBeta;      //The wrapped thin stripe phase.
    cv::Mat                 matBaseWrappedGamma;     //The wrapped thin stripe phase.
    cv::Mat                 matPhaseToHeightK;       //The factor to convert phase to height. This is the single group of image calibration result.
    int                     nRemoveJumpSpan;         //The phase jump span in X and Y direction under this value in beta phase will be removed.
    int                     nCompareRemoveJumpSpan;  //The compared phase jump span in X and Y direction under this value in gamma phase will be removed.
    //Below 2 parameters are result of PR_MotorCalib3D, they are calibrated from positive, negative and H = 5mm surface phase. If these 2 parameters are used, then matPhaseToHeightK will be ignored.
    float                   fMaxPhase;
    float                   fMinPhase;
    cv::Mat                 mat3DBezierK;           // The 100 3D Bezier parameters. 5 x 5 x 4 = 100
    cv::Mat                 mat3DBezierSurface;     // The 3D bezier surface calculated from bezier parameters
};

struct PR_CALC_3D_HEIGHT_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matPhase;
    cv::Mat                 matHeight;
    cv::Mat                 matNanMask;
};

struct PR_SET_DLP_PARAMS_TO_GPU_CMD {
    cv::Mat                 vec3DBezierSurface[NUM_OF_DLP];
    cv::Mat                 vecMatAlphaBase[NUM_OF_DLP];
    cv::Mat                 vecMatBetaBase[NUM_OF_DLP];
    cv::Mat                 vecMatGammaBase[NUM_OF_DLP];
};

struct PR_SET_DLP_PARAMS_TO_GPU_RPY {
    VisionStatus            enStatus;
};

struct PR_CALC_3D_HEIGHT_GPU_CMD {
    PR_CALC_3D_HEIGHT_GPU_CMD() :
        nDlpNo(0),
        bEnableGaussianFilter(true),
        bReverseSeq(true),
        enProjectDir(PR_DIRECTION::LEFT),
        enScanDir(PR_DIRECTION::DOWN),
        bUseThinnestPattern(false),
        fMinAmplitude(3.f),
        fPhaseShift(0.f),
        fBaseRangeMin(0.25f),
        fBaseRangeMax(0.45f),
        nRemoveJumpSpan(7),
        nCompareRemoveJumpSpan(15) {}
    VectorOfMat             vecInputImgs;
    int                     nDlpNo;
    bool                    bEnableGaussianFilter;
    bool                    bReverseSeq;             //Change the image sequence.
    PR_DIRECTION            enProjectDir;            //The DLP project direction. Get it from PR_CALIB_3D_BASE_RPY.
    PR_DIRECTION            enScanDir;               //The phase correction direction. Get it from PR_CALIB_3D_BASE_RPY.
    bool                    bUseThinnestPattern;     //Choose to use the thinnest stripe pattern. Otherwise just use the normal thin stripe.
    float                   fMinAmplitude;           //In a group of 4 images, if a pixel's gray scale amplitude less than this value, this pixel will be discarded.
    float                   fPhaseShift;             //Shift the phase measure range. Normal is -1~1, sometimes the H5 surface has corner over this range. So if shift is 0.1, the measure range is -0.9~1.1, so it can measure all the H5 surface.
    float                   fBaseRangeMin;           //The average height in the base range will be used as the base height. fBaseRangeMin is the start
    float                   fBaseRangeMax;           //The average height in the base range will be used as the base height. fBaseRangeMax is the end
    cv::Mat                 matThickToThinK;         //The factor between thick stripe and thin stripe.
    cv::Mat                 matThickToThinnestK;     //The factor between thick stripe and thinnest stripe.
    cv::Mat                 matPhaseToHeightK;       //The factor to convert phase to height. This is the single group of image calibration result.
    int                     nRemoveJumpSpan;         //The phase jump span in X and Y direction under this value in beta phase will be removed.
    int                     nCompareRemoveJumpSpan;  //The compared phase jump span in X and Y direction under this value in gamma phase will be removed.
    //Below 2 parameters are result of PR_MotorCalib3D, they are calibrated from positive, negative and H = 5mm surface phase. If these 2 parameters are used, then matPhaseToHeightK will be ignored.
    float                   fMaxPhase;
    float                   fMinPhase;
    cv::Mat                 mat3DBezierK;           // The 100 3D Bezier parameters. 5 x 5 x 4 = 100
};

struct PR_MERGE_3D_HEIGHT_CMD {
    PR_MERGE_3D_HEIGHT_CMD() :
        enMethod (PR_MERGE_3D_HT_METHOD::SELECT_NEAREST_INTERSECT),
        fHeightDiffThreshold(0.2f) {}
    VectorOfMat             vecMatHeight;
    VectorOfMat             vecMatNanMask;
    VectorOfDirection       vecProjDir;
    PR_MERGE_3D_HT_METHOD   enMethod;
    float                   fHeightDiffThreshold;   //The height difference threshold. Unit mm. If height difference less than it, the result height is average of the input height. If larger than it, the result height use the small height.
};

struct PR_MERGE_3D_HEIGHT_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matHeight;
    cv::Mat                 matNanMask;
};

struct PR_CALC_MERGE_4_DLP_HEIGHT_CMD {
    PR_CALC_3D_HEIGHT_GPU_CMD arrCalcHeightCmd[NUM_OF_DLP];
    float                   fHeightDiffThreshold;   //The height difference threshold. Unit mm. If height difference less than it, the result height is average of the input height. If larger than it, the result height use the small height.
};

struct PR_CALC_MERGE_4_DLP_HEIGHT_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matHeight;
};

struct PR_CALC_3D_HEIGHT_DIFF_CMD {
    PR_CALC_3D_HEIGHT_DIFF_CMD() :
        fEffectHRatioStart(0.3f),
        fEffectHRatioEnd(0.7f) {}
    cv::Mat                 matHeight;
    cv::Mat                 matMask;
    VectorOfRect            vecRectBases;           //One or more bases as the reference surface.
    cv::Rect                rectROI;                //The ROI to measure height difference to base.
    float                   fEffectHRatioStart;     //If fEffectHRatioStart = 0.3, the lower 30% points in the window will be removed for fitting.
    float                   fEffectHRatioEnd;       //If fEffectHRatioEnd = 0.7, the upper 30% of points in the window will be removed for fitting.
};

struct PR_CALC_3D_HEIGHT_DIFF_RPY {
    VisionStatus            enStatus;
    float                   fHeightDiff;
};

struct PR_INSP_3D_SOLDER_CMD {
    PR_INSP_3D_SOLDER_CMD() :
        nBaseColorDiff              (20),
        nBaseGrayDiff               (20),
        nWettingWidth               (8),
        fHeightVariationCoverage    (0.75f) {}
    cv::Mat                 matHeight;
    cv::Mat                 matColorImg;
    cv::Rect                rectDeviceROI;
    cv::Scalar              scalarBaseColor;
    Int16                   nBaseColorDiff;
    Int16                   nBaseGrayDiff;
    VectorOfRect            vecRectCheckROIs;
    int                     nWettingWidth;
    float                   fHeightVariationCoverage; // When find the solder area, the highest variation part of the solder is choosen.
};

struct PR_INSP_3D_SOLDER_RPY {
    struct RESULT {
        float               fComponentHeight;
        float               fSolderHeight;
        float               fSolderArea;
        float               fSolderRatio;
    };
    VisionStatus            enStatus;
    std::vector<RESULT>     vecResults;
    cv::Mat                 matResultImg;
};

// Calculate the DLP height offset, it using DLP4 as the base.
struct PR_CALC_DLP_OFFSET_CMD {
    cv::Mat                 matHeight1;
    cv::Mat                 matHeight2;
    cv::Mat                 matHeight3;
    cv::Mat                 matHeight4;
};

struct PR_CALC_DLP_OFFSET_RPY {
    VisionStatus            enStatus;
    float                   fOffset1;
    float                   fOffset2;
    float                   fOffset3;
    float                   fOffset4;
};

// Interpolate the value of point with the already known points values
// It can be used to calculate 3D height compensation.
struct PR_CALC_FRAME_VALUE_CMD {
    VectorOfVectorOfPoint2f vecVecRefFrameCenters;
    VectorOfVectorOfFloat   vecVecRefFrameValues;
    cv::Point2f             ptTargetFrameCenter;
};

struct PR_CALC_FRAME_VALUE_RPY {
    VisionStatus            enStatus;
    float                   fResult;
};

// Calculate the system optics modulation transfer function.
// The system is from DLP->DLP Optics->Camera Optics->Camera
struct PR_CALC_MTF_CMD {
    PR_CALC_MTF_CMD() : fMagnitudeOfDLP(161) {}
    VectorOfMat             vecInputImgs;
    float                   fMagnitudeOfDLP;        //The setted magnitude of DLP. The captured image magnitude divide the setted maganitude is the MTF result.
};

struct PR_CALC_MTF_RPY {
    VisionStatus            enStatus;
    VectorOfVectorOfFloat   vecVecAbsMtfH;          //The absolute modulation transfer function in horizontal direction.
    VectorOfVectorOfFloat   vecVecRelMtfH;          //The relative modulation transfer function in horizontal direction.
    VectorOfVectorOfFloat   vecVecAbsMtfV;          //The absolute modulation transfer function in vertical direction.
    VectorOfVectorOfFloat   vecVecRelMtfV;          //The relative modulation transfer function in vertical direction.
};

struct PR_CALC_CAMERA_MTF_CMD {
    cv::Mat                 matInputImg;
    cv::Rect                rectVBigPatternROI;
    cv::Rect                rectHBigPatternROI;
    cv::Rect                rectVSmallPatternROI;
    cv::Rect                rectHSmallPatternROI;
};

struct PR_CALC_CAMERA_MTF_RPY {
    VisionStatus            enStatus;
    float                   fBigPatternAbsMtfV;     //The big pattern absolute modulation transfer function in vertical direction.
    float                   fBigPatternAbsMtfH;     //The big pattern absolute modulation transfer function in horizontal direction.
    float                   fSmallPatternAbsMtfV;   //The small pattern absolute modulation transfer function in vertical direction.
    float                   fSmallPatternAbsMtfH;   //The small pattern absolute modulation transfer function in horizontal direction.
    float                   fSmallPatternRelMtfV;   //The small pattern relative modulation transfer function in vertical direction.
    float                   fSmallPatternRelMtfH;   //The small pattern relative modulation transfer function in horizontal direction.
    VectorOfFloat           vecBigPatternAbsMtfV;
    VectorOfFloat           vecBigPatternAbsMtfH;
    VectorOfFloat           vecSmallPatternAbsMtfV;
    VectorOfFloat           vecSmallPatternAbsMtfH;
};

//Calculate the pattern distortion(PD). Use the texture stripe in two directions to find out the system distortion.
struct PR_CALC_PD_CMD {
    PR_CALC_PD_CMD() :
        bReverseSeq(true),
        fMagnitudeOfDLP(161.f),
        szDlpPatternSize(912, 1140),
        fDlpPixelCycle(30),
        nGaussianFilterSize(9),
        fGaussianFilterSigma(3.16f) {}
    VectorOfMat             vecInputImgs;
    bool                    bReverseSeq;            //Change the image sequence.
    float                   fMagnitudeOfDLP;
    cv::Size                szDlpPatternSize;
    float                   fDlpPixelCycle;
    Int32                   nGaussianFilterSize;
    float                   fGaussianFilterSigma;
};

struct PR_CALC_PD_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matCaptureRegionImg;    //The camera capture region in the DLP incident area.
    VectorOfFloat           vecDistortionLeft;
    VectorOfFloat           vecDistortionRight;
    VectorOfFloat           vecDistortionTop;
    VectorOfFloat           vecDistortionBottom;
};

struct PR_COMBINE_IMG_CMD {
    VectorOfMat             vecInputImages;
    int                     nCountOfImgPerFrame;    //The count of images in one frame.
    int                     nCountOfFrameX;
    int                     nCountOfFrameY;
    int                     nOverlapX;
    int                     nOverlapY;
    int                     nCountOfImgPerRow;
    PR_SCAN_IMAGE_DIR       enScanDir;
    PR_COMBINE_IMG_CMD() :
        nCountOfImgPerFrame (0),
        nCountOfFrameX      (0),
        nCountOfFrameY      (0),
        nOverlapX           (0),
        nOverlapY           (0),
        nCountOfImgPerRow   (0),
        enScanDir           (PR_SCAN_IMAGE_DIR::RIGHT_TO_LEFT) {}
};

struct PR_COMBINE_IMG_RPY {
    VisionStatus            enStatus;
    VectorOfMat             vecResultImages;
};

struct PR_COMBINE_IMG_NEW_CMD {
    PR_COMBINE_IMG_NEW_CMD() :
        nCutBorderPixel(5),
        bDrawFrame(false) {}
    VectorOfMat             vecInputImages;
    VectorOfVectorOfPoint   vecVecFrameCtr;
    int                     nCutBorderPixel;
    bool                    bDrawFrame;
};

struct PR_COMBINE_IMG_NEW_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matResultImage;
};

struct PR_THRESHOLD_CMD {
    PR_THRESHOLD_CMD() :
        bDoubleThreshold    (false),
        bInverseResult      (false) {}
    cv::Mat                 matInputImg;
    bool                    bDoubleThreshold;
    bool                    bInverseResult;
    Int32                   nThreshold1;
    Int32                   nThreshold2;    
};

struct PR_THRESHOLD_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matResultImg;
};

struct PR_HEIGHT_TO_GRAY_CMD {
    cv::Mat                 matHeight;
};

struct PR_HEIGHT_TO_GRAY_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matGray;
};

struct PR_LRN_OCV_CMD {
    PR_LRN_OCV_CMD() :
        enDirection (PR_DIRECTION::RIGHT) {}
    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    PR_DIRECTION            enDirection;
    UInt16                  nCharCount;
};

struct PR_LRN_OCV_RPY {
    VisionStatus            enStatus;
    Int32                   nRecordId;
    cv::Mat                 matResultImg;
};

struct PR_OCV_RECORD_INFO {
    cv::Mat                 matTmplImg;
    VectorOfRect            vecCharRects;
};

struct PR_OCV_CMD {
    PR_OCV_CMD() :
        enDirection     (PR_DIRECTION::RIGHT),
        bAcceptReverse  (false),
        fMinMatchScore  (60) {}
    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    PR_DIRECTION            enDirection;
    bool                    bAcceptReverse;
    std::vector<Int32>      vecRecordId;
    float                   fMinMatchScore;
};

struct PR_OCV_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matResultImg;
    float                   fOverallScore;
    VectorOfFloat           vecCharScore;
};

struct PR_READ_2DCODE_CMD {
    PR_READ_2DCODE_CMD() :
        nEdgeThreshold      (30),
        nRemoveNoiseArea    (20) {}
    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    int                     nEdgeThreshold;
    int                     nRemoveNoiseArea;   // The standalone edge which area smaller than this value will be removed.
};

struct PR_READ_2DCODE_RPY {
    VisionStatus            enStatus;
    String                  strReadResult;
    cv::Mat                 matResultImg;
};

struct PR_LRN_SIMILARITY_CMD {
    cv::Mat                 matInputImg;
    cv::Mat                 matMask;
    cv::Rect                rectROI;
};

struct PR_LRN_SIMILARITY_RPY {
    VisionStatus            enStatus;
    Int32                   nRecordId;
    cv::Mat                 matTmpl;
};

struct PR_INSP_SIMILARITY_CMD {
    PR_INSP_SIMILARITY_CMD() :
        fMinSimilarity  (60.f) {}
    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    std::vector<Int32>      vecRecordId;
    float                   fMinSimilarity;
};

struct PR_INSP_SIMILARITY_RPY {
    VisionStatus            enStatus;
    VectorOfFloat           vecSimilarity;
    cv::Mat                 matResultImg;
};

struct PR_TABLE_MAPPING_CMD {
    struct FramePoint {
        cv::Point2f     targetPoint;
        cv::Point2f     actualPoint;
    };

    using VectorOfFramePoint = std::vector<FramePoint>;
    using VectorOfFramePoints = std::vector<VectorOfFramePoint>;

    PR_TABLE_MAPPING_CMD() :
        fBoardPointDist(5.f),
        nBezierRankX(DEFAULT_TABLE_MAPPING_BEZIER_RANK),
        nBezierRankY(DEFAULT_TABLE_MAPPING_BEZIER_RANK),
        fFrameBorderPointWeight(100) {}
    VectorOfFramePoints     vecFramePoints;
    float                   fBoardPointDist; // The physical calibration point distance on the chess board. Unit: mm
    int                     nBezierRankX;
    int                     nBezierRankY;
    float                   fFrameBorderPointWeight;
};

struct PR_TABLE_MAPPING_RPY {
    cv::Mat                 matXOffsetParam;
    cv::Mat                 matYOffsetParam;
    float                   a;
    float                   b;
    float                   c;
    float                   fMinX;
    float                   fMaxX;
    float                   fMinY;
    float                   fMaxY;
    VisionStatus            enStatus;
};

struct PR_CALC_TABLE_OFFSET_CMD {
    PR_CALC_TABLE_OFFSET_CMD() :
        nBezierRankX(DEFAULT_TABLE_MAPPING_BEZIER_RANK),
        nBezierRankY(DEFAULT_TABLE_MAPPING_BEZIER_RANK),
        a(1.f), b(1.f), c(0.f) {}
    cv::Point2f             ptTablePos;
    cv::Mat                 matXOffsetParam;
    cv::Mat                 matYOffsetParam;
    float                   a;
    float                   b;
    float                   c;
    float                   fMinX;
    float                   fMaxX;
    float                   fMinY;
    float                   fMaxY;
    int                     nBezierRankX;
    int                     nBezierRankY;
};

struct PR_CALC_TABLE_OFFSET_RPY {
    float                   fOffsetX;
    float                   fOffsetY;
    VisionStatus            enStatus;
};

// Use the table mapping result to calculate using which restore matrix to restore the image
// Because the table offset too big will cause image like rotated.
struct PR_CALC_RESTORE_IDX_CMD {
    PR_CALC_RESTORE_IDX_CMD() :
        nCenterIdx                  (5),
        fRestoreMatrixAngleInterval (0.5f / 2048.f){}
    cv::Mat                 matXOffsetParam;
    cv::Mat                 matYOffsetParam;
    int                     nBezierRankX;
    int                     nBezierRankY;
    float                   fMinX;
    float                   fMaxX;
    float                   fMinY;
    float                   fMaxY;
    int                     nCenterIdx;
    float                   fRestoreMatrixAngleInterval;
    VectorOfVectorOfPoint2f vecVecFrameCtr;
};

struct PR_CALC_RESTORE_IDX_RPY {
    VisionStatus            enStatus;
    VectorOfVectorOfInt     vecVecRestoreIndex;
};

}
}
#endif /*_AOI_VISION_STRUCT_H_*/