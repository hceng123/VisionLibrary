#ifndef _AOI_STRUCT_H_
#define _AOI_STRUCT_H_

#include "VisionType.h"
#include "VisionStatus.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "BaseType.h"
#include <list>

namespace AOI
{
namespace Vision
{

using VectorOfDMatch = std::vector<cv::DMatch>;
using VectorOfVectorOfDMatch = std::vector<VectorOfDMatch>;
using VectorOfKeyPoint =  std::vector<cv::KeyPoint> ;
using VectorOfVectorKeyPoint = std::vector<VectorOfKeyPoint>;
using VectorOfPoint = std::vector<cv::Point>;
using VectorOfVectorOfPoint = std::vector<VectorOfPoint>;
using VectorOfPoint2f = std::vector<cv::Point2f>;
using VectorOfPoint3f = std::vector<cv::Point3f>;
using VectorOfVectorOfPoint2f = std::vector<VectorOfPoint2f>;
using VectorOfRect = std::vector<cv::Rect>;
using ListOfPoint = std::list<cv::Point>;
using VectorOfListOfPoint = std::vector<ListOfPoint>;
using VectorOfSize2f = std::vector<cv::Size2f>;
using VectorOfMat = std::vector<cv::Mat>;
using VectorOfFloat = std::vector<float>;
using VectorOfVectorOfFloat = std::vector<VectorOfFloat>;

template <typename Tp> inline Int32 ToInt32(Tp param) { return static_cast<Int32>(param); }
template <typename Tp> inline Int16 ToInt16(Tp param) { return static_cast<Int16>(param); }
template <typename Tp> inline float ToFloat(Tp param) { return static_cast<float>(param); }

struct PR_VERSION_INFO {
    char                    chArrVersion[100];
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
        enAlgorithm (PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF),
        nRecordId(-1),
        bSubPixelRefine(false),
        enMotion (PR_OBJECT_MOTION::TRANSLATION) {}
    cv::Mat                 matInputImg;
    PR_MATCH_TMPL_ALGORITHM enAlgorithm;
    Int32                   nRecordId;
    cv::Rect                rectSrchWindow;
    bool                    bSubPixelRefine;
    PR_OBJECT_MOTION        enMotion;
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
    float                   fArea;	// For general defect.
    float                   fLength;// For line defect only. If != 0, will regard as line defect detecton and rAreaThreshold will be ignored
    float                   fWidth;	// For noise removal. Note: unit is in 8192 coord sys (in old convention is phy coord sys, which is no good)
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

struct PR_Defect {
    PR_DEFECT_TYPE      enType;
    float               fArea;
    float               fRadius;
    float               fLength;
    Int16               n16Constrast;
};

struct PR_INSP_SURFACE_RPY {
    Int16               nStatus;
    Int16               n16NDefect;
    PR_Defect           astDefect[MAX_NUM_OF_DEFECT_RESULT];
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
    cv::Mat                 matInputImg;
    cv::Rect                rectSrchWindow;
    PR_FIDUCIAL_MARK_TYPE   enType;
    float                   fSize;
    float                   fMargin;
};

struct PR_SRCH_FIDUCIAL_MARK_RPY {
    VisionStatus            enStatus;
    cv::Point2f             ptPos;
    float                   fScore;
    cv::Mat                 matResultImg;
};

//Fit line is for accurately fit line in the preprocessed image, the line should be obvious compared to the background.
struct PR_FIT_LINE_CMD {
    PR_FIT_LINE_CMD() : enMethod ( PR_FIT_METHOD::LEAST_SQUARE_REFINE ), bPreprocessed ( false ), nMaxRansacTime(20) {}
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

// Detect line is find the lines in the image.
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

struct PR_CALIPER_CMD {
    PR_CALIPER_CMD() :
        enAlgorithm         ( PR_CALIPER_ALGORITHM::PROJECTION ),
        bCheckLinerity      (false),
        fPointMaxOffset     (0.f),
        fMinLinerity        (0.f),
        bCheckAngle         (false),
        fExpectedAngle      (0.f),
        fAngleDiffTolerance (0.f) {}
    cv::Mat                 matInputImg;
    cv::Mat                 matMask;
    cv::RotatedRect         rectRotatedROI;
    PR_CALIPER_ALGORITHM    enAlgorithm;
    PR_CALIPER_DIR          enDetectDir;
    bool                    bCheckLinerity;
    float                   fPointMaxOffset;
    float                   fMinLinerity;
    bool                    bCheckAngle;
    float                   fExpectedAngle;
    float                   fAngleDiffTolerance;
};

struct PR_CALIPER_RPY {
    VisionStatus            enStatus;    
    bool                    bReversedFit;   //If it is true, then the result is x = fSlope * y + fIntercept. Otherwise the line is y = fSlope * x + fIntercept.
    float                   fSlope;
    float                   fIntercept;
    PR_Line2f               stLine;
    bool                    bLinerityCheckPass;
    float                   fLinerity;
    bool                    bAngleCheckPass;
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

struct PR_FIT_CIRCLE_CMD {
    cv::Mat                 matInputImg;    
    cv::Mat                 matMask;
    cv::Rect                rectROI;
    PR_FIT_METHOD           enMethod;
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    bool                    bPreprocessed;
    bool                    bAutoThreshold;
    Int32                   nThreshold;
    PR_OBJECT_ATTRIBUTE     enAttribute;
    float                   fErrTol;
    int                     nMaxRansacTime;             //Only used when fitting method is ransac.
    int                     nNumOfPtToFinishRansac;     //Only used when fitting method is ransac.
};

struct PR_FIT_CIRCLE_RPY {
    VisionStatus            enStatus;
    cv::Point2f             ptCircleCtr;
    float                   fRadius;
    cv::Mat                 matResultImg;
};

struct PR_GET_ERROR_INFO_RPY {
    PR_STATUS_ERROR_LEVEL	enErrorLevel;
	char				    achErrorStr[PR_MAX_ERR_STR_LEN];
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
    cv::Point2f             ptInput;
    bool                    bReversedFit;
    float                   fSlope;
    float                   fIntercept;
};

struct PR_POINT_LINE_DISTANCE_RPY {
    float                   fDistance;
};

struct PR_RGB_RATIO {
    PR_RGB_RATIO() : fRatioR(0.299f), fRatioG(0.587f), fRatioB(0.114f) {}
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
    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    cv::Point               ptPick;
    Int16                   nColorDiff;
    Int16                   nGrayDiff;
};

struct PR_PICK_COLOR_RPY {
    VisionStatus            enStatus;
    UInt32                  nPickPointCount;
    cv::Mat                 matResultImg;
};

struct PR_CALIBRATE_CAMERA_CMD {
    PR_CALIBRATE_CAMERA_CMD() : fPatternDist(1.f), fMinTmplMatchScore(90.f) {}
    cv::Mat                 matInputImg;
    cv::Size                szBoardPattern; // Number of corners per chessboard row and col. szBoardPattern = cv::Size(points_per_row, points_per_col) = cv::Size(columns, rows).
    float                   fPatternDist;   //The real chess board corner to corner distance. Unit: mm.
    float                   fMinTmplMatchScore;
};

struct PR_CALIBRATE_CAMERA_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matIntrinsicMatrix; //type: CV_64FC1.
    cv::Mat                 matExtrinsicMatrix; //type: CV_64FC1.
    cv::Mat                 matDistCoeffs;      //type: CV_64FC1.
    double                  dResolutionX;       //Unit: um/pixel.
    double                  dResolutionY;       //Unit: um/pixel.
    std::vector<cv::Mat>    vecMatRestoreImage; //The remap matrix to restore image. vector size is 2, the matrix dimension is same as input image.
    //Intermediate result.
    cv::Mat                 matCornerPointsImg;
    VectorOfPoint2f         vecImagePoints;
    VectorOfPoint3f         vecObjectPoints;
    cv::Mat                 matInitialIntrinsicMatrix;  //type: CV_64FC1.
    cv::Mat                 matInitialExtrinsicMatrix;  //type: CV_64FC1.
};

struct PR_RESTORE_IMG_CMD {
    cv::Mat                 matInputImg;
    std::vector<cv::Mat>    vecMatRestoreImage;
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
    std::vector<cv::Mat>    vecMatRestoreImage;
};

struct PR_AUTO_LOCATE_LEAD_CMD {
    cv::Mat                 matInputImg;
    cv::Rect                rectSrchWindow;
    cv::Rect                rectChipBody;
    PR_OBJECT_ATTRIBUTE     enLeadAttribute;
};

struct PR_AUTO_LOCATE_LEAD_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matResultImg;
    VectorOfRect            vecLeadLocation;
};

struct PR_INSP_BRIDGE_CMD {
    struct INNER_INSP_CRITERIA {
        INNER_INSP_CRITERIA() : fMaxLengthX(0.f), fMaxLengthY(0.f) {}
        float                   fMaxLengthX;
        float                   fMaxLengthY;
    };
    using INSP_DIRECTION_VECTOR = std::vector<PR_INSP_BRIDGE_DIRECTION>;
    struct INSP_ITEM {
        cv::Rect                rectInnerWindow;
        cv::Rect                rectOuterWindow;
        PR_INSP_BRIDGE_MODE     enMode;
        INSP_DIRECTION_VECTOR   vecOuterInspDirection;
        INNER_INSP_CRITERIA     stInnerInspCriteria;
    };
    using INSP_ITEM_VECTOR = std::vector<INSP_ITEM>;
    cv::Mat                 matInputImg;
    INSP_ITEM_VECTOR        vecInspItems;
};

struct PR_INSP_BRIDGE_RPY {
    struct ITEM_RESULT {
        bool                bWithBridge;
        VectorOfRect        vecBridgeWindow;
    };
    using ITEM_RESULT_VECTOR = std::vector<ITEM_RESULT>;
    VisionStatus            enStatus;
    cv::Mat                 matResultImg;
    ITEM_RESULT_VECTOR      vecInspResults;
};

/******************************************
* Chip Inspection Section *
******************************************/
struct PR_LRN_CHIP_CMD {
    PR_LRN_CHIP_CMD() : enInspMode(PR_INSP_CHIP_MODE::HEAD), bAutoThreshold(true), nThreshold(0) {}
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
    PR_INSP_CONTOUR_CMD() : nRecordId (0), nDefectThreshold(30), fMinDefectArea (100.f), fDefectInnerLengthTol(20), fDefectOuterLengthTol(20), fInnerMaskDepth(5), fOuterMaskDepth(5) {}
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

    cv::Mat                 matInputImg;
    cv::Mat                 matMask;
    cv::Rect                rectROI;
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
/******************************************
* End of Inspect Lead Section *
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
    PR_CALIB_3D_BASE_CMD() : bEnableGaussianFilter(true), bReverseSeq(true), fRemoveHarmonicWaveK(0.f) {}
    VectorOfMat             vecInputImgs;
    bool                    bEnableGaussianFilter;
    bool                    bReverseSeq;            //Change the image sequence.
    float                   fRemoveHarmonicWaveK;   //The factor to remove the harmonic wave in the thick stripe. If it is 0, then this procedure will be skipped.
};

struct PR_CALIB_3D_BASE_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matThickToThinStripeK;  //The factor between thick stripe and thin stripe.
    cv::Mat                 matBaseSurfaceParam;    //The bezier parameters to form the base surface.
    float                   fBaseStartAvgPhase;     //The average phase of the top-left corner of the thick stripe. It is used to constrain the object's phase within -pi~pi of base phase.
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
        fRemoveHarmonicWaveK(0.f),
        fBaseStartAvgPhase(0),
        fMinIntensityDiff(3.f),
        fMinAvgIntensity(3.f),
        nBlockStepCount(4),
        fBlockStepHeight(1.f),        
        nResultImgGridRow(8),
        nResultImgGridCol(8),
        szMeasureWinSize (40, 40) {}
    VectorOfMat             vecInputImgs;
    bool                    bEnableGaussianFilter;
    bool                    bReverseSeq;            //Change the image sequence.
    bool                    bReverseHeight;         //The calibration base is align to the top of calibration block.
    float                   fRemoveHarmonicWaveK;   //The factor to remove the harmonic wave in the thick stripe. If it is 0, then this procedure will be skipped.
    float                   fMinIntensityDiff;      //In a group of 4 images, if a pixel's max and min intensity less than this value, this pixel will be discarded.
    float                   fMinAvgIntensity;       //In a group of 4 images, if a pixel's average intensity less than this value, this pixel will be discarded.
    cv::Mat                 matThickToThinStripeK;  //The factor between thick stripe and thin stripe.
    cv::Mat                 matBaseSurface;
    float                   fBaseStartAvgPhase;     //The average phase of the top-left corner of the thick stripe. It is used to constrain the object's phase within -pi~pi of base phase.
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
    cv::Mat                 matDivideStepResultImg; //Use auto threshold to divide each step of the phase image. This result image can show to user confirm if the auto threshold is working correctly.
    cv::Mat                 matResultImg;
};

//Combine negative and positive calibration result to a single matPhaseToHeightK.
struct PR_COMB_3D_CALIB_CMD {
    PR_COMB_3D_CALIB_CMD() :
        fBlockStepHeight(1.f),
        nImageRows(2048),
        nImageCols(2040) {}
    VectorOfVectorOfFloat   vecVecStepPhasePos; //The phase of 4 corners and the center of the positive calibration. The step on top of base plane.
    VectorOfVectorOfFloat   vecVecStepPhaseNeg; //The phase of 4 corners and the center of the negative calibration. The step below the base plane.
    int                     nImageRows;         //The rows of camera image.
    int                     nImageCols;         //The cols of camera image.
    float                   fBlockStepHeight;   //The height of each step, unit mm. So the total block height is nBlockStepCount x fBlockStepHeight.
};

struct PR_COMB_3D_CALIB_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matPhaseToHeightK;  //The factor to convert phase to height.
    VectorOfVectorOfFloat   vecVecStepPhaseDiff;
};

struct PR_CALC_3D_HEIGHT_CMD {
    PR_CALC_3D_HEIGHT_CMD() :
        bEnableGaussianFilter(true),
        bReverseSeq(true),
        fRemoveHarmonicWaveK(0.f),
        fMinIntensityDiff(3.f),
        fMinAvgIntensity(3.f) {}
    VectorOfMat             vecInputImgs;
    bool                    bEnableGaussianFilter;
    bool                    bReverseSeq;            //Change the image sequence.
    float                   fRemoveHarmonicWaveK;   //The factor to remove the harmonic wave in the thick stripe. If it is 0, then this procedure will be skipped.
    float                   fMinIntensityDiff;      //In a group of 4 images, if a pixel's max and min intensity less than this value, this pixel will be discarded.
    float                   fMinAvgIntensity;       //In a group of 4 images, if a pixel's average intensity less than this value, this pixel will be discarded.
    cv::Mat                 matThickToThinStripeK;  //The factor between thick stripe and thin stripe.
    cv::Mat                 matBaseSurface;
    float                   fBaseStartAvgPhase;     //The average phase of the top-left corner of the thick stripe. It is used to constrain the object's phase within -pi~pi of base phase.
    cv::Mat                 matPhaseToHeightK;      //The factor to convert phase to height.
};

struct PR_CALC_3D_HEIGHT_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matPhase;
    cv::Mat                 matHeight;
};

struct PR_CALC_3D_HEIGHT_DIFF_CMD {
    PR_CALC_3D_HEIGHT_DIFF_CMD() :
        fEffectHRatioStart(0.3f),
        fEffectHRatioEnd(0.7f) {}
    cv::Mat                 matHeight;
    VectorOfRect            vecRectBases;           //One or more bases as the reference surface.
    cv::Rect                rectROI;                //The ROI to measure height difference to base.
    float                   fEffectHRatioStart;     //If fEffectHRatioStart = 0.3, the lower 30% points in the window will be removed for fitting.
    float                   fEffectHRatioEnd;       //If fEffectHRatioEnd = 0.7, the upper 30% of points in the window will be removed for fitting.
};

struct PR_CALC_3D_HEIGHT_DIFF_RPY {
    VisionStatus            enStatus;
    float                   fHeightDiff;
};

//Calculate the optics modulation transfer function.
struct PR_CALC_MTF_CMD {
    VectorOfMat             vecInputImgs;
    float                   fMagnitudeOfDLP;        //The setted magnitude of DLP. The captured image magnitude divide the setted maganitude is the MTF result.
};

struct PR_CALC_MTF_RPY {
    VisionStatus            enStatus;
    VectorOfVectorOfFloat   vevVecAbsMtfH;          //The absolute modulation transfer function in horizontal direction.
    VectorOfVectorOfFloat   vevVecRelMtfH;          //The relative modulation transfer function in horizontal direction.
    VectorOfVectorOfFloat   vevVecAbsMtfV;          //The absolute modulation transfer function in vertical direction.
    VectorOfVectorOfFloat   vevVecRelMtfV;          //The relative modulation transfer function in vertical direction.
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

}
}
#endif /*_AOI_STRUCT_H_*/