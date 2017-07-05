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

#define ToInt32(param)      (static_cast<AOI::Int32>(param))
#define ToFloat(param)      (static_cast<float>(param))

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

struct PR_MATCH_TEMPLATE_CMD {
    cv::Mat                 matInputImg;
    cv::Mat                 matTmpl;
    cv::Rect                rectSrchWindow;
    PR_OBJECT_MOTION        enMotion;
};

struct PR_MATCH_TEMPLATE_RPY {
    VisionStatus            enStatus;
    cv::Point2f             ptObjPos;
    float                   fRotation;
    float                   fMatchScore;
    cv::Mat                 matResultImg;
};

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
    cv::Rect                rectSrchRange;
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

struct PR_DETECT_EDGE_CMD {
    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    Int16                   nThreshold1;
    Int16                   nThreshold2;
    Int16                   nApertureSize;
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
        Int16               nStart;
        Int16               nEnd;
    };
    struct COLOR_RANGE {
        Int16               nStartB;    //Blue range start.
        Int16               nEndB;      //Blue range end.
        Int16               nStartG;    //Green range start.
        Int16               nEndG;      //Green range end.
        Int16               nStartR;    //Red range start.
        Int16               nEndR;      //Red range end.
    };
    struct RATIO_MODE_CRITERIA {
        float               fMaxRatio;
        float               fMinRatio;
    };
    struct BLOB_MODE_CRITERIA {
        struct ADVANCED_CRITERIA {
            float           fMaxLengthWidthRatio;
            float           fMinLengthWidthRatio;
            float           fMaxCircularity;
            float           fMinCircularity;
        };
        float               fMaxArea;
        float               fMinArea;
        Int16               nMaxBlobCount;
        Int16               nMinBlobCount;
        bool                bEnableAdvancedCriteria;
        ADVANCED_CRITERIA   stAdvancedCriteria;
    };
    cv::Mat                 matInputImg;
    cv::Rect                rectROI;
    PR_IMG_SEGMENT_METHOD   enSegmentMethod;
    GRAY_SCALE_RANGE        stGrayScaleRange;   //It is used when enSegmentMethod is PR_IMG_SEGMENT_METHOD::GRAY_SCALE_RANGE.
    COLOR_RANGE             stColorRange;       //It is used when enSegmentMethod is PR_IMG_SEGMENT_METHOD::COLOR_RANGE.
    PR_INSP_HOLE_MODE       enInspMode;
    RATIO_MODE_CRITERIA     stRatioModeCriteria;
    BLOB_MODE_CRITERIA      stBlobModeCriteria;
};

struct PR_INSP_HOLE_RPY {
    VisionStatus            enStatus;
    cv::Mat                 matResultImg;
};
/******************************************
* End of Inspect Hole Section *
******************************************/

}
}
#endif
