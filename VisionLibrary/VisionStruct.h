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
using VectorOfRect = std::vector<cv::Rect>;
using ListOfPoint = std::list<cv::Point>;
using VectorOfListOfPoint = std::vector<ListOfPoint>;


#define ToInt32(param)      (static_cast<AOI::Int32>(param))
#define ToFloat(param)      (static_cast<float>(param))

struct PR_VERSION_INFO
{
    char                    chArrVersion[100];
};

struct PR_LRN_TMPL_CMD
{
	PR_ALIGN_ALGORITHM      enAlgorithm;
	cv::Mat                 mat;
	cv::Mat                 mask;
	cv::Rect2f              rectLrn;
};

struct PR_LRN_TMPL_RPY
{
	Int16                       nStatus;
	std::vector<cv::KeyPoint>   vecKeyPoint;
	cv::Mat                     matDescritor;
	cv::Mat                     matTmpl;
	cv::Point2f                 ptCenter;
    Int32                       nRecordID;
};

struct PR_SRCH_TMPL_CMD
{
	PR_ALIGN_ALGORITHM     enAlgorithm;
	cv::Rect2f             rectLrn;
	cv::Mat                mat;
	cv::Rect               rectSrchWindow;
	cv::Point2f            ptExpectedPos;
    Int32                  nRecordID;
};

struct PR_SRCH_TMPL_RPY
{
	Int16					nStatus;
	float					fMatchScore;
	cv::Mat					matHomography;
	cv::Point2f				ptObjPos;
	cv::Size2f				szOffset;
	float					fRotation;
};

struct PR_MATCH_TEMPLATE_CMD
{
	cv::Mat                 matInput;
	cv::Mat					matTmpl;
    cv::Rect				rectSrchWindow;
    PR_OBJECT_MOTION        enMotion;
};

struct PR_MATCH_TEMPLATE_RPY
{
	VisionStatus            enStatus;
	cv::Point2f				ptObjPos;
	float                   fRotation;
    cv::Mat					matResult;
};

struct PR_DefectCriteria
{
	PR_DEFECT_ATTRIBUTE		enAttribute;
	Byte					ubContrast;
	float					fArea;	// For general defect.
	float					fLength;// For line defect only. If != 0, will regard as line defect detecton and rAreaThreshold will be ignored
	float					fWidth;	// For noise removal. Note: unit is in 8192 coord sys (in old convention is phy coord sys, which is no good)
};

struct PR_INSP_SURFACE_CMD
{
	cv::Mat					matInsp;	
	cv::Mat					matTmpl;
    cv::Mat					matMask;
	cv::Rect2f				rectLrn;
	cv::Point2f				ptObjPos;
	float					fRotation;
	UInt16					u16NumOfDefectCriteria;
	PR_DefectCriteria		astDefectCriteria[MAX_NUM_OF_DEFECT_CRITERIA];
};

struct PR_Defect
{
	PR_DEFECT_TYPE		enType;
	float				fArea;
	float				fRadius;
	float				fLength;
	Int16				n16Constrast;
};

struct PR_INSP_SURFACE_RPY
{
	Int16				nStatus;
	Int16				n16NDefect;
	PR_Defect			astDefect[MAX_NUM_OF_DEFECT_RESULT];
};

template<typename _Tp> class PR_Line_
{
public:
	PR_Line_()
	{
		pt1.x = 0;
		pt1.y = 0;
		pt2.x = 0;
		pt2.y = 0;
	}
	PR_Line_(cv::Point_<_Tp> inPt1, cv::Point_<_Tp> inPt2 ) : pt1( inPt1 ), pt2 ( inPt2 )
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

struct PR_INSP_DEVICE_ITEM
{
	bool        bCheckMissed;			//check if the device is missed
	bool        bCheckShift;			//check if the device shift exceed tolerance
    bool        bCheckRotation;         //check if rotation exceed tolerance
    bool        bCheckScale;
};

struct PR_SINGLE_DEVICE_INFO
{
    cv::Point2f             stCtrPos;
    cv::Size2f              stSize;
    cv::Rect2f              rectSrchWindow;
    PR_INSP_DEVICE_ITEM     stInspItem;
    Int16                   nCriteriaNo;
};

struct PR_INSP_DEVICE_CRITERIA
{
	float                   fMaxOffsetX; 
	float                   fMaxOffsetY;
    float                   fMaxRotate;
    float                   fMaxScale;
    float                   fMinScale;
};

struct PR_LRN_DEVICE_CMD
{
    cv::Mat                 matInput;
    cv::Rect2f              rectDevice;
    bool                    bAutoThreshold;
    Int16                   nElectrodeThreshold;
};

struct PR_LRN_DEVICE_RPY
{
    Int32                   nStatus;
    Int32                   nRecordID;
    cv::Size2f              sizeDevice;
    Int16                   nElectrodeThreshold;
};

struct PR_INSP_DEVICE_CMD
{
    cv::Mat                 matInput;
    Int32                   nRecordID;
    Int32                   nDeviceCount;
    Int16                   nElectrodeThreshold;
    PR_SINGLE_DEVICE_INFO   astDeviceInfo[PR_MAX_DEVICE_COUNT];
    PR_INSP_DEVICE_CRITERIA astCriteria[PR_MAX_CRITERIA_COUNT];
};

struct PR_DEVICE_INSP_RESULT
{
    Int32                   nStatus;
    cv::Point2f             ptPos;
    float                   fRotation;    
    float                   fOffsetX;
    float                   fOffsetY;
    float                   fScale;
};

struct PR_INSP_DEVICE_RPY
{
    Int32                   nStatus;
    Int32                   nDeviceCount;
    PR_DEVICE_INSP_RESULT   astDeviceResult[PR_MAX_DEVICE_COUNT];
};

/******************************************
* End of Device Inspection Section
******************************************/

struct PR_SRCH_FIDUCIAL_MARK_CMD
{
    cv::Mat                 matInput;
    cv::Rect                rectSrchRange;
    PR_FIDUCIAL_MARK_TYPE   enType;
    float                   fSize;
    float                   fMargin;
};

struct PR_SRCH_FIDUCIAL_MARK_RPY
{
    VisionStatus            enStatus;
    cv::Point2f             ptPos;
    cv::Mat                 matResult;
};

//Fit line is for accurately fit line in the preprocessed image, the line should be obvious compared to the background.
struct PR_FIT_LINE_CMD
{
    cv::Mat                 matInput;
    cv::Mat                 matMask;
    cv::Rect                rectROI;
    bool                    bPreprocessed;
    Int32                   nThreshold;
    PR_OBJECT_ATTRIBUTE     enAttribute;
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    float                   fErrTol;
};

struct PR_FIT_LINE_RPY
{
    VisionStatus            enStatus;    
    bool                    bReversedFit;   //If it is true, then the result is x = fSlope * y + fIntercept. Otherwise the line is y = fSlope * x + fIntercept.
    float                   fSlope;
    float                   fIntercept;
    PR_Line2f               stLine;
    cv::Mat                 matResult;
};

//Detect line is find the lines in the image.
struct PR_DETECT_LINE_CMD
{
    cv::Mat                 matInput;
    cv::Mat                 matMask;
    cv::Rect                rectROI;
    PR_DETECT_LINE_DIR      enDetectDir;
};

struct PR_DETECT_LINE_RPY
{
    VisionStatus            enStatus;    
    bool                    bReversedFit;   //If it is true, then the result is x = fSlope * y + fIntercept. Otherwise the line is y = fSlope * x + fIntercept.
    float                   fSlope;
    float                   fIntercept;
    PR_Line2f               stLine;
    cv::Mat                 matResult;
};

struct PR_FIT_PARALLEL_LINE_CMD
{
    cv::Mat                 matInput;
    cv::Rect                rectArrROI[2];
    Int32                   nThreshold;
    PR_OBJECT_ATTRIBUTE     enAttribute;
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;    
    float                   fErrTol;
};

struct PR_FIT_PARALLEL_LINE_RPY
{
    Int32                   nStatus;
    bool                    bReversedFit;
    float                   fSlope;
    float                   fIntercept1;
    float                   fIntercept2;
    PR_Line2f               stLine1;
    PR_Line2f               stLine2;
    cv::Mat                 matResult;
};

//The rectArrROI is the ROI of the rect edge. the 1st and 2nd should be parallel, and the 3rd and 4th is parallel.
struct PR_FIT_RECT_CMD
{
    cv::Mat                 matInput;
    cv::Rect                rectArrROI[PR_RECT_EDGE_COUNT];
    Int32                   nThreshold;
    PR_OBJECT_ATTRIBUTE     enAttribute;
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    float                   fErrTol;
};

struct PR_FIT_RECT_RPY
{
    VisionStatus            enStatus;
    bool                    bLineOneReversedFit;
    float                   fSlope1;
    bool                    bLineTwoReversedFit;
    float                   fSlope2;
    float                   fArrIntercept[PR_RECT_EDGE_COUNT];
    PR_Line2f               fArrLine[PR_RECT_EDGE_COUNT];
    cv::Mat                 matResult;
};

struct PR_FIND_EDGE_CMD
{
    cv::Mat                 matInput;
    cv::Rect                rectROI;
    bool                    bAutothreshold;
    Int32                   nThreshold;
    PR_EDGE_DIRECTION       enDirection;
    float                   fMinLength;
};

struct PR_FIND_EDGE_RPY
{
    Int32                   nStatus;
    Int32                   nEdgeCount;
};

struct PR_FIT_CIRCLE_CMD
{
    cv::Mat                 matInput;    
    cv::Mat                 matMask;
    cv::Rect                rectROI;
    PR_FIT_CIRCLE_METHOD    enMethod;
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    bool                    bPreprocessed;
    bool                    bAutoThreshold;
    Int32                   nThreshold;
    PR_OBJECT_ATTRIBUTE     enAttribute;
    float                   fErrTol;
    int                     nMaxRansacTime;             //Only used when fitting method is ransac.
    int                     nNumOfPtToFinishRansac;     //Only used when fitting method is ransac.
};

struct PR_FIT_CIRCLE_RPY
{
    VisionStatus            enStatus;
    cv::Point2f             ptCircleCtr;
    float                   fRadius;
    cv::Mat                 matResult;
};

struct PR_GET_ERROR_STR_RPY
{
	char				    achErrorStr[PR_MAX_ERR_STR_LEN];
	PR_STATUS_ERROR_LEVEL	enErrorLevel;
};

struct PR_OCR_CMD
{
    cv::Mat                 matInput;
    PR_DIRECTION            enDirection;
    cv::Rect                rectROI;
};

struct PR_OCR_RPY
{
    VisionStatus            enStatus;
    String                  strResult;
};

struct PR_POINT_LINE_DISTANCE_CMD
{
    cv::Point2f             ptInput;
    bool                    bReversedFit;
    float                   fSlope;
    float                   fIntercept;
};

struct PR_POINT_LINE_DISTANCE_RPY
{
    float                   fDistance;
};

struct PR_RGB_RATIO
{
    PR_RGB_RATIO() : fRatioR(0.299f), fRatioG(0.587f), fRatioB(0.114f) {}
    PR_RGB_RATIO(float fRatioR, float fRatioG, float fRatioB ) : fRatioR(fRatioR), fRatioG(fRatioG), fRatioB(fRatioB)   {}
    float                   fRatioR;
    float                   fRatioG;
    float                   fRatioB;
};

struct PR_COLOR_TO_GRAY_CMD
{
    cv::Mat                 matInput;
    PR_RGB_RATIO            stRatio;
};

struct PR_COLOR_TO_GRAY_RPY
{
    VisionStatus            enStatus;
    cv::Mat                 matResult;
};

struct PR_FILTER_CMD
{
    cv::Mat                 matInput;
    cv::Rect                rectROI;
    PR_FILTER_TYPE          enType;
    cv::Size                szKernel;
    double                  dSigmaX;        //Only used when filter type is Guassian.
    double                  dSigmaY;        //Only used when filter type is Guassian.
    Int16                   nDiameter;      //Only used when filter type is Bilaterial.
    double                  dSigmaColor;    //Only used when filter type is Bilaterial.
    double                  dSigmaSpace;    //Only used when filter type is Bilaterial.
};

struct PR_FILTER_RPY
{
    VisionStatus            enStatus;
    cv::Mat                 matResult;
};

struct PR_AUTO_THRESHOLD_CMD
{
    cv::Mat                 matInput;
    cv::Rect                rectROI;
    Int16                   nThresholdNum;
};

struct PR_AUTO_THRESHOLD_RPY
{
    VisionStatus            enStatus;
    std::vector<Int16>      vecThreshold;
};

//CC is acronym of of connected components.
struct PR_REMOVE_CC_CMD
{
    cv::Mat                 matInput;
    cv::Rect                rectROI;
    Int16                   nConnectivity;      //connect by 4 or 8 points.
    PR_COMPARE_TYPE         enCompareType;
    float                   fAreaThreshold;    
};

struct PR_REMOVE_CC_RPY
{
    VisionStatus            enStatus;
    Int32                   nTotalCC;
    Int32                   nRemovedCC;
    cv::Mat                 matResult;
};

struct PR_DETECT_EDGE_CMD
{
    cv::Mat                 matInput;
    cv::Rect                rectROI;
    Int16                   nThreshold1;
    Int16                   nThreshold2;
    Int16                   nApertureSize;
};

struct PR_DETECT_EDGE_RPY
{
    VisionStatus            enStatus;
    cv::Mat                 matResult;
};

struct PR_CIRCLE_ROUNDNESS_CMD
{
    cv::Mat                 matInput;
    cv::Mat                 matMask;
    cv::Rect                rectROI;
};

struct PR_CIRCLE_ROUNDNESS_RPY
{
    VisionStatus            enStatus;
    float                   fRoundness;
    cv::Point2f             ptCircleCtr;
    float                   fRadius;
    cv::Mat                 matResult;
};

struct PR_FILL_HOLE_CMD
{
    cv::Mat                 matInput;
    cv::Rect                rectROI;
    PR_OBJECT_ATTRIBUTE     enAttribute;
    PR_FILL_HOLE_METHOD     enMethod;
    cv::MorphShapes         enMorphShape;
    cv::Size                szMorphKernel;
    Int16                   nMorphIteration;
};

struct PR_FILL_HOLE_RPY
{
    VisionStatus            enStatus;
    cv::Mat                 matResult;
};

}
}
#endif
