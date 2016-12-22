#ifndef _AOI_STRUCT_H_
#define _AOI_STRUCT_H_

#include "VisionType.h"
#include "VisionStatus.h"
#include "opencv2/core/core.hpp"
#include "BaseType.h"

namespace AOI
{
namespace Vision
{

using VectorOfDMatch = std::vector<cv::DMatch>;
using VectorOfVectorOfDMatch = std::vector<VectorOfDMatch>;
using VectorOfKeyPoint =  std::vector<cv::KeyPoint> ;
using VectorOfVectorKeyPoint = std::vector<VectorOfKeyPoint>;
using VectorOfPoint = std::vector<cv::Point2f>;
using VectorOfVectorOfPoint = std::vector<VectorOfPoint>;
using VectorOfRect = std::vector<cv::Rect>;

#define ToInt32(param)      (static_cast<Int32>(param))
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

struct PR_AlignCmd
{
	cv::Rect2f				    rectLrn;
	cv::Mat					    matInput;
	cv::Mat					    matTmpl;
	cv::Point2f				    ptExpectedPos;
	cv::Rect2f				    rectSrchWindow;
	std::vector<cv::KeyPoint>	vecModelKeyPoint;
	cv::Mat					    matModelDescritor;
};

struct PR_AlignRpy
{
	Int16					nStatus;
	cv::Mat					matAffine;
	cv::Point2f				ptObjPos;
	cv::Size2f				szOffset;
	float                   fRotation;
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
    Int32                   nStatus;
    cv::Point2f             ptPos;
    cv::Mat                 matResult;
};

struct PR_FIT_LINE_CMD
{
    cv::Mat                 matInput;
    cv::Rect                rectROI;
    Int32                   nThreshold;
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    float                   fErrTol;
};

struct PR_FIT_LINE_RPY
{
    Int32                   nStatus;
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
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    float                   fErrTol;
};

struct PR_FIT_PARALLEL_LINE_RPY
{
    Int32                   nStatus;
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
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    float                   fErrTol;
};

struct PR_FIT_RECT_RPY
{
    Int32                   nStatus;
    float                   fSlope1;
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
    PR_FIT_CIRCLE_METHOD    enMethod;
    PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod;
    bool                    bAutothreshold;
    Int32                   nThreshold;
    cv::Point2f             ptRangeCtr;
    float                   fRangeInnterRadius;
    float                   fRangeOutterRadius;
    float                   fErrTol;
    int                     nMaxRansacTime;             //Only used when fitting method is ransac.
    int                     nNumOfPtToFinishRansac;     //Only used when fitting method is ransac.
};

struct PR_FIT_CIRCLE_RPY
{
    Int32                   nStatus;
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
    cv::Rect                rectROI;
};

struct PR_OCR_RPY
{
    Int32                   nStatus;
    String                  strResult;
};

}
}
#endif
