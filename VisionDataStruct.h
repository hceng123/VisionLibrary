#ifndef _AOI_DATASTRUCT_H_
#define _AOI_DATASTRUCT_H_

#include "prtype.h"
#include "opencv2/core/core.hpp"
#include "BaseType.h"

using namespace cv;

namespace AOI
{
namespace Vision
{

#define MAX_NUM_OF_DEFECT_CRITERIA				(5)
#define MAX_NUM_OF_DEFECT_RESULT				(20)
#define PARALLEL_LINE_SLOPE_DIFF_LMT			(0.1)
#define PARALLEL_LINE_MERGE_DIST_LMT			(20)

struct PR_LearnTmplCmd
{
	Int16                   nAlgorithm;
	cv::Mat                 mat;
	cv::Mat                 mask;
	cv::Rect2f              rectLrn;

};

struct PR_LearnTmplRpy
{
	Int16                  nStatus;
	std::vector<KeyPoint>  vecKeyPoint;
	cv::Mat                matDescritor;
	cv::Mat                matTmpl;
	Point2f                ptCenter;
};

struct PR_FindObjCmd
{
	Int16                  nAlgorithm;
	cv::Rect2f             rectLrn;
	cv::Mat                mat;
	cv::Rect               rectSrchWindow;
	std::vector<KeyPoint>  vecModelKeyPoint;
	cv::Mat                matModelDescritor;
	cv::Point2f            ptExpectedPos;
};

struct PR_FindObjRpy
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
	cv::Rect2f				rectLrn;
	cv::Mat					matInput;
	cv::Mat					matTmpl;
	cv::Point2f				ptExpectedPos;
	cv::Rect2f				rectSrchWindow;
	std::vector<KeyPoint>	vecModelKeyPoint;
	cv::Mat					matModelDescritor;
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

struct PR_InspCmd
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

struct PR_InspRpy
{
	Int16				nStatus;
	Int16				n16NDefect;
	PR_Defect			astDefect[MAX_NUM_OF_DEFECT_RESULT];
};

//struct PR_Line
//{
//	Point pt1;
//	Point pt2;
//};

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
	PR_Line_(Point_<_Tp> inPt1, Point_<_Tp> inPt2 ) : pt1( inPt1 ), pt2 ( inPt2)
	{
	}
	Point_<_Tp> pt1;
	Point_<_Tp> pt2;
};

using PR_Line = PR_Line_<int>;
using PR_Line2f = PR_Line_<float>;

}
}
#endif
