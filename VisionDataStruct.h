#ifndef _AOI_DATASTRUCT_H_
#define _AOI_DATASTRUCT_H_

#include "opencv2/core/core.hpp"

using namespace cv;

namespace AOI
{
namespace Vision
{
    enum ALIGN_ALGORITHM
    {
          ALIGN_ALGORITHM_SIFT,
          ALIGN_ALGORITHM_SURF,
    };

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
         Int16                  nStatus;
         cv::Mat                matHomography;
         cv::Point2f            ptObjPos;
         cv::Size2f             szOffset;
         float                  fRotation;
     };

     struct PR_AlignCmd
     {
         cv::Mat                matInput;
         cv::Mat                matTmpl;
         cv::Point2f            ptOriginalPos;
         cv::Rect               rectSrchWindow;
         std::vector<KeyPoint>  vecModelKeyPoint;
         cv::Mat                matModelDescritor;
     };

     struct PR_AlignRpy
     {
         Int16                  nStatus;
         cv::Mat                matAffine;
         cv::Point2f            ptObjPos;
         cv::Size2f             szOffset;
         float                  fRotation;
     };
}
}
#endif
