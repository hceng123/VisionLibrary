#ifndef _AOI_DATASTRUCT_H_
#define _AOI_DATASTRUCT_H_

#include "opencv2/core/core.hpp"

using namespace cv;

namespace AOI
{
namespace Vision
{

    struct PR_LearnTmplCmd
    {
        cv::Mat                 mat;
        cv::Mat                 mask;
        cv::Rect                rectLrn;
        Int16                   nAlgorithm;
    };

     struct PR_LearnTmplRpy
     {
         Int16                  nStatus;
         std::vector<KeyPoint>  vecKeyPoint;
         cv::Mat                matDescritor;
     };

     struct PR_AlignCmd
     {
         cv::Mat                mat;
         cv::Rect               rectSrchWindow;
         std::vector<KeyPoint>  vecModelKeyPoint;
         cv::Mat                matModelDescritor;
     };

     struct PR_AlignRpy
     {
         Int16                  nStatus;
         cv::Mat                matHomography;
     };
}
}
#endif
