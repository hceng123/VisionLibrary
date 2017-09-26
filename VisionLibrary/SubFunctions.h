#include "VisionHeader.h"
namespace AOI
{
namespace Vision
{

VisionStatus GetErrorInfo(VisionStatus enStatus, PR_GET_ERROR_INFO_RPY *pstRpy);
void InternalTest();

float autoThresholdObjFunction(const VectorOfFloat &vecX, const VectorOfVectorOfFloat &vecVecOmega, const VectorOfVectorOfFloat &vecVecMu);
float matlabObjFunction(const VectorOfFloat &vecThreshold, int num_bins, const cv::Mat &omega, const cv::Mat &mu, float mu_t );
typedef float(*objFuntion)(const VectorOfFloat &vecThreshold, int num_bins, const cv::Mat &omega, const cv::Mat &mu, float mu_t);
int fminsearch (objFuntion          funfcn,
                VectorOfFloat      &vecX,
                int                 num_bins,
                const cv::Mat      &omega,
                const cv::Mat      &mu,
                float               mu_t );
}
}