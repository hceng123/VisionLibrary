#ifndef _VISION_REGRESSION_TEST_UTILITY_FUNCTION_H_
#define _VISION_REGRESSION_TEST_UTILITY_FUNCTION_H_

#include "../VisionLibrary/VisionAPI.h"

template<class T>
void printfMat ( const cv::Mat &mat)
{
    for ( short row = 0; row < mat.rows; ++ row )
    {
        for ( short col = 0; col < mat.cols; ++ col )
        {
            printf ("%.1f ", mat.at<T>(row, col) );
        }
        printf("\n");
    }
}

template<class T>
void printfVectorOfVector(const std::vector<std::vector<T>> &vevVecInput)
{
	for (const auto &vecInput : vevVecInput )
	{
		for (const auto value : vecInput )
		{
			printf("%.2f ", value);
		}
		printf("\n");
	}
}

namespace AOI
{
namespace Vision
{

VectorOfFloat split ( const std::string &s, char delim );
VectorOfVectorOfFloat parseData(const std::string &strContent);
VectorOfVectorOfFloat readDataFromFile(const std::string &strFilePath);

}
}
#endif /*_VISION_REGRESSION_TEST_UTILITY_FUNCTION_H_*/