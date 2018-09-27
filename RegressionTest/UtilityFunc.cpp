#include "UtilityFunc.h"

namespace AOI
{
namespace Vision
{

VectorOfFloat split ( const std::string &s, char delim ) {
    VectorOfFloat elems;
    std::stringstream ss ( s );
    std::string strItem;
    while( std::getline ( ss, strItem, delim ) ) {
        elems.push_back ( ToFloat ( std::atof ( strItem.c_str() ) ) );
    }
    return elems;
}

VectorOfVectorOfFloat parseData(const std::string &strContent) {
    std::stringstream ss ( strContent );
    std::string strLine;
    VectorOfVectorOfFloat vevVecResult;
    while( std::getline ( ss, strLine ) ) {
        vevVecResult.push_back ( split ( strLine, ',' ) );
    }
    return vevVecResult;
}

VectorOfVectorOfFloat readDataFromFile(const std::string &strFilePath) {
    /* Open input file */
    FILE *fp;
    fopen_s ( &fp, strFilePath.c_str(), "r" );
    if(fp == NULL) {
        return VectorOfVectorOfFloat();
    }
    /* Get file size */
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);

    /* Allocate buffer and read */
    char *buff = new char[size + 1];
    size_t bytes = fread(buff, 1, size, fp);
    buff[bytes] = '\0';
    std::string strContent(buff);
    delete []buff;
    fclose ( fp );
    return parseData ( strContent );
}

void copyVectorOfVectorToMat(const VectorOfVectorOfFloat &vecVecInput, cv::Mat &matOutput) {
    matOutput = cv::Mat ( ToInt32 ( vecVecInput.size() ), ToInt32 ( vecVecInput[0].size() ), CV_32FC1 );
    for ( int row = 0; row < matOutput.rows; ++ row )
    for ( int col = 0; col < matOutput.cols; ++ col ) {
        matOutput.at<float>(row, col) = vecVecInput[row][col];
    }
}

cv::Mat readMatFromCsvFile(const std::string &strFilePath) {
    auto vecVecData = readDataFromFile(strFilePath);
    cv::Mat matOutput;
    copyVectorOfVectorToMat(vecVecData, matOutput);
    return matOutput;
}

}
}