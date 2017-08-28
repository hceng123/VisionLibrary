#include "Unwrap.h"
#include "opencv2/imgproc.hpp"
#include "CalcUtils.h"
#include <numeric>

namespace AOI
{
namespace Vision
{

Unwrap::Unwrap ()
{
}

Unwrap::~Unwrap ()
{
}

/*static*/ const float Unwrap::GUASSIAN_FILTER_SIGMA = pow ( 20, 0.5);
/*static*/ const float Unwrap::ONE_HALF_CYLE = CV_PI;

cv::Mat Unwrap::getResidualPoint(const cv::Mat &matInput, cv::Mat &matPosPole, cv::Mat &matNegPole)
{
    cv::Mat Kernely = (cv::Mat_<float>(2, 1) << -1, 1);
    cv::Mat Kernelx = (cv::Mat_<float>(1, 2) << -1, 1);
    cv::Mat matDx, matDy, matD1, matD2, matD3, matD4;
    cv::filter2D(matInput, matDx, -1, Kernelx, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
    matDx = cv::Mat(matDx, cv::Rect(1, 0, matDx.cols - 1, matDx.rows));    

    cv::filter2D(matInput, matDy, -1, Kernely, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
    matDy = cv::Mat(matDy, cv::Rect(0, 1, matDy.cols, matDy.rows - 1));

    matD1 =  cv::Mat(matDx, cv::Rect(0, 0, matDx.cols,     matDx.rows - 1)).clone();
    matD2 =  cv::Mat(matDy, cv::Rect(1, 0, matDy.cols - 1, matDy.rows    )).clone();
    matD3 = -cv::Mat(matDx, cv::Rect(0, 1, matDx.cols,     matDx.rows - 1)).clone();
    matD4 = -cv::Mat(matDy, cv::Rect(0, 0, matDy.cols - 1, matDy.rows    )).clone();

    //printf("matD1 result\n");
    //printfMat<float>(matD1);
    //printf("matD2 result\n");
    //printfMat<float>(matD2);
    //printf("matD3 result\n");
    //printfMat<float>(matD3);
    //printf("matD4 result\n");
    //printfMat<float>(matD4);

    matD1 = matD1.reshape(1, 1);
    matD2 = matD2.reshape(1, 1);
    matD3 = matD3.reshape(1, 1);
    matD4 = matD4.reshape(1, 1);

    cv::Mat matD;
    matD.push_back(matD1);
    matD.push_back(matD2);
    matD.push_back(matD3);
    matD.push_back(matD4);

    cv::Mat matPosIndex = cv::Mat::zeros(matD.size(), CV_8UC1);
    cv::Mat matNegIndex = cv::Mat::zeros(matD.size(), CV_8UC1);

    cv::compare ( matD, cv::Scalar::all( 0.99 * CV_PI), matPosIndex, cv::CmpTypes::CMP_GT );
    cv::compare ( matD, cv::Scalar::all(-0.99 * CV_PI), matNegIndex, cv::CmpTypes::CMP_LT );

    cv::Mat matDR1, matDR2;
    cv::subtract ( matD, cv::Scalar::all( 2.f * CV_PI), matDR1, matPosIndex );
    cv::subtract ( matD, cv::Scalar::all(-2.f * CV_PI), matDR2, matNegIndex );

    //printf("matPosIndex result\n");
    //printIntMat<uchar>(matPosIndex);

    //printf("matNegIndex result\n");
    //printIntMat<uchar>(matNegIndex);

    matDR1.copyTo ( matD, matPosIndex );
    matDR2.copyTo ( matD, matNegIndex );

    //printf("matD result\n");
    //printfMat<float>(matD);

    cv::Mat matSum;
    cv::reduce ( matD, matSum, 0, CV_REDUCE_SUM );

    //printf("matSum result\n");
    //printfMat<float>(matSum);

    cv::compare ( matSum, cv::Scalar::all( 0.99 * CV_PI * 2.f), matPosIndex, cv::CmpTypes::CMP_GT );
    cv::compare ( matSum, cv::Scalar::all(-0.99 * CV_PI * 2.f), matNegIndex, cv::CmpTypes::CMP_LT );

    //printf("matPosIndex result\n");
    //printIntMat<uchar>(matPosIndex);

    //printf("matNegIndex result\n");
    //printIntMat<uchar>(matNegIndex);

    int nSizeOfResultMat = ( matInput.rows - 1 ) * ( matInput.cols - 1 );
    cv::Mat matResult = cv::Mat::zeros ( 1, nSizeOfResultMat, CV_8SC1 );
    matResult.setTo( cv::Scalar::all(1), matPosIndex );
    matResult.setTo( cv::Scalar::all(-1), matNegIndex );
    matResult = matResult.reshape ( 1, matInput.rows - 1 );

    matPosPole = matPosIndex.clone().reshape(1, matInput.rows - 1 );
    matNegPole = matNegIndex.clone().reshape(1, matInput.rows - 1 );
    //printf("Mat result\n");
    //printIntMat<char>(matResult);
    return matResult;
}

template<typename T>
inline std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

VectorOfPoint Unwrap::selectLonePole(ListOfPoint &listPoint, int width, int height, size_t nLonePoneCount )
{    
    VectorOfPoint vecResult;
    if ( nLonePoneCount > listPoint.size() )
        return vecResult;

    std::vector<int> vecDistance;
    for ( auto iter = listPoint.cbegin(); iter != listPoint.cend(); ++ iter ) {
        cv::Point point = *iter;
        std::vector<int> vec4SideDistance;
        vec4SideDistance.push_back ( point.x );
        vec4SideDistance.push_back ( width - point.x );
        vec4SideDistance.push_back ( point.y );
        vec4SideDistance.push_back ( height - point.y );

        auto minElement = std::min_element ( vec4SideDistance.begin(), vec4SideDistance.end() );
        int minDistance = *minElement;
        vecDistance.push_back ( minDistance );
    }

    std::vector<size_t> vecIndex = sort_indexes<int> ( vecDistance );
    std::vector<ListOfPoint::const_iterator> vecIter;
    
    for (size_t j = 0; j < nLonePoneCount; ++ j )
    {
        size_t index = 0;
        for ( auto iter = listPoint.cbegin(); iter != listPoint.cend(); ++ iter )
        {
            if (index == vecIndex[j]) {
                vecIter.push_back(iter);
                vecResult.push_back(*iter);
                ++index;
            }
            ++index;
        }
    }
    for ( const auto &iter : vecIter )  {
        listPoint.erase ( iter );
    }
    return vecResult;
}

void Unwrap::selectPolePair(const VectorOfPoint &vecPosPoint, const VectorOfPoint &vecNegPoint, int rows, int cols, VectorOfPoint &vecResult1, VectorOfPoint &vecResult2, VectorOfPoint &vecResult3 )
{
    const int MAX_RANGE = 20;
    ListOfPoint listPosPoint, listNegPoint;
    for ( const auto &point : vecPosPoint ) listPosPoint.push_back ( point );
    for ( const auto &point : vecNegPoint ) listNegPoint.push_back ( point );

    for ( int range = 1; range <= MAX_RANGE; ++ range ) {
        std::vector<ListOfPoint::const_iterator> vecIter1;
        for ( auto iter1 = listPosPoint.cbegin(); iter1 != listPosPoint.cend(); ++ iter1 )   {
            std::vector<ListOfPoint::const_iterator> vecIter2;
            for ( auto iter2 = listNegPoint.cbegin(); iter2 != listNegPoint.cend(); ++ iter2 )   {
                cv::Point point1 = *iter1, point2 = *iter2;
                if ( abs ( point1.x - point2.x ) <= range && abs ( point1.y - point2.y ) <= range )   {
                    vecIter1.push_back ( iter1 );
                    vecIter2.push_back ( iter2 );

                    vecResult1.push_back ( point1 );
                    vecResult2.push_back ( point2 );

                    listNegPoint.erase ( iter2 );
                    break;
                }
            }
        }
        if ( ! vecIter1.empty() )   {
            for (const auto &iter : vecIter1)
                listPosPoint.erase(iter);
        }

        if ( 3 == range )   {
            if ( listPosPoint.size() > listNegPoint.size() ) {
                vecResult3 = selectLonePole ( listPosPoint, cols, rows, listPosPoint.size() -  listNegPoint.size() );
            }else
            if ( listPosPoint.size() < listNegPoint.size() ) {
                vecResult3 = selectLonePole ( listNegPoint, cols, rows, listNegPoint.size() -  listPosPoint.size() );
            }
        }
    }

    for (const auto &point1 : listPosPoint )  {
        double dMinDistance = 100000000000.;
        ListOfPoint::const_iterator iterMatch;        
        for (auto iter2 = listNegPoint.cbegin(); iter2 != listNegPoint.cend(); ++ iter2 )  {
            cv::Point point2 = *iter2;
            double dDistance = sqrt ( ( point1.x - point2.x ) * ( point1.x - point2.x ) + ( point1.y - point2.y ) * ( point1.y - point2.y ) );
            if ( dDistance < dMinDistance ) {
                dMinDistance = dDistance;
                iterMatch = iter2;
            }
        }
        cv::Point point2 = *iterMatch;
        vecResult1.push_back(point1);
        vecResult2.push_back(point2);
        listNegPoint.erase(iterMatch);
    }
}

VectorOfPoint Unwrap::getIntegralTree(const VectorOfPoint &vecResult1, const VectorOfPoint &vecResult2, const VectorOfPoint &vecResult3 )
{
    VectorOfPoint vecResult;
    for ( int i = 0; i < vecResult1.size(); ++ i ) {
        const auto point1 = vecResult1[i];
        const auto point2 = vecResult2[i];

        const auto minX = std::min ( point1.x, point2.x );
        const auto maxX = std::max ( point1.x, point2.x );
        const auto minY = std::min ( point1.y, point2.y );
        const auto maxY = std::max ( point1.y, point2.y );

        std::vector<int> vecXX { point1.x, point2.x };
        std::vector<int> vecYY { point1.y, point2.y };
        int index = std::min_element ( vecYY.begin(), vecYY.end() ) - vecYY.begin();

        std::vector<int> vecTrkYOne ( maxY - minY + 1 );
        std::iota ( vecTrkYOne.begin(), vecTrkYOne.end(), minY );
        std::vector<int> vecTrkXOne ( vecTrkYOne.size(), vecXX[index] );

        std::vector<int> vecTrkXTwo ( maxX - minX + 1 );
        std::iota ( vecTrkXTwo.begin(), vecTrkXTwo.end(), minX );
        if ( maxX == vecXX[index] )
            std::reverse ( vecTrkXTwo.begin(), vecTrkXTwo.end() );
        std::vector<int> vecTrkYTwo ( vecTrkXTwo.size(), maxY );
        
        for ( int k = 0; k < vecTrkXOne.size(); ++ k )
            vecResult.emplace_back ( vecTrkXOne[k], vecTrkYOne[k] );

        for ( int k = 0; k < vecTrkXTwo.size(); ++ k )
            vecResult.emplace_back ( vecTrkXTwo[k], vecTrkYTwo[k] );
    }
    return vecResult;
}

/*static*/ void Unwrap::calib3DBase(const std::vector<cv::Mat> &vecInputImgs, bool bGuassianFilter, bool bReverseSeq )
{
    std::vector<cv::Mat> vecConvertedImgs;
    for ( auto &mat : vecInputImgs ) {
        cv::Mat matConvert;
        mat.convertTo ( matConvert, CV_32FC1 );
        vecConvertedImgs.push_back ( matConvert );
    }

    if ( bGuassianFilter ) {        
        //Filtering can reduce residues at the expense of a loss of spatial resolution.
        for ( auto &mat : vecConvertedImgs )        
            cv::GaussianBlur ( mat, mat, cv::Size(GUASSIAN_FILTER_SIZE, GUASSIAN_FILTER_SIZE), GUASSIAN_FILTER_SIGMA, GUASSIAN_FILTER_SIGMA, cv::BorderTypes::BORDER_REPLICATE );
    }

    if ( bReverseSeq ) {
        std::swap ( vecConvertedImgs[1], vecConvertedImgs[3] );
        std::swap ( vecConvertedImgs[5], vecConvertedImgs[7] );
    }
    cv::Mat mat00 = vecConvertedImgs[2] - vecConvertedImgs[0];
    cv::Mat mat01 = vecConvertedImgs[3] - vecConvertedImgs[1];

    cv::Mat mat10 = vecConvertedImgs[6] - vecConvertedImgs[4];
    cv::Mat mat11 = vecConvertedImgs[7] - vecConvertedImgs[5];

    cv::Mat matAlpha, matBeta;
    cv::phase ( mat00, mat01, matAlpha );
    cv::phase ( mat10, mat11, matBeta );

    matAlpha = phaseUnwrapSurface ( matAlpha );
    matBeta  = phaseUnwrapSurface ( matBeta );

    cv::Mat matAlphaReshape = matAlpha.reshape ( 1, 1 );
    cv::Mat matBetaReshape  = matBeta.reshape ( 1, matBeta.rows * matBeta.cols );
    cv::Mat matXX = matAlphaReshape;
    matXX.push_back ( cv::Mat::ones(1, matAlpha.rows * matAlpha.cols, CV_32FC1) );
    cv::transpose ( matXX, matXX );
    cv::Mat matK;
    cv::solve ( matXX, matBetaReshape, matK );

    int ROWS = vecInputImgs[0].rows;
    int COLS = vecInputImgs[0].cols;
    cv::Mat matX, matY;
    CalcUtils::meshgrid<float> ( 1, 1, COLS, 1, 1, ROWS, matX, matY );

    cv::Mat matPPz = calculatePPz ( matX, matY, matBeta );
}

inline static cv::Mat setBySign(cv::Mat &matInput, DATA_TYPE value ) {
    cv::Mat matResult = matInput.clone();
    cv::Mat matPosIndex = cv::Mat::zeros(matResult.size(), CV_8UC1);
    cv::Mat matNegIndex = cv::Mat::zeros(matResult.size(), CV_8UC1);
    cv::compare ( matResult, cv::Scalar::all( 0.0001), matPosIndex, cv::CmpTypes::CMP_GT );
    cv::compare ( matResult, cv::Scalar::all(-0.0001), matNegIndex, cv::CmpTypes::CMP_LT );

    matResult.setTo ( cv::Scalar::all(-value), matPosIndex );
    matResult.setTo ( cv::Scalar::all( value), matNegIndex );
    return matResult;
}

/*static*/ cv::Mat Unwrap::phaseUnwrapSurface(const cv::Mat &matPhase) {
    cv::Mat matPhaseResult;
    const double OneCycle = 2. * ONE_HALF_CYLE;

    cv::Mat matPhaseT;
    cv::transpose ( matPhase, matPhaseT );
    cv::Mat matDp = CalcUtils::diff ( matPhaseT, 1, 2 );
    std::vector<std::vector<float>> vecVecArray = CalcUtils::matToVector<float>( matDp );

    cv::Mat matAbsUnderPI;
    cv::compare ( cv::abs ( matDp ), cv::Scalar::all ( ONE_HALF_CYLE ), matAbsUnderPI, cv::CmpTypes::CMP_LT );
    matDp.setTo ( 0, matAbsUnderPI );

    matDp = setBySign ( matDp, OneCycle);       //Correspoinding to matlab: dp(idx) = -sign(dp(idx))*Th;
    matDp = cv::repeat ( matDp, matPhase.cols, 1);

    cv::Mat matTmp ( matPhaseT, cv::Rect(1, 0, matPhaseT.cols - 1, matPhaseT.rows ) );
    matTmp += CalcUtils::cumsum<DATA_TYPE> ( matDp, 2 );

    cv::transpose ( matPhaseT, matPhaseResult );
    return matPhaseResult;
}

static inline cv::Mat calcBezierCoeff ( const cv::Mat &matU ) {
    cv::Mat matP;
    cv::Mat matUSubBy1Pow1 = 1.f - matU;
    cv::Mat matUPow2 = matU * matU;
    cv::Mat matUPow3 = matUPow2 * matU;
    cv::Mat matUPow4 = matUPow3 * matU;
    cv::Mat matUSubBy1Pow2 = matUSubBy1Pow1 * matUSubBy1Pow1;
    cv::Mat matUSubBy1Pow3 = matUSubBy1Pow2 * matUSubBy1Pow1;
    cv::Mat matUSubBy1Pow4 = matUSubBy1Pow3 * matUSubBy1Pow1;
    matP = matUSubBy1Pow4;
    matP.push_back ( 4 * matUSubBy1Pow3 * matU );
    matP.push_back ( 6 * matUSubBy1Pow2 * matUPow2 );
    matP.push_back ( 4 * matUSubBy1Pow1 * matUPow3 );
    matP.push_back ( matUPow4 );
    return matP;
}

/*static*/ cv::Mat Unwrap::calculatePPz(const cv::Mat &matX, const cv::Mat &matY, const cv::Mat &matZ)
{
    int ROWS = matX.rows;
    int COLS = matX.cols;
    cv::Mat matU, matV;
    CalcUtils::meshgrid<float> ( 0, 1.f / ( COLS - 1 ), 1.f, 0, 1.f / ( ROWS - 1 ), ROWS, matU, matV );
    matU = matU.reshape ( 1, 1 );
    matV = matV.reshape ( 1, 1 );
    cv::Mat matZReshape = matZ.reshape ( 1, matZ.rows * matZ.cols );
    cv::Mat matXReshape = matX.reshape ( 1, 1 );
    cv::Mat matYReshape = matY.reshape ( 1, 1 );
    cv::Mat matP1 = calcBezierCoeff ( matU );
    cv::Mat matP2 = calcBezierCoeff ( matV );
    cv::Mat matXX;
    for ( int i = 0; i < BEZIER_RANK; ++ i )
    for ( int j = 0; j < BEZIER_RANK; ++ j ) {
        matXX.push_back ( matP1.row(i) * matP2.row(j) );
    }
    cv::transpose ( matXX, matXX );
    cv::Mat matPPz;
    cv::solve ( matXX, matZReshape, matPPz );
    return matPPz;
}

}
}