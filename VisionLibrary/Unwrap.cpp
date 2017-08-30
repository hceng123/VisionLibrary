#include "Unwrap.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
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

/*static*/ const float Unwrap::GUASSIAN_FILTER_SIGMA =      ToFloat ( pow ( 20, 0.5 ) );
/*static*/ const float Unwrap::ONE_HALF_CYCLE =             ToFloat ( CV_PI );
/*static*/ const float Unwrap::UNSTABLE_DIFF =              3.f;
/*static*/ const float Unwrap::REMOVE_HEIGHT_NOSIE_RATIO =  0.995f;

/*static*/ cv::Mat Unwrap::_getResidualPoint(const cv::Mat &matInput, cv::Mat &matPosPole, cv::Mat &matNegPole) {
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

    cv::compare ( matD, cv::Scalar::all( 0.999 * CV_PI), matPosIndex, cv::CmpTypes::CMP_GT );
    cv::compare ( matD, cv::Scalar::all(-0.999 * CV_PI), matNegIndex, cv::CmpTypes::CMP_LT );

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
    std::vector<size_t> idx ( v.size () );
    std::iota ( idx.begin (), idx.end (), 0 );

    // sort indexes based on comparing values in v
    sort ( idx.begin (), idx.end (),
           [&v]( size_t i1, size_t i2 ) {return v[i1] < v[i2]; } );
    return idx;
}

/*static*/ VectorOfPoint Unwrap::_selectLonePole(ListOfPoint &listPoint, int width, int height, size_t nLonePoneCount ) {    
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

/*static*/ void Unwrap::_selectPolePair( const VectorOfPoint &vecPosPoint,
                                         const VectorOfPoint &vecNegPoint,
                                         int rows, int cols,
                                         VectorOfPoint &vecResult1,
                                         VectorOfPoint &vecResult2,
                                         VectorOfPoint &vecResult3 )
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
                vecResult3 = _selectLonePole ( listPosPoint, cols, rows, listPosPoint.size() -  listNegPoint.size() );
            }else
            if ( listPosPoint.size() < listNegPoint.size() ) {
                vecResult3 = _selectLonePole ( listNegPoint, cols, rows, listNegPoint.size() -  listPosPoint.size() );
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

/*static*/ VectorOfPoint Unwrap::_getIntegralTree(const VectorOfPoint &vecResult1, const VectorOfPoint &vecResult2, const VectorOfPoint &vecResult3 ) {
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
        auto index = std::min_element ( vecYY.begin(), vecYY.end() ) - vecYY.begin();

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

/*static*/ void Unwrap::calib3DBase(const std::vector<cv::Mat> &vecInputImgs, bool bEnableGaussianFilter, bool bReverseSeq, cv::Mat &matK, cv::Mat &matPPz ) {
    std::vector<cv::Mat> vecConvertedImgs;
    for ( auto &mat : vecInputImgs ) {
        cv::Mat matConvert = mat;;
        if ( mat.channels() > 1 )
            cv::cvtColor ( mat, matConvert, CV_BGR2GRAY );
        matConvert.convertTo ( matConvert, CV_32FC1 );
        vecConvertedImgs.push_back ( matConvert );
    }

    if ( bEnableGaussianFilter ) {
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
    cv::phase ( -mat00, mat01, matAlpha );
    //The opencv cv::phase function return result is 0~2*PI, but matlab is -PI~PI, so here need to do a conversion.
    cv::Mat matOverPi;
    cv::compare ( matAlpha, cv::Scalar::all ( CV_PI ), matOverPi, cv::CmpTypes::CMP_GT );
    cv::subtract ( matAlpha, cv::Scalar::all ( 2.f * CV_PI ), matAlpha, matOverPi );

    cv::phase ( -mat10, mat11, matBeta );
    cv::compare ( matBeta, cv::Scalar::all ( CV_PI ), matOverPi, cv::CmpTypes::CMP_GT );
    cv::subtract ( matBeta, cv::Scalar::all ( 2.f * CV_PI ), matBeta, matOverPi );

    matAlpha = _phaseUnwrapSurface ( matAlpha );
    matBeta  = _phaseUnwrapSurface ( matBeta );

    cv::Mat matAlphaReshape = matAlpha.reshape ( 1, 1 );
    cv::Mat matYY = matBeta.reshape ( 1, matBeta.rows * matBeta.cols );
    cv::Mat matXX = matAlphaReshape.clone();
    cv::Mat matOne = cv::Mat::ones(1, matAlpha.rows * matAlpha.cols, CV_32FC1);
    matXX.push_back ( matOne );
    cv::transpose ( matXX, matXX );
    cv::solve ( matXX, matYY, matK, cv::DecompTypes::DECOMP_QR );
#ifdef _DEBUG
    auto vecAlpha = CalcUtils::matToVector<DATA_TYPE>(matAlpha);
    auto vecBeta  = CalcUtils::matToVector<DATA_TYPE>(matBeta);
    auto vecMatK = CalcUtils::matToVector<DATA_TYPE>(matK);
#endif
    int ROWS = vecInputImgs[0].rows;
    int COLS = vecInputImgs[0].cols;
    cv::Mat matX, matY;
    CalcUtils::meshgrid<float> ( 1.f, 1.f, ToFloat ( COLS ), 1.f, 1.f, ToFloat ( ROWS ), matX, matY );

    matPPz = _calculatePPz ( matX, matY, matBeta );
    cv::Mat matZP1 = _calculateSurface ( matAlpha, matPPz );
#ifdef _DEBUG
    auto vecPPz = CalcUtils::matToVector<DATA_TYPE>(matPPz);
    auto vecZP1 = CalcUtils::matToVector<DATA_TYPE>(matZP1);
#endif
}

/*static*/ inline cv::Mat Unwrap::_setBySign(cv::Mat &matInput, DATA_TYPE value ) {
    cv::Mat matResult = matInput.clone();
    cv::Mat matPosIndex = cv::Mat::zeros(matResult.size(), CV_8UC1);
    cv::Mat matNegIndex = cv::Mat::zeros(matResult.size(), CV_8UC1);
    cv::compare ( matResult, cv::Scalar::all( 0.0001), matPosIndex, cv::CmpTypes::CMP_GT );
    cv::compare ( matResult, cv::Scalar::all(-0.0001), matNegIndex, cv::CmpTypes::CMP_LT );

    matResult.setTo ( cv::Scalar::all(-value), matPosIndex );
    matResult.setTo ( cv::Scalar::all( value), matNegIndex );
    return matResult;
}

/*static*/ cv::Mat Unwrap::_phaseUnwrapSurface(const cv::Mat &matPhase) {    
    const float OneCycle = 2.f * ONE_HALF_CYCLE;

    cv::Mat matPhaseT;
    cv::transpose ( matPhase, matPhaseT );
    cv::Mat matTmpPhase = cv::Mat ( matPhaseT, cv::Rect ( 0, 0, matPhaseT.cols, 1 ) );
    cv::Mat matDp = CalcUtils::diff ( matTmpPhase, 1, 2 );

#ifdef _DEBUG 
    auto vecDp = CalcUtils::matToVector<DATA_TYPE>( matDp );
#endif

    cv::Mat matAbsUnderPI;
    cv::compare ( cv::abs ( matDp ), cv::Scalar::all ( ONE_HALF_CYCLE ), matAbsUnderPI, cv::CmpTypes::CMP_LT );
    matDp.setTo ( 0, matAbsUnderPI );

    matDp = _setBySign ( matDp, OneCycle);       //Correspoinding to matlab: dp(idx) = -sign(dp(idx))*Th;
    matDp = cv::repeat ( matDp, matPhase.cols, 1);

    cv::Mat matTmp ( matPhaseT, cv::Rect(1, 0, matPhaseT.cols - 1, matPhaseT.rows ) );
    cv::Mat matCumSum = CalcUtils::cumsum<DATA_TYPE> ( matDp, 2 );
    matTmp += matCumSum;

    cv::Mat matPhaseResult;
    cv::transpose ( matPhaseT, matPhaseResult );
#ifdef _DEBUG 
    auto vecCumsum = CalcUtils::matToVector<DATA_TYPE>( matCumSum );
    auto vecTmp = CalcUtils::matToVector<DATA_TYPE>( matTmp );
    auto vecPhaseT = CalcUtils::matToVector<DATA_TYPE>( matPhaseT );
    auto vecPhaseResult = CalcUtils::matToVector<DATA_TYPE>( matPhaseResult );
#endif

     matDp = CalcUtils::diff ( matPhaseResult, 1, 2 );
     cv::compare ( cv::abs ( matDp ), cv::Scalar::all ( ONE_HALF_CYCLE ), matAbsUnderPI, cv::CmpTypes::CMP_LT );
     matDp.setTo ( 0, matAbsUnderPI );
     matDp = _setBySign ( matDp, OneCycle);

     cv::Mat matTmp1 ( matPhaseResult,cv::Rect(1, 0, matPhaseResult.cols - 1, matPhaseResult.rows ) );
     matTmp1 += CalcUtils::cumsum<DATA_TYPE> ( matDp, 2 );
#ifdef _DEBUG 
     vecPhaseResult = CalcUtils::matToVector<DATA_TYPE>( matPhaseResult );
#endif
    return matPhaseResult;
}

static inline cv::Mat calcBezierCoeff ( const cv::Mat &matU ) {
    cv::Mat matP;
    cv::Mat matUSubBy1Pow1 = 1.f - matU;
    cv::Mat matUPow2 = matU.mul ( matU );
    cv::Mat matUPow3 = matUPow2.mul ( matU );
    cv::Mat matUPow4 = matUPow3.mul ( matU );
    cv::Mat matUSubBy1Pow2 = matUSubBy1Pow1.mul ( matUSubBy1Pow1 );
    cv::Mat matUSubBy1Pow3 = matUSubBy1Pow2.mul ( matUSubBy1Pow1 );
    cv::Mat matUSubBy1Pow4 = matUSubBy1Pow3.mul ( matUSubBy1Pow1 );
    matP = matUSubBy1Pow4;
    matP.push_back ( cv::Mat ( 4 * matUSubBy1Pow3.mul ( matU ) ) );
    matP.push_back ( cv::Mat ( 6 * matUSubBy1Pow2.mul ( matUPow2 ) ) );
    matP.push_back ( cv::Mat ( 4 * matUSubBy1Pow1.mul ( matUPow3 ) ) );
    matP.push_back ( matUPow4 );
    return matP;
}

/*static*/ cv::Mat Unwrap::_calculatePPz(const cv::Mat &matX, const cv::Mat &matY, const cv::Mat &matZ)
{
    int ROWS = matX.rows;
    int COLS = matX.cols;
    cv::Mat matU, matV;
    CalcUtils::meshgrid<float> ( 0, 1.f / ( COLS - 1 ), 1.f, 0, 1.f / ( ROWS - 1 ), 1.f, matU, matV );
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
        matXX.push_back ( cv::Mat (  matP1.row(i).mul ( matP2.row(j) ) ) );
    }
    cv::transpose ( matXX, matXX );
    cv::Mat matPPz;
    cv::solve ( matXX, matZReshape, matPPz, cv::DecompTypes::DECOMP_QR );
    return matPPz;
}

/*static*/ cv::Mat Unwrap::_calculateSurface(const cv::Mat &matZ, const cv::Mat &matPPz)
{
    int ROWS = matZ.rows;
    int COLS = matZ.cols;
    cv::Mat matU, matV;
    CalcUtils::meshgrid<float> ( 0, 1.f / ( COLS - 1 ), 1.f, 0, 1.f / ( ROWS - 1 ), 1.f, matU, matV );
    matU = matU.reshape ( 1, 1 );
    matV = matV.reshape ( 1, 1 );
    cv::Mat matP1 = calcBezierCoeff ( matU );
    cv::Mat matP2 = calcBezierCoeff ( matV );
    cv::Mat matXX;
    for ( int i = 0; i < BEZIER_RANK; ++ i )
    for ( int j = 0; j < BEZIER_RANK; ++ j ) {
        matXX.push_back ( cv::Mat ( matP1.row(i).mul ( matP2.row(j) ) ) );
    }
    cv::transpose ( matXX, matXX );
    cv::Mat matZP = matXX * matPPz;
    matZP = matZP.reshape ( 1, ROWS );
    return matZP;
}

/*static*/ void Unwrap::calc3DHeight( const PR_CALC_3D_HEIGHT_CMD *const pstCmd, PR_CALC_3D_HEIGHT_RPY *const pstRpy) {
    std::vector<cv::Mat> vecConvertedImgs;
    for ( auto &mat : pstCmd->vecInputImgs ) {
        cv::Mat matConvert = mat;;
        if ( mat.channels() > 1 )
            cv::cvtColor ( mat, matConvert, CV_BGR2GRAY );
        matConvert.convertTo ( matConvert, CV_32FC1 );
        vecConvertedImgs.push_back ( matConvert );
    }

    if ( pstCmd->bEnableGaussianFilter ) {   
        //Filtering can reduce residues at the expense of a loss of spatial resolution.
        for ( auto &mat : vecConvertedImgs )        
            cv::GaussianBlur ( mat, mat, cv::Size(GUASSIAN_FILTER_SIZE, GUASSIAN_FILTER_SIZE), GUASSIAN_FILTER_SIGMA, GUASSIAN_FILTER_SIGMA, cv::BorderTypes::BORDER_REPLICATE );
    }

    if ( pstCmd->bReverseSeq ) {
        std::swap ( vecConvertedImgs[1], vecConvertedImgs[3] );
        std::swap ( vecConvertedImgs[5], vecConvertedImgs[7] );
    }
    cv::Mat mat00 = vecConvertedImgs[2] - vecConvertedImgs[0];
    cv::Mat mat01 = vecConvertedImgs[3] - vecConvertedImgs[1];

    cv::Mat mat10 = vecConvertedImgs[6] - vecConvertedImgs[4];
    cv::Mat mat11 = vecConvertedImgs[7] - vecConvertedImgs[5];

    cv::Mat matAlpha, matBeta;
    cv::phase ( -mat00, mat01, matAlpha );
    //The opencv cv::phase function return result is 0~2*PI, but matlab is -PI~PI, so here need to do a conversion.
    cv::Mat matOverPi;
    cv::compare ( matAlpha, cv::Scalar::all ( CV_PI ), matOverPi, cv::CmpTypes::CMP_GT );
    cv::subtract ( matAlpha, cv::Scalar::all ( 2.f * CV_PI ), matAlpha, matOverPi );

    cv::phase ( -mat10, mat11, matBeta );
    cv::compare ( matBeta, cv::Scalar::all ( CV_PI ), matOverPi, cv::CmpTypes::CMP_GT );
    cv::subtract ( matBeta, cv::Scalar::all ( 2.f * CV_PI ), matBeta, matOverPi );
    
    cv::Mat matPosPole, matNegPole, matResidue;
    matResidue = _getResidualPoint ( matAlpha, matPosPole, matNegPole );
    {
        cv::Mat matNewResidue = cv::Mat::zeros ( matAlpha.rows, matAlpha.cols, CV_8SC1 );
        cv::Mat matROI ( matNewResidue, cv::Rect ( 0, 0, matAlpha.cols - 1, matAlpha.rows - 1 ) );
        matResidue.copyTo ( matROI );
        matResidue = matNewResidue;
    }

    VectorOfPoint vecPosPole, vecNegPole;
    cv::findNonZero ( matPosPole, vecPosPole );
    cv::findNonZero ( matNegPole, vecNegPole );

    VectorOfPoint vecPoint1, vecPoint2, vecPoint3;
    _selectPolePair ( vecPosPole, vecNegPole, matResidue.rows, matResidue.cols, vecPoint1, vecPoint2, vecPoint3 );

    VectorOfPoint vecTree = _getIntegralTree ( vecPoint1, vecPoint2, vecPoint3 );
    cv::Mat matBranchCut = cv::Mat::zeros ( matResidue.rows, matResidue.cols, CV_8UC1 );
    for ( const auto &point : vecTree ) {
        matBranchCut.at<uchar>(point) = 255;
    }

    matAlpha = _phaseUnwrapSurfaceTrk ( matAlpha, matBranchCut );
    cv::Mat matDiffUnderTolIndex, matAvgUnderTolIndex;
    _findUnstablePoint ( vecConvertedImgs, pstCmd->fMinIntensityDiff, pstCmd->fMinAvgIntensity, matDiffUnderTolIndex, matAvgUnderTolIndex );
#ifdef _DEBUG
    int nUnstableCount = cv::countNonZero( matDiffUnderTolIndex );
#endif
    matAlpha.setTo ( NAN, matDiffUnderTolIndex );
    matAlpha = matAlpha * pstCmd->matK.at<DATA_TYPE>(0, 0) + pstCmd->matK.at<DATA_TYPE>(1, 0);
    cv::Mat matZP1 = _calculateSurface ( matAlpha, pstCmd->matPPz );
    matBeta = _phaseUnwrapSurfaceByRefer ( matBeta, matAlpha );

#ifdef _DEBUG
    auto vecAlpha = CalcUtils::matToVector<DATA_TYPE>(matAlpha);
    auto vecZP1 = CalcUtils::matToVector<DATA_TYPE>(matZP1);
    auto vecBeta = CalcUtils::matToVector<DATA_TYPE>(matBeta);
#endif

    matAlpha -= matZP1;
    matBeta  -= matZP1;

    auto matH = matBeta.clone();
    matH.setTo ( NAN, matAvgUnderTolIndex );
    matH.setTo ( NAN, matDiffUnderTolIndex );

    cv::medianBlur ( matH, matH, 5 );

#ifdef _DEBUG
    auto vecVecH = CalcUtils::matToVector<DATA_TYPE>( matH );
#endif

    cv::Mat matHReshape = matH.reshape ( 1, 1 );
    auto vecHReshape = CalcUtils::matToVector<float> ( matHReshape );
    auto vecSortedIndex = sort_indexes ( vecHReshape[0] );
    for ( int i = ToInt32 ( REMOVE_HEIGHT_NOSIE_RATIO * vecSortedIndex.size() ); i < ToInt32 ( vecSortedIndex.size() ); ++ i )
        matH.at<DATA_TYPE> ( i ) = NAN;
#ifdef _DEBUG
    vecVecH = CalcUtils::matToVector<DATA_TYPE>( matH );
#endif
    pstRpy->matHeight = matH;
}

/*static*/ cv::Mat Unwrap::_phaseUnwrapSurfaceTrk ( const cv::Mat &matPhase, const cv::Mat &matBranchCut) {
    cv::Mat matPhaseResult;
    const float OneCycle = 2.f * ONE_HALF_CYCLE;
    cv::Mat matUnwrapped = matBranchCut.clone();

    cv::Mat dyPhase = CalcUtils::diff ( matPhase, 1, 1 );
    cv::Mat dxPhase = CalcUtils::diff ( matPhase, 1, 2 );

    auto vecVecArray = CalcUtils::matToVector<DATA_TYPE>( dxPhase );

    cv::Mat matPhaseT;
    cv::transpose ( matPhase, matPhaseT );

    cv::Mat matTmpPhase = cv::Mat ( matPhaseT, cv::Rect ( 0, 0, matPhaseT.cols, 1 ) );
    auto vecmatTmp = CalcUtils::matToVector<DATA_TYPE> ( matTmpPhase );

    cv::Mat matDp = CalcUtils::diff ( matTmpPhase , 1, 2 );

#ifdef _DEBUG
    auto vecMatDp_1 = CalcUtils::matToVector<DATA_TYPE>( matDp );
#endif

    cv::Mat matAbsUnderPI;
    cv::compare ( cv::abs ( matDp ), cv::Scalar::all ( ONE_HALF_CYCLE ), matAbsUnderPI, cv::CmpTypes::CMP_LT );
    matDp.setTo ( 0, matAbsUnderPI );

    matDp = _setBySign ( matDp, OneCycle);
    matDp = cv::repeat ( matDp, matPhase.cols, 1);

#ifdef _DEBUG
    auto vecMatDp_2 = CalcUtils::matToVector<DATA_TYPE> ( matDp );
#endif

    cv::Mat matTmp ( matPhaseT,cv::Rect(1, 0, matPhaseT.cols - 1, matPhaseT.rows ) );
    matTmp += CalcUtils::cumsum<DATA_TYPE> ( matDp, 2 );

    cv::transpose ( matPhaseT, matPhaseResult );

    matPhaseResult.setTo ( cv::Scalar::all(NAN), matBranchCut );

#ifdef _DEBUG
    //displayOneRowData<double> ( matPhaseT, 0 );
#endif

    matDp = CalcUtils::diff ( matPhaseResult, 1, 2 );
    cv::compare ( cv::abs ( matDp ), cv::Scalar::all ( ONE_HALF_CYCLE ), matAbsUnderPI, cv::CmpTypes::CMP_LT );
    matDp.setTo ( 0, matAbsUnderPI );
    matDp = _setBySign ( matDp, OneCycle);

    cv::Mat matTmp1 ( matPhaseResult,cv::Rect(1, 0, matPhaseResult.cols - 1, matPhaseResult.rows ) );
    matTmp1 += CalcUtils::cumsum<DATA_TYPE> ( matDp, 2 );

    matPhaseResult.setTo ( cv::Scalar(0), matBranchCut );
    
    cv::Mat matNan = cv::Mat ( matPhaseResult != matPhaseResult );  //Find out points been blocked by branch cut.

#ifdef _DEBUG
    cv::imwrite("./data/NanMap_0.png", matNan * 255);
#endif

    cv::Mat matIsRowNeedUnwrap;
    cv::reduce ( matNan, matIsRowNeedUnwrap, 1, CV_REDUCE_MAX );
    std::vector<cv::Point> vecRowsNeedToUnwrap;
    cv::findNonZero ( matIsRowNeedUnwrap, vecRowsNeedToUnwrap );
    for ( const auto &point : vecRowsNeedToUnwrap )    {
        int row = point.y;
        cv::Mat matOneRowIndex ( matNan, cv::Range ( row, row + 1 ) );
        if ( cv::sum ( matOneRowIndex )[0] <= 1 )
            continue;
        cv::Mat matPhaseLastRow ( matPhaseResult, cv::Range ( row - 1, row ) );
        cv::Mat matPhaseDiff(dyPhase, cv::Range ( row - 1, row ) );

        cv::Mat matZero;
        cv::compare ( matPhaseLastRow, cv::Scalar(0), matZero, cv::CmpTypes::CMP_EQ );
        if ( cv::sum ( matZero )[0] > 0 )
            matPhaseLastRow.setTo ( cv::Scalar(NAN), matZero );

        matDp = matPhaseDiff;
        cv::compare ( cv::abs ( matDp ), cv::Scalar ( ONE_HALF_CYCLE ), matAbsUnderPI, cv::CmpTypes::CMP_LT );
        matDp.setTo ( 0, matAbsUnderPI );
        matDp = _setBySign ( matDp, OneCycle );
        
        cv::Mat matPhaseLastRowClone = matPhaseLastRow.clone();
        cv::accumulate ( matPhaseDiff, matPhaseLastRowClone, matOneRowIndex );
        cv::accumulate ( matDp,        matPhaseLastRowClone, matOneRowIndex );

        cv::Mat matOneRowPhase ( matPhaseResult, cv::Range ( row, row + 1 ) );
        matPhaseLastRowClone.copyTo ( matOneRowPhase, matOneRowIndex );
    }

    matNan = cv::Mat ( matPhaseResult != matPhaseResult );

#ifdef _DEBUG
    cv::imwrite("./data/NanMap_1.png", matNan * 255);
#endif

    cv::Mat matIsColNeedUnwrap;
    cv::reduce ( matNan, matIsColNeedUnwrap, 0, CV_REDUCE_MAX );
    std::vector<cv::Point> vecColsNeedToUnwrap;
    cv::findNonZero ( matIsColNeedUnwrap, vecColsNeedToUnwrap );
    for ( const auto &point : vecColsNeedToUnwrap )    {
        int col = point.x;
        cv::Mat matOneColIndex ( matNan, cv::Range::all(), cv::Range ( col, col + 1 ) );
        if ( cv::sum ( matOneColIndex )[0] <= 1 )
            continue;
        cv::Mat matPhaseLastCol ( matPhaseResult, cv::Range::all(), cv::Range ( col - 1, col ) );
        cv::Mat matPhaseDiff ( dxPhase, cv::Range::all(), cv::Range ( col - 1, col ) );

        cv::Mat matZero;
        cv::compare ( matPhaseLastCol, cv::Scalar(0), matZero, cv::CmpTypes::CMP_EQ );
        if ( cv::sum ( matZero )[0] > 0 )
            matPhaseLastCol.setTo ( cv::Scalar(NAN), matZero );

        matDp = matPhaseDiff;
        cv::compare ( cv::abs(matDp), cv::Scalar::all(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT );
        matDp.setTo ( 0, matAbsUnderPI );
        matDp = _setBySign ( matDp, OneCycle );
        
        cv::Mat matPhaseLastColClone = matPhaseLastCol.clone();
        cv::accumulate ( matPhaseDiff, matPhaseLastColClone, matOneColIndex );
        cv::accumulate ( matDp,        matPhaseLastColClone, matOneColIndex );

        cv::Mat matOneColPhase ( matPhaseResult, cv::Range::all(), cv::Range ( col, col + 1 ) );
        matPhaseLastColClone.copyTo ( matOneColPhase, matOneColIndex );
    }

#ifdef _DEBUG
    matNan = cv::Mat ( matPhaseResult != matPhaseResult );
    cv::imwrite("./data/NanMap_2.png", matNan * 255);
#endif

    matPhaseResult.setTo ( cv::Scalar(NAN), matBranchCut );

#ifdef _DEBUG
    matNan = cv::Mat ( matPhaseResult != matPhaseResult );
    cv::imwrite("./data/NanMap_3.png", matNan * 255);
#endif

#ifdef _DEBUG
    //displayOneRowData<double> ( matPhaseResult, 100 );
#endif
    //cv::cum
    return matPhaseResult;
}

/*static*/ cv::Mat Unwrap::_phaseUnwrapSurfaceByRefer(const cv::Mat &matPhase, const cv::Mat &matRef ) {
    const float OneCycle = 2.f * ONE_HALF_CYCLE;
    cv::Mat matPhaseResult = matPhase.clone();
    cv::Mat matNan = cv::Mat ( matRef != matRef );
    matPhaseResult.setTo ( NAN, matNan );

    cv::Mat matErr = matPhaseResult - matRef;
    matErr /= OneCycle;
    auto matN = CalcUtils::floor<DATA_TYPE>(matErr);
#ifdef _DEBUG
    auto vecVecN = CalcUtils::matToVector<DATA_TYPE>(matN);
#endif
    matPhaseResult = matPhaseResult - matN * OneCycle;

    matErr = matPhaseResult - matRef;
    
    // Same as matlab idx = find( abs(pErr) > 0.9999*Th );
    cv::Mat matPosOverHalfCycleIndex, matNegOverHalfCycleIndex;
    cv::compare ( matErr,   0.9999 * ONE_HALF_CYCLE, matPosOverHalfCycleIndex, cv::CmpTypes::CMP_GT );
    cv::compare ( matErr, - 0.9999 * ONE_HALF_CYCLE, matNegOverHalfCycleIndex, cv::CmpTypes::CMP_LT );
    // Same as matlab Phase(idx) = Phase(idx) - sign(pErr(idx))*2*Th;
    cv::subtract ( matPhaseResult, OneCycle, matPhaseResult, matPosOverHalfCycleIndex );
    cv::add      ( matPhaseResult, OneCycle, matPhaseResult, matNegOverHalfCycleIndex );
    return matPhaseResult;
}

//For shadow area
/*static*/ void Unwrap:: _findUnstablePoint(const std::vector<cv::Mat> &vecInputImgs, float fDiffTol, float fAvgTol, cv::Mat &matDiffUnderTolIndex, cv::Mat &matAvgUnderTolIndex) {
    int ROWS = vecInputImgs[0].rows;
    int COLS = vecInputImgs[0].cols;
    cv::Mat matCombine;
    const int GROUP_IMG_COUNT = 4;
    for ( int i = 0; i < GROUP_IMG_COUNT; ++ i )
        matCombine.push_back ( vecInputImgs[i].reshape ( 1, 1 ) );
    cv::Mat matMax, matMin, matAvg;
    cv::reduce ( matCombine, matMax, 0, cv::ReduceTypes::REDUCE_MAX );
    cv::reduce ( matCombine, matMin, 0, cv::ReduceTypes::REDUCE_MIN );
    cv::reduce ( matCombine, matAvg, 0, cv::ReduceTypes::REDUCE_AVG );
    cv::Mat matDiff = matMax - matMin;
    cv::compare ( matDiff, fDiffTol, matDiffUnderTolIndex, cv::CmpTypes::CMP_LT);
    cv::compare ( matAvg,  fAvgTol,  matAvgUnderTolIndex,  cv::CmpTypes::CMP_LT);
    matDiffUnderTolIndex = matDiffUnderTolIndex.reshape ( 1, ROWS );
    matAvgUnderTolIndex  = matAvgUnderTolIndex. reshape ( 1, ROWS );
}

}
}