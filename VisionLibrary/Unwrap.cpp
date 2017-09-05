#include "Unwrap.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "CalcUtils.h"
#include "VisionAlgorithm.h"
#include "TimeLog.h"
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
    const float OneCycle = 2.f * ONE_HALF_CYCLE;
    cv::Mat matKernelY = (cv::Mat_<DATA_TYPE>(2, 1) << -1, 1);
    cv::Mat matKernelX = (cv::Mat_<DATA_TYPE>(1, 2) << -1, 1);
    cv::Mat matDx, matDy, matD1, matD2, matD3, matD4;
    cv::filter2D(matInput, matDx, -1, matKernelX, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
    matDx = cv::Mat(matDx, cv::Rect(1, 0, matDx.cols - 1, matDx.rows)).clone();

    cv::filter2D(matInput, matDy, -1, matKernelY, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
    matDy = cv::Mat(matDy, cv::Rect(0, 1, matDy.cols, matDy.rows - 1)).clone();

    matD1 =  cv::Mat(matDx, cv::Rect(0, 0, matDx.cols,     matDx.rows - 1)).clone();
    matD2 =  cv::Mat(matDy, cv::Rect(1, 0, matDy.cols - 1, matDy.rows    )).clone();
    matD3 = -cv::Mat(matDx, cv::Rect(0, 1, matDx.cols,     matDx.rows - 1)).clone();
    matD4 = -cv::Mat(matDy, cv::Rect(0, 0, matDy.cols - 1, matDy.rows    )).clone();

    cv::Mat matD;
    matD.push_back ( matD1.reshape(1, 1) );
    matD.push_back ( matD2.reshape(1, 1) );
    matD.push_back ( matD3.reshape(1, 1) );
    matD.push_back ( matD4.reshape(1, 1) );
    cv::transpose ( matD, matD );

    cv::Mat matPosIndex = cv::Mat::zeros ( matD.size(), CV_8UC1 );
    cv::Mat matNegIndex = cv::Mat::zeros ( matD.size(), CV_8UC1 );

    cv::compare ( matD, cv::Scalar::all( 0.9999 * CV_PI), matPosIndex, cv::CmpTypes::CMP_GT );
    cv::compare ( matD, cv::Scalar::all(-0.9999 * CV_PI), matNegIndex, cv::CmpTypes::CMP_LT );

    // Work as ds(idx) = ds(idx) - sign(d(idx))*pi*2;
    cv::subtract ( matD, OneCycle, matD, matPosIndex );
    cv::add      ( matD, OneCycle, matD, matNegIndex );

#ifdef _DEBUG
    //auto vevVecD = CalcUtils::matToVector<DATA_TYPE> ( matD );
#endif

    cv::Mat matSum;
    cv::reduce ( matD, matSum, 1, CV_REDUCE_SUM );

#ifdef _DEBUG
    cv::Mat matSumT;
    cv::transpose ( matSum, matSumT );
    auto vevVecSumT = CalcUtils::matToVector<DATA_TYPE>( matSumT );
#endif

    matPosIndex.setTo(0);
    matNegIndex.setTo(0);
    cv::compare ( matSum, cv::Scalar::all( 0.9999 * CV_PI * 2.f), matPosIndex, cv::CmpTypes::CMP_GT );
    cv::compare ( matSum, cv::Scalar::all(-0.9999 * CV_PI * 2.f), matNegIndex, cv::CmpTypes::CMP_LT );

    //printf("matPosIndex result\n");
    //printIntMat<uchar>(matPosIndex);

    //printf("matNegIndex result\n");
    //printIntMat<uchar>(matNegIndex);

    int nSizeOfResultMat = ( matInput.rows - 1 ) * ( matInput.cols - 1 );
    cv::Mat matResult = cv::Mat::zeros ( nSizeOfResultMat, 1, CV_8SC1 );
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
    std::sort ( idx.begin (), idx.end (),
           [&v]( size_t i1, size_t i2 ) { return v[i1] < v[i2]; } );
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
    
    for (size_t j = 0; j < nLonePoneCount; ++ j ) {
        size_t index = 0;
        for ( auto iter = listPoint.cbegin(); iter != listPoint.cend(); ++ iter ) {
            if (index == vecIndex[j]) {
                vecIter.push_back(iter);
                vecResult.push_back(*iter);
                ++ index;
            }
            ++ index;
        }
    }

    for ( const auto &iter : vecIter )
        listPoint.erase ( iter );
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

    if ( ! vecResult3.empty() )
        throw std::exception("Unwrap::_getIntegralTree not finish yet.");
    return vecResult;
}

/*static*/ void Unwrap::calib3DBase(const std::vector<cv::Mat> &vecInputImgs, bool bEnableGaussianFilter, bool bReverseSeq, cv::Mat &matThickToThinStripeK, cv::Mat &matBaseSurfaceParam ) {
    std::vector<cv::Mat> vecConvertedImgs;
    for ( auto &mat : vecInputImgs ) {
        cv::Mat matConvert = mat;
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
    cv::solve ( matXX, matYY, matThickToThinStripeK, cv::DecompTypes::DECOMP_QR );
#ifdef _DEBUG
    auto vecAlpha = CalcUtils::matToVector<DATA_TYPE>(matAlpha);
    auto vecBeta  = CalcUtils::matToVector<DATA_TYPE>(matBeta);
    auto vecMatK  = CalcUtils::matToVector<DATA_TYPE>(matThickToThinStripeK);
#endif
    int ROWS = vecInputImgs[0].rows;
    int COLS = vecInputImgs[0].cols;
    cv::Mat matX, matY;
    CalcUtils::meshgrid<float> ( 1.f, 1.f, ToFloat ( COLS ), 1.f, 1.f, ToFloat ( ROWS ), matX, matY );

    matBaseSurfaceParam = _calculatePPz ( matX, matY, matBeta );
    cv::Mat matZP1 = _calculateBaseSurface ( matAlpha.rows, matAlpha.cols, matBaseSurfaceParam );
#ifdef _DEBUG
    auto vecPPz = CalcUtils::matToVector<DATA_TYPE>(matBaseSurfaceParam);
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
    auto vecVecDp = CalcUtils::matToVector<DATA_TYPE>( matDp );
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

     cv::Mat matTmpl ( matPhaseResult,cv::Rect(1, 0, matPhaseResult.cols - 1, matPhaseResult.rows ) );
     matTmpl += CalcUtils::cumsum<DATA_TYPE> ( matDp, 2 );
#ifdef _DEBUG 
     vecPhaseResult = CalcUtils::matToVector<DATA_TYPE>( matPhaseResult );
#endif
    return matPhaseResult;
}

static inline cv::Mat calcBezierCoeff ( const cv::Mat &matU ) {
    cv::Mat matUSubBy1Pow1 = 1.f - matU;
    cv::Mat matUSubBy1Pow2 = matUSubBy1Pow1.mul ( matUSubBy1Pow1 );
    cv::Mat matUSubBy1Pow3 = matUSubBy1Pow2.mul ( matUSubBy1Pow1 );
    cv::Mat matUSubBy1Pow4 = matUSubBy1Pow3.mul ( matUSubBy1Pow1 );
    cv::Mat matUPow2 = matU.    mul ( matU );
    cv::Mat matUPow3 = matUPow2.mul ( matU );
    cv::Mat matUPow4 = matUPow3.mul ( matU );
    
    cv::Mat matP;
    matP = matUSubBy1Pow4;
    matP.push_back ( cv::Mat ( 4 * matUSubBy1Pow3.mul ( matU ) ) );
    matP.push_back ( cv::Mat ( 6 * matUSubBy1Pow2.mul ( matUPow2 ) ) );
    matP.push_back ( cv::Mat ( 4 * matUSubBy1Pow1.mul ( matUPow3 ) ) );
    matP.push_back ( matUPow4 );
    return matP;
}

/*static*/ cv::Mat Unwrap::_calculatePPz(const cv::Mat &matX, const cv::Mat &matY, const cv::Mat &matZ) {
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
    cv::Mat matBaseSurfaceParam;
    cv::solve ( matXX, matZReshape, matBaseSurfaceParam, cv::DecompTypes::DECOMP_QR );
    return matBaseSurfaceParam;
}

/*static*/ cv::Mat Unwrap::_calculateBaseSurface(int rows, int cols, const cv::Mat &matBaseSurfaceParam) {
    CStopWatch stopWatch;
    cv::Mat matU, matV;
    CalcUtils::meshgrid<float> ( 0, 1.f / ( cols - 1 ), 1.f, 0, 1.f / ( rows - 1 ), 1.f, matU, matV );
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
    cv::Mat matZP = matXX * matBaseSurfaceParam;
    matZP = matZP.reshape ( 1, rows );

    TimeLog::GetInstance()->addTimeLog( __FUNCTION__, stopWatch.Span() );
    return matZP;
}

/*static*/ void Unwrap::calc3DHeight( const PR_CALC_3D_HEIGHT_CMD *const pstCmd, PR_CALC_3D_HEIGHT_RPY *const pstRpy) {
    std::vector<cv::Mat> vecConvertedImgs;
    for ( auto &mat : pstCmd->vecInputImgs ) {
        cv::Mat matConvert = mat;
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

    const auto ROWS = matAlpha.rows;
    const auto COLS = matAlpha.cols;
    
    cv::Mat matPosPole, matNegPole, matResidue;
    matResidue = _getResidualPoint ( matAlpha, matPosPole, matNegPole );
    //{
    //    cv::Mat matNewResidue = cv::Mat::zeros ( matAlpha.rows, matAlpha.cols, CV_8SC1 );
    //    cv::Mat matROI ( matNewResidue, cv::Rect ( 0, 0, matAlpha.cols - 1, matAlpha.rows - 1 ) );
    //    matResidue.copyTo ( matROI );
    //    matResidue = matNewResidue;
    //}

    VectorOfPoint vecPosPole, vecNegPole;
    cv::findNonZero ( matPosPole, vecPosPole );
    cv::findNonZero ( matNegPole, vecNegPole );

    VectorOfPoint vecPoint1, vecPoint2, vecPoint3;
    _selectPolePair ( vecPosPole, vecNegPole, ROWS - 1, COLS - 1, vecPoint1, vecPoint2, vecPoint3 );

    VectorOfPoint vecTree = _getIntegralTree ( vecPoint1, vecPoint2, vecPoint3 );
    cv::Mat matBranchCut = cv::Mat::zeros ( ROWS, COLS, CV_8UC1 );
    for ( const auto &point : vecTree )
        matBranchCut.at<uchar>(point) = 255;

    matAlpha = _phaseUnwrapSurfaceTrk ( matAlpha, matBranchCut );
#ifdef _DEBUG
    auto vevVecBranchCut = CalcUtils::matToVector<uchar>(matBranchCut);
    auto vecVecAlpha = CalcUtils::matToVector<DATA_TYPE>(matAlpha);
#endif
    cv::Mat matDiffUnderTolIndex, matAvgUnderTolIndex;
    _findUnstablePoint ( vecConvertedImgs, pstCmd->fMinIntensityDiff, pstCmd->fMinAvgIntensity, matDiffUnderTolIndex, matAvgUnderTolIndex );
#ifdef _DEBUG
    int nDiffUnderTolCount = cv::countNonZero ( matDiffUnderTolIndex );
    int nAvgUnderTolCount  = cv::countNonZero ( matAvgUnderTolIndex );
#endif
    matAlpha.setTo ( NAN, matDiffUnderTolIndex );
    matAlpha = matAlpha * pstCmd->matThickToThinStripeK.at<DATA_TYPE>(0, 0) + pstCmd->matThickToThinStripeK.at<DATA_TYPE>(1, 0);    
    matBeta = _phaseUnwrapSurfaceByRefer ( matBeta, matAlpha );

    cv::Mat matZP1 = _calculateBaseSurface ( matAlpha.rows, matAlpha.cols, pstCmd->matBaseSurfaceParam );
#ifdef _DEBUG
    vecVecAlpha = CalcUtils::matToVector<DATA_TYPE>(matAlpha);
    auto vecZP1   = CalcUtils::matToVector<DATA_TYPE>(matZP1);
    auto vecBeta  = CalcUtils::matToVector<DATA_TYPE>(matBeta);
#endif
    matAlpha = matAlpha - matZP1;
    matBeta  = matBeta  - matZP1;

    auto matH = matBeta.clone();
    matH.setTo ( NAN, matAvgUnderTolIndex );
    matH.setTo ( NAN, matDiffUnderTolIndex );

#ifdef _DEBUG
    auto vecVecH = CalcUtils::matToVector<DATA_TYPE>( matH );
#endif

    cv::medianBlur ( matH, matH, 5 );

#ifdef _DEBUG
    vecVecH = CalcUtils::matToVector<DATA_TYPE>( matH );
#endif    

    if ( CalcUtils::countOfNan ( matH ) < (1 - REMOVE_HEIGHT_NOSIE_RATIO) * matH.total() ) {
        cv::Mat matHReshape = matH.reshape ( 1, 1 );

#ifdef _DEBUG
        cv::Mat matTopPoints = cv::Mat::zeros ( ROWS, COLS, CV_8UC1 );
#endif

        auto vecVecHReshape = CalcUtils::matToVector<DATA_TYPE> ( matHReshape );
        auto vecSortedIndex = sort_indexes ( vecVecHReshape[0] );
        for (int i = ToInt32 ( REMOVE_HEIGHT_NOSIE_RATIO * vecSortedIndex.size () ); i < ToInt32 ( vecSortedIndex.size () ); ++ i ) {
            matH.at<DATA_TYPE> ( i ) = NAN;

#ifdef _DEBUG
            matTopPoints.at<uchar> ( i ) = 255;
#endif
        }

#ifdef _DEBUG
        cv::Mat matDisplay = (vecConvertedImgs[0] + vecConvertedImgs[1] + vecConvertedImgs[2] + vecConvertedImgs[3]) / 4;
        matDisplay.convertTo ( matDisplay, CV_8UC1 );
        cv::cvtColor ( matDisplay, matDisplay, CV_GRAY2BGR );
        matDisplay.setTo ( cv::Scalar ( 0, 255, 255 ), matAvgUnderTolIndex );
        matDisplay.setTo ( cv::Scalar ( 0, 255, 0 ), matDiffUnderTolIndex );
        matDisplay.setTo ( cv::Scalar ( 0, 0, 255 ), matTopPoints );
        cv::imwrite ( "./data/Nan_5.png", matDisplay );
        vecVecH = CalcUtils::matToVector<DATA_TYPE> ( matH );
#endif
    }

    pstRpy->matPhase = matH;
    if ( ! pstCmd->matPhaseToHeightK.empty() ) {
        cv::Mat matX, matY;
        CalcUtils::meshgrid<DATA_TYPE> ( 1.f, 1.f, ToFloat ( matH.cols ), 1.f, 1.f, ToFloat ( matH.rows ), matX, matY );
        cv::Mat matRatio = pstCmd->matPhaseToHeightK.at<DATA_TYPE>(0, 0) * matX + pstCmd->matPhaseToHeightK.at<DATA_TYPE>(1, 0) * matY + pstCmd->matPhaseToHeightK.at<DATA_TYPE>(2, 0);
        cv::divide ( matH, matRatio, pstRpy->matHeight );
    }else {
        pstRpy->matHeight = matH;
    }
}

/*static*/ cv::Mat Unwrap::_phaseUnwrapSurfaceTrk ( const cv::Mat &matPhase, const cv::Mat &matBranchCut ) {
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

    matDp = CalcUtils::diff ( matPhaseResult, 1, 2 );
    cv::compare ( cv::abs ( matDp ), cv::Scalar::all ( ONE_HALF_CYCLE ), matAbsUnderPI, cv::CmpTypes::CMP_LT );
    matDp.setTo ( 0, matAbsUnderPI );
    matDp = _setBySign ( matDp, OneCycle);

    cv::Mat matTmp1 ( matPhaseResult,cv::Rect(1, 0, matPhaseResult.cols - 1, matPhaseResult.rows ) );
    matTmp1 += CalcUtils::cumsum<DATA_TYPE> ( matDp, 2 );

    matPhaseResult.setTo ( cv::Scalar(0), matBranchCut );
    
    cv::Mat matNan = CalcUtils::getNanMask ( matPhaseResult );  //Find out points been blocked by branch cut.

    cv::Mat matIsRowNeedUnwrap;
    cv::reduce ( matNan, matIsRowNeedUnwrap, 1, CV_REDUCE_MAX );
    std::vector<cv::Point> vecRowsNeedToUnwrap;
    cv::findNonZero ( matIsRowNeedUnwrap, vecRowsNeedToUnwrap );
    for ( const auto &point : vecRowsNeedToUnwrap ) {
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

    matNan = CalcUtils::getNanMask ( matPhaseResult );

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
    matNan = CalcUtils::getNanMask ( matPhaseResult );
    cv::imwrite("./data/NanMap_2.png", matNan * 255);
#endif

    matPhaseResult.setTo ( cv::Scalar(NAN), matBranchCut );

#ifdef _DEBUG
    matNan = CalcUtils::getNanMask ( matPhaseResult );
    cv::imwrite("./data/NanMap_3.png", matNan * 255);
#endif

    return matPhaseResult;
}

/*static*/ cv::Mat Unwrap::_phaseUnwrapSurfaceByRefer(const cv::Mat &matPhase, const cv::Mat &matRef ) {
    const float OneCycle = 2.f * ONE_HALF_CYCLE;
    cv::Mat matPhaseResult = matPhase.clone();
    cv::Mat matNan = CalcUtils::getNanMask ( matRef );
    matPhaseResult.setTo ( NAN, matNan );

    cv::Mat matErr = matPhaseResult - matRef;
    matErr /= OneCycle;
    auto matN = CalcUtils::floor<DATA_TYPE>(matErr);
#ifdef _DEBUG
    auto vecVecN = CalcUtils::matToVector<DATA_TYPE> ( matN );
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
    cv::transpose ( matCombine, matCombine );
    cv::Mat matMax, matMin, matAvg;
    cv::reduce ( matCombine, matMax, 1, cv::ReduceTypes::REDUCE_MAX );
    cv::reduce ( matCombine, matMin, 1, cv::ReduceTypes::REDUCE_MIN );
    cv::reduce ( matCombine, matAvg, 1, cv::ReduceTypes::REDUCE_AVG );
    cv::Mat matDiff = matMax - matMin;
    cv::compare ( matDiff, fDiffTol, matDiffUnderTolIndex, cv::CmpTypes::CMP_LT);
    cv::compare ( matAvg,  fAvgTol,  matAvgUnderTolIndex,  cv::CmpTypes::CMP_LT);
    matDiffUnderTolIndex = matDiffUnderTolIndex.reshape ( 1, ROWS );
    matAvgUnderTolIndex  = matAvgUnderTolIndex. reshape ( 1, ROWS );
}

/*static*/ void Unwrap::calib3DHeight(const PR_CALIB_3D_HEIGHT_CMD *const pstCmd, PR_CALIB_3D_HEIGHT_RPY *const pstRpy) {
    PR_CALC_3D_HEIGHT_CMD stCalcCmd;
    PR_CALC_3D_HEIGHT_RPY stCalcRpy;
    stCalcCmd.vecInputImgs = std::move ( pstCmd->vecInputImgs );
    stCalcCmd.bEnableGaussianFilter = pstCmd->bEnableGaussianFilter;
    stCalcCmd.bReverseSeq = pstCmd->bReverseSeq;
    stCalcCmd.fMinIntensityDiff = pstCmd->fMinIntensityDiff;
    stCalcCmd.fMinAvgIntensity = pstCmd->fMinAvgIntensity;
    stCalcCmd.matThickToThinStripeK = pstCmd->matThickToThinStripeK;
    stCalcCmd.matBaseSurfaceParam = pstCmd->matBaseSurfaceParam;
    calc3DHeight ( &stCalcCmd, &stCalcRpy );
    cv::Mat matPhase = stCalcRpy.matPhase;
    int ROWS = matPhase.rows;
    int COLS = matPhase.cols;

    double dMinValue = 0, dMaxValue = 0;
    cv::Mat matMask = ( matPhase == matPhase ); //Find out value is not NAN.
    cv::minMaxIdx ( matPhase, &dMinValue, &dMaxValue, 0, 0, matMask );
    
    cv::Mat matNormalizedPhase = ( matPhase - dMinValue ) / ( dMaxValue - dMinValue );
    cv::Mat matGrayScalePhase = matNormalizedPhase * PR_MAX_GRAY_LEVEL;
    cv::cvtColor ( matGrayScalePhase, pstRpy->matDivideStepResultImg, CV_GRAY2BGR );
    auto vecThreshold = VisionAlgorithm::autoMultiLevelThreshold ( matGrayScalePhase, matMask, pstCmd->nBlockStepCount );
    std::vector<float> vecNormaledThreshold;
    vecNormaledThreshold.push_back(0.f);
    for ( const auto thresh : vecThreshold )
        vecNormaledThreshold.push_back ( ToFloat ( thresh ) / ToFloat ( PR_MAX_GRAY_LEVEL ) );
    vecNormaledThreshold.push_back(1.f);

    auto meanValue = cv::mean ( matPhase, matMask )[0];
    bool bReverseHeight = meanValue < 0;

    auto nKK = 0;
    if ( bReverseHeight )
        nKK = ToInt32 ( vecNormaledThreshold.size() ) - 2;
    cv::Mat matInRange;
    cv::inRange ( matNormalizedPhase, vecNormaledThreshold[nKK], vecNormaledThreshold[nKK + 1], matInRange );
#ifdef _DEBUG
    auto vevVecNormalizedPhase = CalcUtils::matToVector<DATA_TYPE>(matNormalizedPhase);
#endif
    VectorOfPoint vecPtLocations;
    cv::findNonZero ( matInRange, vecPtLocations );
    std::sort ( vecPtLocations.begin(), vecPtLocations.end(), [&matNormalizedPhase](const cv::Point &pt1, const cv::Point &pt2) {
        return matNormalizedPhase.at<DATA_TYPE>(pt1) < matNormalizedPhase.at<DATA_TYPE>(pt2);
    } );
    VectorOfPoint vecTrimedLocations ( vecPtLocations.begin() + ToInt32 ( vecPtLocations.size() * 0.1 ), vecPtLocations.begin() + ToInt32 ( vecPtLocations.size() * 0.9 ) );
    cv::Mat matX, matY;
    CalcUtils::meshgrid<DATA_TYPE> ( 1.f, 1.f, ToFloat ( COLS ), 1.f, 1.f, ToFloat ( ROWS ), matX, matY );
    std::vector<DATA_TYPE> vecXt, vecYt, vecPhaseTmp;
    vecXt.reserve ( vecTrimedLocations.size() );
    vecYt.reserve ( vecTrimedLocations.size() );
    vecPhaseTmp.reserve ( vecTrimedLocations.size() );
    for ( const auto &point : vecTrimedLocations ) {
        vecXt.push_back ( matX.at<DATA_TYPE>( point ) );
        vecYt.push_back ( matY.at<DATA_TYPE>( point ) );
        vecPhaseTmp.push_back ( matPhase.at<DATA_TYPE>( point ) );
    }
    cv::Mat matXX;
    matXX.push_back ( cv::Mat(vecXt).reshape(1, 1) );
    matXX.push_back ( cv::Mat(vecYt).reshape(1, 1) );
    matXX.push_back ( cv::Mat ( cv::Mat::ones ( 1, ToInt32 ( vecTrimedLocations.size() ), CV_32FC1 ) ) );
    cv::transpose ( matXX, matXX );
    cv::Mat matYY(vecPhaseTmp);
    cv::Mat matK;
    cv::solve ( matXX, matYY, matK, cv::DecompTypes::DECOMP_QR );
#ifdef _DEBUG
    auto vevVecK = CalcUtils::matToVector<DATA_TYPE>(matK);
#endif
    cv::Mat matZp = matX * matK.at<DATA_TYPE>(0, 0) + matY * matK.at<DATA_TYPE>(1, 0) +  matK.at<DATA_TYPE>(2, 0);
    matPhase = matPhase - matZp;

    cv::Mat matST = cv::getStructuringElement ( cv::MORPH_RECT, cv::Size ( ERODE_WIN_SIZE, ERODE_WIN_SIZE ) );

    cv::Mat matHz;
    VectorOfPoint vecPointToFectch;
    vecPointToFectch.emplace_back ( COLS / 2, ROWS / 2 );
    vecPointToFectch.emplace_back ( 0, 0 );
    vecPointToFectch.emplace_back ( COLS/ 2, 0 );
    vecPointToFectch.emplace_back ( 0, ROWS / 2 );
    vecPointToFectch.emplace_back ( COLS - 1, ROWS - 1 );
    for ( int index = 0; index < vecNormaledThreshold.size() - 1; ++ index ) {
        cv::Mat matBW = cv::Mat::zeros( ROWS, COLS, CV_8UC1 );
        cv::inRange ( matNormalizedPhase, vecNormaledThreshold[index], vecNormaledThreshold[index + 1], matBW );
        cv::Mat matFirstRow ( matBW, cv::Rect (0, 0, COLS, 1 ) ); matFirstRow.setTo ( 0 );
        cv::Mat matFirstCol ( matBW, cv::Rect (0, 0, 1, ROWS ) ); matFirstCol.setTo ( 0 );
        cv::erode( matBW, matBW, matST );
        pstRpy->matDivideStepResultImg.setTo ( MATLAB_COLOR_ORDER[index], matBW );

        VectorOfPoint vecPtLocations;
        cv::findNonZero ( matBW, vecPtLocations );
        std::vector<DATA_TYPE> vecXt, vecYt, vecPhaseTmp;
        vecXt.reserve ( vecPtLocations.size () );
        vecYt.reserve ( vecPtLocations.size () );
        vecPhaseTmp.reserve ( vecPtLocations.size () );
        for( const auto &point : vecPtLocations ) {
            vecXt.push_back ( matX.at<DATA_TYPE> ( point ) );
            vecYt.push_back ( matY.at<DATA_TYPE> ( point ) );
            vecPhaseTmp.push_back ( matPhase.at<DATA_TYPE> ( point ) );
        }
        cv::Mat matXX;
        matXX.push_back ( cv::Mat ( vecXt ).reshape ( 1, 1 ) );
        matXX.push_back ( cv::Mat ( vecYt ).reshape ( 1, 1 ) );
        matXX.push_back ( cv::Mat ( cv::Mat::ones ( 1, ToInt32 ( vecPtLocations.size() ), CV_32FC1 ) ) );
        cv::transpose ( matXX, matXX );
        cv::Mat matYY ( vecPhaseTmp );
        matK;
        cv::solve ( matXX, matYY, matK, cv::DecompTypes::DECOMP_QR );
        matZp = matX * matK.at<DATA_TYPE>(0, 0) + matY * matK.at<DATA_TYPE>(1, 0) + matK.at<DATA_TYPE>(2, 0);
        pstRpy->vecMatStepSurface.push_back ( matZp );
        std::vector<DATA_TYPE> vecValue;
        for ( const auto &point : vecPointToFectch )
            vecValue.push_back ( matZp.at<DATA_TYPE> ( point ) );
        matHz.push_back ( cv::Mat ( vecValue ).reshape ( 1, 1 ) );
    }

#ifdef _DEBUG
    auto vevVecHz = CalcUtils::matToVector<DATA_TYPE>( matHz );
#endif

    {//Temparory code scope.
        cv::Mat matXX;
        if( bReverseHeight )
            matXX = CalcUtils::intervals<DATA_TYPE> ( pstCmd->nBlockStepCount, -1, 0 );
        else
            matXX = CalcUtils::intervals<DATA_TYPE> ( 0, 1, pstCmd->nBlockStepCount );

        pstRpy->vecStepPhaseSlope.clear();
        cv::Mat matAllDiff;
        for( int i = 0; i < ToInt32 ( vecPointToFectch.size () ); ++ i ) {
            cv::Mat matYY = cv::Mat ( matHz, cv::Rect ( i, 0, 1, matHz.rows ) ).clone();
            cv::Mat matK;
            cv::solve ( matXX, matYY, matK, cv::DecompTypes::DECOMP_SVD );
            pstRpy->vecStepPhaseSlope.push_back ( matK.at<DATA_TYPE> ( 0, 0 ) );
            cv::Mat matDiff = matYY - matXX * matK;
            cv::transpose(matDiff, matDiff);
            matAllDiff.push_back ( matDiff );
        }
        cv::transpose ( matAllDiff, matAllDiff );
        pstRpy->vecVecStepPhaseDiff = CalcUtils::matToVector<DATA_TYPE>(matAllDiff);
    }

    vecXt = std::vector<DATA_TYPE> ( { ToFloat ( COLS / 2 ), 1.f, ToFloat ( COLS ), 1.f, ToFloat ( COLS ) } );
    matXX = cv::Mat ( vecXt ).reshape (1, 1);
    vecYt = std::vector<DATA_TYPE> ( { ToFloat ( ROWS / 2 ), 1.f, 1.f, ToFloat ( ROWS ), ToFloat ( ROWS ) } );
    matXX.push_back ( cv::Mat ( vecYt ).reshape(1, 1) );
    matXX.push_back ( cv::Mat ( cv::Mat::ones ( 1, matXX.cols, CV_32FC1 ) ) );
    cv::transpose ( matXX, matXX );
    if ( bReverseHeight )
        matYY = cv::Mat ( matHz, cv::Rect ( 0, 0, matHz.cols, 1 ) ).clone();
    else
        matYY = cv::Mat ( matHz, cv::Rect ( 0, matHz.rows - 1, matHz.cols, 1 ) ).clone();
    cv::transpose ( matYY, matYY );
    cv::solve ( matXX, matYY, matK, cv::DecompTypes::DECOMP_QR );
#ifdef _DEBUG
    auto vevVecXX = CalcUtils::matToVector<DATA_TYPE> ( matXX );
    auto vecVecYY = CalcUtils::matToVector<DATA_TYPE> ( matYY );
    vevVecK = CalcUtils::matToVector<DATA_TYPE>(matK);
#endif
    if ( bReverseHeight )
        pstRpy->matPhaseToHeightK = - matK / ( pstCmd->fBlockStepHeight * pstCmd->nBlockStepCount );
    else
        pstRpy->matPhaseToHeightK =   matK / ( pstCmd->fBlockStepHeight * pstCmd->nBlockStepCount );

    cv::Mat matHeight;
    CalcUtils::meshgrid<DATA_TYPE> ( 1.f, 1.f, ToFloat ( matPhase.cols ), 1.f, 1.f, ToFloat ( matPhase.rows ), matX, matY );
    cv::Mat matRatio = pstRpy->matPhaseToHeightK.at<DATA_TYPE> ( 0, 0 ) * matX + pstRpy->matPhaseToHeightK.at<DATA_TYPE> ( 1, 0 ) * matY + pstRpy->matPhaseToHeightK.at<DATA_TYPE> ( 2, 0 );
    cv::divide ( matPhase, matRatio, matHeight );
    pstRpy->matResultImg = _drawHeightGrid ( matHeight, pstCmd->nResultImgGridRow, pstCmd->nResultImgGridCol );
    pstRpy->vecVecStepPhase = CalcUtils::matToVector<DATA_TYPE>(matHz);
}

/*static*/ cv::Mat Unwrap::_drawHeightGrid(const cv::Mat &matHeight, int nGridRow, int nGridCol) {
    double dMinValue = 0, dMaxValue = 0;
    cv::Mat matMask = ( matHeight == matHeight );
    cv::minMaxIdx ( matHeight, &dMinValue, &dMaxValue, 0, 0, matMask );
    
    cv::Mat matNewPhase = matHeight - dMinValue;

    float dRatio = 255.f / ToFloat( dMaxValue - dMinValue );
    matNewPhase = matNewPhase * dRatio;

    cv::Mat matResultImg;
    matNewPhase.convertTo ( matResultImg, CV_8UC1);
    cv::cvtColor ( matResultImg, matResultImg, CV_GRAY2BGR );

    int ROWS = matNewPhase.rows;
    int COLS = matNewPhase.cols;
    int nIntervalX = matNewPhase.cols / nGridCol;
    int nIntervalY = matNewPhase.rows / nGridRow;

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    int thickness = 3;
    for ( int j = 0; j < nGridRow; ++ j ) {
        for ( int i = 0; i < nGridCol; ++ i ) {
            cv::Rect rectROI ( i * nIntervalX, j * nIntervalY, nIntervalX, nIntervalY );
            cv::Mat matROI ( matHeight, rectROI );
            cv::Mat matMask = matROI == matROI;
            float fAverage = ToFloat ( cv::mean ( matROI, matMask )[0] );

            char strAverage[100];
            _snprintf(strAverage, sizeof(strAverage), "%.4f", fAverage);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(strAverage, fontFace, fontScale, thickness, &baseline);
            //The height use '+' because text origin start from left-bottom.
            cv::Point ptTextOrg( rectROI.x + (rectROI.width - textSize.width)/2, rectROI.y + ( rectROI.height + textSize.height ) / 2 );
            cv::putText ( matResultImg, strAverage, ptTextOrg, fontFace, fontScale, cv::Scalar(255, 0, 0), thickness );
        }
    }

    int nGridLineSize = 3;
    cv::Scalar scalarCyan(255, 255, 0);
    for ( int i = 1; i < nGridCol; ++ i )
        cv::line ( matResultImg, cv::Point(i * nIntervalX, 0), cv::Point(i * nIntervalX, ROWS), scalarCyan, nGridLineSize );
    for ( int i = 1; i < nGridRow; ++ i )
        cv::line ( matResultImg, cv::Point(0, i * nIntervalY), cv::Point(COLS, i * nIntervalY), scalarCyan, nGridLineSize );

    return matResultImg;
}

/*static*/ void Unwrap::calcMTF(const PR_CALC_MTF_CMD *const pstCmd, PR_CALC_MTF_RPY *const pstRpy) {
    std::vector<cv::Mat> vecConvertedImgs;
    for ( auto &mat : pstCmd->vecInputImgs ) {
        cv::Mat matConvert = mat;
        if ( mat.channels() > 1 )
            cv::cvtColor ( mat, matConvert, CV_BGR2GRAY );
        matConvert.convertTo ( matConvert, CV_32FC1 );
        vecConvertedImgs.push_back ( matConvert );
    }

    pstRpy->vevVecAbsMtfH.clear();
    pstRpy->vevVecRelMtfH.clear();
    pstRpy->vevVecAbsMtfV.clear();
    pstRpy->vevVecRelMtfV.clear();
    int TOTAL_GROUP = ToInt32 ( pstCmd->vecInputImgs.size() ) / PR_GROUP_TEXTURE_IMG_COUNT;
    for ( int nGroup = 0; nGroup < TOTAL_GROUP; ++ nGroup ) {
        cv::Mat mat00 = vecConvertedImgs[nGroup * PR_GROUP_TEXTURE_IMG_COUNT + 0] - vecConvertedImgs[nGroup * PR_GROUP_TEXTURE_IMG_COUNT + 2];
        cv::Mat mat01 = vecConvertedImgs[nGroup * PR_GROUP_TEXTURE_IMG_COUNT + 3] - vecConvertedImgs[nGroup * PR_GROUP_TEXTURE_IMG_COUNT + 1];

        cv::Mat matPhase;
        cv::phase ( mat00, mat01, matPhase );
        //The opencv cv::phase function return result is 0~2*PI, but matlab is -PI~PI, so here need to do a conversion.
        cv::Mat matOverPi;
        cv::compare ( matPhase, cv::Scalar::all ( CV_PI ), matOverPi, cv::CmpTypes::CMP_GT );
        cv::subtract ( matPhase, cv::Scalar::all ( 2.f * CV_PI ), matPhase, matOverPi );

        matPhase = _phaseUnwrapSurface ( matPhase );
        cv::Mat matSin = CalcUtils::sin<DATA_TYPE> ( matPhase );
        cv::Mat matCos = CalcUtils::cos<DATA_TYPE> ( matPhase );
        std::vector<float> vecMtfAbsH ( matPhase.rows, 1.f );
        std::vector<float> vecMtfRelH ( matPhase.rows, 1.f );
        for ( int row = 0; row < matPhase.rows; ++ row ) {
            cv::Rect rectROI(0, row, matPhase.cols, 1 );
            cv::Mat matYY ( mat00, rectROI );
            cv::Mat matYYT;
            cv::transpose ( matYY, matYYT );
            cv::Mat matXX;
            matXX.push_back ( cv::Mat( matSin, rectROI).clone() );
            matXX.push_back ( cv::Mat( matCos, rectROI).clone() );
            matXX.push_back ( cv::Mat( cv::Mat::ones( 1, matPhase.cols, matPhase.type() ) ) );
            cv::transpose ( matXX, matXX );
            cv::Mat matK;
            cv::solve ( matXX, matYYT, matK, cv::DecompTypes::DECOMP_SVD );
            vecMtfAbsH[row] = std::sqrt ( matK.at<DATA_TYPE>(0, 0) * matK.at<DATA_TYPE>(0, 0) + matK.at<DATA_TYPE>(1, 0) * matK.at<DATA_TYPE>(1, 0) ) / pstCmd->fMagnitudeOfDLP;
            if ( nGroup > 0 )
                vecMtfRelH[row] = vecMtfAbsH[row] / pstRpy->vevVecAbsMtfH[0][row];
        }
        pstRpy->vevVecAbsMtfH.push_back ( vecMtfAbsH );
        pstRpy->vevVecRelMtfH.push_back ( vecMtfRelH );

        cv::transpose ( matPhase, matPhase );
        cv::transpose ( mat00, mat00 );
        cv::transpose ( matSin, matSin );
        cv::transpose ( matCos, matCos );

        std::vector<float> vecMtfAbsV ( matPhase.rows, 1.f );
        std::vector<float> vecMtfRelV ( matPhase.rows, 1.f );
        for ( int row = 0; row < matPhase.rows; ++ row ) {
            cv::Rect rectROI(0, row, matPhase.cols, 1 );
            cv::Mat matYY ( mat00, rectROI );
            cv::Mat matYYT;
            cv::transpose ( matYY, matYYT );
            cv::Mat matXX;
            matXX.push_back ( cv::Mat( matSin, rectROI).clone() );
            matXX.push_back ( cv::Mat( matCos, rectROI).clone() );
            matXX.push_back ( cv::Mat( cv::Mat::ones( 1, matPhase.cols, matPhase.type() ) ) );
            cv::transpose ( matXX, matXX );
            cv::Mat matK;
            cv::solve ( matXX, matYYT, matK, cv::DecompTypes::DECOMP_SVD );
            vecMtfAbsV[row] = std::sqrt ( matK.at<DATA_TYPE>(0, 0) * matK.at<DATA_TYPE>(0, 0) + matK.at<DATA_TYPE>(1, 0) * matK.at<DATA_TYPE>(1, 0) ) / pstCmd->fMagnitudeOfDLP;
            if ( nGroup > 0 )
                vecMtfRelV[row] = vecMtfAbsV[row] / pstRpy->vevVecAbsMtfV[0][row];
        }
        pstRpy->vevVecAbsMtfV.push_back ( vecMtfAbsV );
        pstRpy->vevVecRelMtfV.push_back ( vecMtfRelV );
    }
}

}
}