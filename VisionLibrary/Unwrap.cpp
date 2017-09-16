#include "Unwrap.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "CalcUtils.h"
#include "VisionAlgorithm.h"
#include "TimeLog.h"
#include "Log.h"
#include <numeric>

#define MARK_FUNCTION_START_TIME    CStopWatch      stopWatch; __int64 functionStart = stopWatch.AbsNow()
#define MARK_FUNCTION_END_TIME      TimeLog::GetInstance()->addTimeLog( __FUNCTION__, stopWatch.AbsNow() - functionStart )

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

/*static*/ const float Unwrap::GAUSSIAN_FILTER_SIGMA        = ToFloat ( pow ( 20, 0.5 ) );
/*static*/ const float Unwrap::ONE_HALF_CYCLE               = ToFloat ( CV_PI );
/*static*/ const float Unwrap::CALIB_HEIGHT_STEP_USEFUL_PT  =  0.8f;
/*static*/ const float Unwrap::REMOVE_HEIGHT_NOSIE_RATIO    =  0.995f;
/*static*/ const float Unwrap::LOW_BASE_PHASE               = - ToFloat( CV_PI );

/*static*/ cv::Mat Unwrap::_getResidualPoint(const cv::Mat &matInput, cv::Mat &matPosPole, cv::Mat &matNegPole) {
    MARK_FUNCTION_START_TIME;

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

    int nSizeOfResultMat = ( matInput.rows - 1 ) * ( matInput.cols - 1 );
    cv::Mat matResult = cv::Mat::zeros ( nSizeOfResultMat, 1, CV_8SC1 );
    matResult.setTo( cv::Scalar::all(1), matPosIndex );
    matResult.setTo( cv::Scalar::all(-1), matNegIndex );
    matResult = matResult.reshape ( 1, matInput.rows - 1 );

    matPosPole = matPosIndex.clone().reshape(1, matInput.rows - 1 );
    matNegPole = matNegIndex.clone().reshape(1, matInput.rows - 1 );

    MARK_FUNCTION_END_TIME;
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
        if ( ! vecIter1.empty() ) {
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

/*static*/ VectorOfPoint Unwrap::_getIntegralTree(const VectorOfPoint &vecResult1, const VectorOfPoint &vecResult2, const VectorOfPoint &vecResult3, int rows, int cols ) {
    MARK_FUNCTION_START_TIME;

    VectorOfPoint vecResult;
    for ( size_t i = 0; i < vecResult1.size(); ++ i ) {
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
        
        for ( size_t k = 0; k < vecTrkXOne.size(); ++ k )
            vecResult.emplace_back ( vecTrkXOne[k], vecTrkYOne[k] );

        for ( size_t k = 0; k < vecTrkXTwo.size(); ++ k )
            vecResult.emplace_back ( vecTrkXTwo[k], vecTrkYTwo[k] );
    }

    if ( ! vecResult3.empty() ) {
        for ( const auto &point : vecResult3 ) {
            std::vector<int> vecDist({ point.x, cols - point.x, point.y, rows - point.y});
            auto iterMin = std::min_element ( vecDist.begin(), vecDist.end() );
            int nIndex = ToInt32 ( std::distance ( vecDist.begin(), iterMin ) );
            if ( 0 == nIndex ) {
                for ( int i = 0; i <= point.x; ++ i )
                    vecResult.emplace_back ( i, point.y );
            }else if ( 1 == nIndex ) {
                for ( int i = point.x; i < cols; ++ i )
                    vecResult.emplace_back ( i, point.y );
            }else if ( 2 == nIndex ) {
                for ( int i = 0; i <= point.y; ++ i )
                    vecResult.emplace_back ( point.x, i );
            }else if ( 3 == nIndex ) {
                for ( int i = point.y; i < rows; ++ i )
                    vecResult.emplace_back ( point.x, i );
            }
        }
    }

    MARK_FUNCTION_END_TIME;
    return vecResult;
}

/*static*/ void Unwrap::calib3DBase( const PR_CALIB_3D_BASE_CMD *const pstCmd, PR_CALIB_3D_BASE_RPY *const pstRpy ) {
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
            cv::GaussianBlur ( mat, mat, cv::Size(GUASSIAN_FILTER_SIZE, GUASSIAN_FILTER_SIZE), GAUSSIAN_FILTER_SIGMA, GAUSSIAN_FILTER_SIGMA, cv::BorderTypes::BORDER_REPLICATE );
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

    matAlpha = _phaseUnwrapSurface ( matAlpha );
    matBeta  = _phaseUnwrapSurface ( matBeta );

    if ( fabs ( pstCmd->fRemoveHarmonicWaveK ) > 1e-5 )
        matAlpha = matAlpha + CalcUtils::sin<DATA_TYPE>(4.f * matAlpha) * pstCmd->fRemoveHarmonicWaveK;

    cv::Mat matSnoop(matAlpha, cv::Rect(0, 0, PHASE_SNOOP_WIN_SIZE, PHASE_SNOOP_WIN_SIZE ) );
    pstRpy->fBaseStartAvgPhase = ToFloat ( cv::mean ( matSnoop )[0] );

    cv::Mat matAlphaReshape = matAlpha.reshape ( 1, 1 );
    cv::Mat matYY = matBeta.reshape ( 1, matBeta.rows * matBeta.cols );
    cv::Mat matXX = matAlphaReshape.clone();
    cv::Mat matOne = cv::Mat::ones(1, matAlpha.rows * matAlpha.cols, CV_32FC1);
    matXX.push_back ( matOne );
    cv::transpose ( matXX, matXX );
    cv::solve ( matXX, matYY, pstRpy->matThickToThinStripeK, cv::DecompTypes::DECOMP_QR );
#ifdef _DEBUG
    auto vecVecAlpha = CalcUtils::matToVector<DATA_TYPE>(matAlpha);
    auto vecVecBeta  = CalcUtils::matToVector<DATA_TYPE>(matBeta);
    auto vecVecMatK  = CalcUtils::matToVector<DATA_TYPE>(pstRpy->matThickToThinStripeK);
#endif
    int ROWS = pstCmd->vecInputImgs[0].rows;
    int COLS = pstCmd->vecInputImgs[0].cols;
    cv::Mat matX, matY;
    CalcUtils::meshgrid<float> ( 1.f, 1.f, ToFloat ( COLS ), 1.f, 1.f, ToFloat ( ROWS ), matX, matY );

    pstRpy->matBaseSurfaceParam = _calculatePPz ( matX, matY, matBeta );
    cv::Mat matZP1 = _calculateBaseSurface ( matAlpha.rows, matAlpha.cols, pstRpy->matBaseSurfaceParam );
#ifdef _DEBUG
    auto vecPPz = CalcUtils::matToVector<DATA_TYPE>(pstRpy->matBaseSurfaceParam);
    auto vecZP1 = CalcUtils::matToVector<DATA_TYPE>(matZP1);
#endif

    pstRpy->enStatus = VisionStatus::OK;
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
    CStopWatch stopWatch;
    std::vector<cv::Mat> vecConvertedImgs;
    for ( auto &mat : pstCmd->vecInputImgs ) {
        cv::Mat matConvert = mat;
        if ( mat.channels() > 1 )
            cv::cvtColor ( mat, matConvert, CV_BGR2GRAY );
        matConvert.convertTo ( matConvert, CV_32FC1 );
        vecConvertedImgs.push_back ( matConvert );
    }

    TimeLog::GetInstance()->addTimeLog( "Convert image to float.", stopWatch.Span() );

    if ( pstCmd->bEnableGaussianFilter ) {
        //Filtering can reduce residues at the expense of a loss of spatial resolution.
        for ( auto &mat : vecConvertedImgs )        
            cv::GaussianBlur ( mat, mat, cv::Size(GUASSIAN_FILTER_SIZE, GUASSIAN_FILTER_SIZE), GAUSSIAN_FILTER_SIGMA, GAUSSIAN_FILTER_SIGMA, cv::BorderTypes::BORDER_REPLICATE );
    }

    TimeLog::GetInstance()->addTimeLog( "Guassian Filter.", stopWatch.Span() );

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

    VectorOfPoint vecPosPole, vecNegPole;
    cv::findNonZero ( matPosPole, vecPosPole );
    cv::findNonZero ( matNegPole, vecNegPole );

    VectorOfPoint vecPoint1, vecPoint2, vecPoint3;
    _selectPolePair ( vecPosPole, vecNegPole, ROWS - 1, COLS - 1, vecPoint1, vecPoint2, vecPoint3 );

    VectorOfPoint vecTree = _getIntegralTree ( vecPoint1, vecPoint2, vecPoint3, ROWS, COLS );
    cv::Mat matBranchCut = cv::Mat::zeros ( ROWS, COLS, CV_8UC1 );
    for ( const auto &point : vecTree )
        matBranchCut.at<uchar>(point) = 255;

#ifdef _DEBUG
    cv::imwrite("./data/BranchCut_1.png", matBranchCut );
#endif

    matAlpha = _phaseUnwrapSurfaceTrk ( matAlpha, matBranchCut );

    cv::Mat matSnoop(matAlpha, cv::Rect(0, 0, PHASE_SNOOP_WIN_SIZE, PHASE_SNOOP_WIN_SIZE ) );
    auto fStartAvgPhase = ToFloat ( cv::mean ( matSnoop )[0] );
    if ( ( fStartAvgPhase - pstCmd->fBaseStartAvgPhase ) > ( LOW_BASE_PHASE + ToFloat ( CV_PI ) * 2.f ) )
        matAlpha = matAlpha - ToFloat ( CV_PI ) * 2.f;
    else if ( ( fStartAvgPhase - pstCmd->fBaseStartAvgPhase ) < LOW_BASE_PHASE )
        matAlpha = matAlpha + ToFloat ( CV_PI ) * 2.f;

    if ( fabs ( pstCmd->fRemoveHarmonicWaveK ) > 1e-5 )
        matAlpha = matAlpha + CalcUtils::sin<DATA_TYPE>(4.f * matAlpha) * pstCmd->fRemoveHarmonicWaveK;

#ifdef _DEBUG
    auto vevVecBranchCut = CalcUtils::matToVector<uchar>(matBranchCut);
    auto vecVecAlpha = CalcUtils::matToVector<DATA_TYPE>(matAlpha);
#endif
    cv::Mat matDiffUnderTolIndex, matAvgUnderTolIndex;
    _findUnstablePoint ( vecConvertedImgs, pstCmd->fMinIntensityDiff, pstCmd->fMinAvgIntensity, matDiffUnderTolIndex, matAvgUnderTolIndex );
#ifdef _DEBUG
    int nDiffUnderTolCount = cv::countNonZero ( matDiffUnderTolIndex );
    int nAvgUnderTolCount  = cv::countNonZero ( matAvgUnderTolIndex );
    auto vecVecBeta  = CalcUtils::matToVector<DATA_TYPE>(matBeta);
#endif
    matAlpha.setTo ( NAN, matDiffUnderTolIndex );
    matAlpha = matAlpha * pstCmd->matThickToThinStripeK.at<DATA_TYPE>(0, 0) + pstCmd->matThickToThinStripeK.at<DATA_TYPE>(1, 0);    
    matBeta = _phaseUnwrapSurfaceByRefer ( matBeta, matAlpha );

    cv::Mat matZP1 = _calculateBaseSurface ( matAlpha.rows, matAlpha.cols, pstCmd->matBaseSurfaceParam );
#ifdef _DEBUG
    vecVecAlpha = CalcUtils::matToVector<DATA_TYPE>(matAlpha);
    auto vecVecZP1 = CalcUtils::matToVector<DATA_TYPE>(matZP1);
    vecVecBeta  = CalcUtils::matToVector<DATA_TYPE>(matBeta);
#endif
    matAlpha = matAlpha - matZP1;
    matBeta  = matBeta  - matZP1;

    auto matH = matBeta.clone();
    matH.setTo ( NAN, matAvgUnderTolIndex );
    matH.setTo ( NAN, matDiffUnderTolIndex );

#ifdef _DEBUG
    auto vecVecH = CalcUtils::matToVector<DATA_TYPE>( matH );
#endif

    stopWatch.Start();
    cv::medianBlur ( matH, matH, 5 );
    TimeLog::GetInstance()->addTimeLog( "MedianBlur.", stopWatch.Span() );
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
    }else
        pstRpy->matHeight = matH;

    pstRpy->enStatus = VisionStatus::OK;
}

/*static*/ cv::Mat Unwrap::_phaseUnwrapSurfaceTrk ( const cv::Mat &matPhase, const cv::Mat &matBranchCut ) {
    MARK_FUNCTION_START_TIME;

    int ROWS = matPhase.rows;
    int COLS = matPhase.cols;

    cv::Mat matPhaseResult;
    const float OneCycle = 2.f * ONE_HALF_CYCLE;
    cv::Mat matUnwrapped = matBranchCut.clone();

    cv::Mat dyPhase = CalcUtils::diff ( matPhase, 1, 1 ); //The size of dyPhase is [ROWS - 1, COLS]
    cv::Mat dxPhase = CalcUtils::diff ( matPhase, 1, 2 ); //The size of dxPhase is [ROWS, COLS - 1]

    //1. Unwrap from left to right, but to accelerate the operation, transpose the matrix first,
    //and do it from top to bottom, it is the same effect as from left to right on original phase.
    cv::Mat matPhaseT;
    cv::transpose ( matPhase, matPhaseT );

    cv::Mat matTmpPhase = cv::Mat ( matPhaseT, cv::Rect ( 0, 0, matPhaseT.cols, 1 ) );
    cv::Mat matDp = CalcUtils::diff ( matTmpPhase, 1, 2 );

#ifdef _DEBUG
    auto vecVecArray = CalcUtils::matToVector<DATA_TYPE>( dxPhase );
    auto vecmatTmp = CalcUtils::matToVector<DATA_TYPE>( matTmpPhase );
    auto vecVecMatDp_1 = CalcUtils::matToVector<DATA_TYPE>( matDp );
#endif

    cv::Mat matAbsUnderPI;
    cv::compare ( cv::abs ( matDp ), cv::Scalar::all ( ONE_HALF_CYCLE ), matAbsUnderPI, cv::CmpTypes::CMP_LT );
    matDp.setTo ( 0, matAbsUnderPI );

    matDp = _setBySign ( matDp, OneCycle);
    matDp = cv::repeat ( matDp, matPhase.cols, 1);

    cv::Mat matTmp ( matPhaseT, cv::Rect(1, 0, matPhaseT.cols - 1, matPhaseT.rows ) );
    auto matCumSum = CalcUtils::cumsum<DATA_TYPE> ( matDp, 2 );
    matTmp += matCumSum;

    //2. Unwrap from top to bottom. Because last step do a transpose, this step need to transpose back.
    cv::transpose ( matPhaseT, matPhaseResult );
    matPhaseResult.setTo ( cv::Scalar::all(NAN), matBranchCut );

    matDp = CalcUtils::diff ( matPhaseResult, 1, 2 );
    cv::compare ( cv::abs ( matDp ), cv::Scalar::all ( ONE_HALF_CYCLE ), matAbsUnderPI, cv::CmpTypes::CMP_LT );
    matDp.setTo ( 0, matAbsUnderPI );
    matDp = _setBySign ( matDp, OneCycle );

    cv::Mat matTmp1 ( matPhaseResult,cv::Rect(1, 0, matPhaseResult.cols - 1, matPhaseResult.rows ) );
    matCumSum = CalcUtils::cumsum<DATA_TYPE> ( matDp, 2 );
    matTmp1 += matCumSum;

#ifdef _DEBUG
    auto vecVecCumSum = CalcUtils::matToVector<DATA_TYPE> ( matCumSum );
#endif

    //3. Unwrap from Top to Bottom. Use the last row to unwrap next row.
    matPhaseResult.setTo ( cv::Scalar(0), matBranchCut );    
    cv::Mat matNan = CalcUtils::getNanMask ( matPhaseResult );  //Find out points been blocked by branch cut.
    cv::Mat matIsRowNeedUnwrap;
    cv::reduce ( matNan, matIsRowNeedUnwrap, 1, CV_REDUCE_MAX );
    std::vector<cv::Point> vecRowsNeedToUnwrap;
    cv::findNonZero ( matIsRowNeedUnwrap, vecRowsNeedToUnwrap );
    for ( const auto &point : vecRowsNeedToUnwrap ) {
        int row = point.y;
        cv::Mat matOneRowIndex ( matNan, cv::Range ( row, row + 1 ) );
        if ( cv::sum ( matOneRowIndex )[0] <= 1 || row <= 0 )
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

    //4. Unwrap from bottom to top. Use the next row to unwrap last row.
    matPhaseResult.setTo ( cv::Scalar(0), matBranchCut );    
    matNan = CalcUtils::getNanMask ( matPhaseResult );  //Find out points been blocked by branch cut.

    cv::reduce ( matNan, matIsRowNeedUnwrap, 1, CV_REDUCE_MAX );
    vecRowsNeedToUnwrap.clear();
    cv::findNonZero ( matIsRowNeedUnwrap, vecRowsNeedToUnwrap );
    std::vector<int> vecRows;
    for ( const auto &point : vecRowsNeedToUnwrap )
        vecRows.push_back (point.y);
    std::sort(vecRows.begin(), vecRows.end() );
    std::reverse( vecRows.begin(), vecRows.end() );
    for ( const auto &row : vecRows ) {
        cv::Mat matOneRowIndex ( matNan, cv::Range ( row, row + 1 ) );
        if ( cv::sum ( matOneRowIndex )[0] <= 1 || row >= ROWS - 1 )
            continue;
        cv::Mat matPhaseNextRow ( matPhaseResult, cv::Range ( row + 1, row + 2 ) );
        cv::Mat matPhaseDiff(dyPhase, cv::Range ( row, row + 1 ) );

        cv::Mat matZero;
        cv::compare ( matPhaseNextRow, cv::Scalar(0), matZero, cv::CmpTypes::CMP_EQ );
        if ( cv::sum ( matZero )[0] > 0 )
            matPhaseNextRow.setTo ( cv::Scalar(NAN), matZero );

        matDp = matPhaseDiff;
        cv::compare ( cv::abs ( matDp ), cv::Scalar ( ONE_HALF_CYCLE ), matAbsUnderPI, cv::CmpTypes::CMP_LT );
        matDp.setTo ( 0, matAbsUnderPI );
        matDp = _setBySign ( matDp, OneCycle );
        
        cv::Mat matPhaseNextRowClone = matPhaseNextRow.clone();
        cv::accumulate ( matPhaseDiff, matPhaseNextRowClone, matOneRowIndex );
        cv::accumulate ( matDp,        matPhaseNextRowClone, matOneRowIndex );

        cv::Mat matOneRowPhase ( matPhaseResult, cv::Range ( row, row + 1 ) );
        matPhaseNextRowClone.copyTo ( matOneRowPhase, matOneRowIndex );
    }

    //5. Unwrap from right to left. One column by one column.
    matPhaseResult.setTo ( cv::Scalar(0), matBranchCut );
    matNan = CalcUtils::getNanMask ( matPhaseResult );

    cv::Mat matIsColNeedUnwrap;
    cv::reduce ( matNan, matIsColNeedUnwrap, 0, CV_REDUCE_MAX );
    std::vector<cv::Point> vecColsNeedToUnwrap;
    cv::findNonZero ( matIsColNeedUnwrap, vecColsNeedToUnwrap );
    std::vector<int> vecCols;
    for ( const auto &point : vecColsNeedToUnwrap )
        vecCols.push_back ( point.x );
    std::sort( vecCols.begin(), vecCols.end() );
    std::reverse( vecCols.begin(), vecCols.end() );
    for ( const auto col : vecCols ) {
        cv::Mat matOneColIndex ( matNan, cv::Range::all(), cv::Range ( col, col + 1 ) );
        if ( cv::sum ( matOneColIndex )[0] <= 1 || col >= COLS - 1 )
            continue;
        cv::Mat matPhaseNextCol ( matPhaseResult, cv::Range::all(), cv::Range ( col + 1, col + 2 ) );
        cv::Mat matPhaseDiff ( dxPhase, cv::Range::all(), cv::Range ( col, col + 1 ) );

        cv::Mat matZero;
        cv::compare ( matPhaseNextCol, cv::Scalar(0), matZero, cv::CmpTypes::CMP_EQ );
        if ( cv::sum ( matZero )[0] > 0 )
            matPhaseNextCol.setTo ( cv::Scalar(NAN), matZero );

        matDp = matPhaseDiff;
        cv::compare ( cv::abs(matDp), cv::Scalar::all(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT );
        matDp.setTo ( 0, matAbsUnderPI );
        matDp = _setBySign ( matDp, OneCycle );
        
        cv::Mat matPhaseNextColClone = matPhaseNextCol.clone();
        cv::accumulate ( matPhaseDiff, matPhaseNextColClone, matOneColIndex );
        cv::accumulate ( matDp,        matPhaseNextColClone, matOneColIndex );

        cv::Mat matOneColPhase ( matPhaseResult, cv::Range::all(), cv::Range ( col, col + 1 ) );
        matPhaseNextColClone.copyTo ( matOneColPhase, matOneColIndex );
    }

#ifdef _DEBUG
    matNan = CalcUtils::getNanMask ( matPhaseResult );
    cv::imwrite("./data/NanMap_2.png", matNan);
#endif

    matPhaseResult.setTo ( cv::Scalar(NAN), matBranchCut );

#ifdef _DEBUG
    matNan = CalcUtils::getNanMask ( matPhaseResult );
    cv::imwrite("./data/NanMap_3.png", matNan);
#endif

    MARK_FUNCTION_END_TIME;

    return matPhaseResult;
}

/*static*/ cv::Mat Unwrap::_phaseUnwrapSurfaceByRefer(const cv::Mat &matPhase, const cv::Mat &matRef ) {
    MARK_FUNCTION_START_TIME;

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

    MARK_FUNCTION_END_TIME;
    return matPhaseResult;
}

//For shadow area
/*static*/ void Unwrap:: _findUnstablePoint(const std::vector<cv::Mat> &vecInputImgs, float fDiffTol, float fAvgTol, cv::Mat &matDiffUnderTolIndex, cv::Mat &matAvgUnderTolIndex) {
    MARK_FUNCTION_START_TIME;

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

    MARK_FUNCTION_END_TIME;
}

/*static*/ void Unwrap::calib3DHeight(const PR_CALIB_3D_HEIGHT_CMD *const pstCmd, PR_CALIB_3D_HEIGHT_RPY *const pstRpy) {
    PR_CALC_3D_HEIGHT_CMD stCalcCmd;
    PR_CALC_3D_HEIGHT_RPY stCalcRpy;
    stCalcCmd.vecInputImgs = std::move ( pstCmd->vecInputImgs );
    stCalcCmd.bEnableGaussianFilter = pstCmd->bEnableGaussianFilter;
    stCalcCmd.bReverseSeq = pstCmd->bReverseSeq;
    stCalcCmd.fRemoveHarmonicWaveK = pstCmd->fRemoveHarmonicWaveK;
    stCalcCmd.fMinIntensityDiff = pstCmd->fMinIntensityDiff;
    stCalcCmd.fMinAvgIntensity = pstCmd->fMinAvgIntensity;
    stCalcCmd.matThickToThinStripeK = pstCmd->matThickToThinStripeK;
    stCalcCmd.matBaseSurfaceParam = pstCmd->matBaseSurfaceParam;
    stCalcCmd.fBaseStartAvgPhase = pstCmd->fBaseStartAvgPhase;
    calc3DHeight ( &stCalcCmd, &stCalcRpy );
    cv::Mat matPhase = stCalcRpy.matPhase;
    cv::Mat matTmpPhase = matPhase.clone();
    int ROWS = matPhase.rows;
    int COLS = matPhase.cols;
    cv::Mat matST = cv::getStructuringElement ( cv::MORPH_RECT, cv::Size ( ERODE_WIN_SIZE, ERODE_WIN_SIZE ) );

    const int NN = 2;
    for ( int n = 1; n <= NN; ++ n ) {
        double dMinValue = 0, dMaxValue = 0;
        cv::Mat matMask = ( matTmpPhase == matTmpPhase ); //Find out value is not NAN.
        cv::minMaxIdx ( matTmpPhase, &dMinValue, &dMaxValue, 0, 0, matMask );

        cv::Mat matNormalizedPhase = (matTmpPhase - dMinValue) / (dMaxValue - dMinValue);
        cv::Mat matGrayScalePhase = matNormalizedPhase * PR_MAX_GRAY_LEVEL;
        matGrayScalePhase.convertTo ( pstRpy->matDivideStepResultImg, CV_8U );
        cv::cvtColor ( pstRpy->matDivideStepResultImg, pstRpy->matDivideStepResultImg, CV_GRAY2BGR );
        auto vecThreshold = VisionAlgorithm::autoMultiLevelThreshold ( matGrayScalePhase, matMask, pstCmd->nBlockStepCount );
        std::vector<float> vecNormaledThreshold;
        vecNormaledThreshold.push_back ( 0.f );
        for (const auto thresh : vecThreshold)
            vecNormaledThreshold.push_back ( ToFloat ( thresh ) / ToFloat ( PR_MAX_GRAY_LEVEL ) );
        vecNormaledThreshold.push_back ( 1.f );

        auto nKK = 0;
        if (pstCmd->bReverseHeight)
            nKK = ToInt32 ( vecNormaledThreshold.size () ) - 2;
        cv::Mat matInRange;
        cv::inRange ( matNormalizedPhase, vecNormaledThreshold[nKK], vecNormaledThreshold[nKK + 1], matInRange );
#ifdef _DEBUG
        auto vevVecNormalizedPhase = CalcUtils::matToVector<DATA_TYPE> ( matNormalizedPhase );
#endif
        auto rectBounding = cv::boundingRect ( matInRange );
        if ( rectBounding.width < CALIB_HEIGHT_MIN_SIZE || rectBounding.height < CALIB_HEIGHT_MIN_SIZE ) {
            WriteLog("The base surface size is too small in calibrate 3D height.");
            pstRpy->enStatus = VisionStatus::CALIB_3D_HEIGHT_SURFACE_TOO_SMALL;
            return;
        }

        VectorOfPoint vecPtLocations;
        cv::findNonZero ( matInRange, vecPtLocations );
        std::sort ( vecPtLocations.begin (), vecPtLocations.end (), [&matNormalizedPhase]( const cv::Point &pt1, const cv::Point &pt2 ) {
            return matNormalizedPhase.at<DATA_TYPE> ( pt1 ) < matNormalizedPhase.at<DATA_TYPE> ( pt2 );
        } );
        VectorOfPoint vecTrimedLocations ( vecPtLocations.begin () + ToInt32 ( vecPtLocations.size () * 0.1 ), vecPtLocations.begin () + ToInt32 ( vecPtLocations.size () * 0.9 ) );
        cv::Mat matX, matY;
        CalcUtils::meshgrid<DATA_TYPE> ( 1.f, 1.f, ToFloat ( COLS ), 1.f, 1.f, ToFloat ( ROWS ), matX, matY );
        std::vector<DATA_TYPE> vecXt, vecYt, vecPhaseTmp;
        vecXt.reserve ( vecTrimedLocations.size () );
        vecYt.reserve ( vecTrimedLocations.size () );
        vecPhaseTmp.reserve ( vecTrimedLocations.size () );
        for (const auto &point : vecTrimedLocations) {
            vecXt.push_back ( matX.at<DATA_TYPE> ( point ) );
            vecYt.push_back ( matY.at<DATA_TYPE> ( point ) );
            vecPhaseTmp.push_back ( matPhase.at<DATA_TYPE> ( point ) );
        }
        cv::Mat matXX;
        matXX.push_back ( cv::Mat ( vecXt ).reshape ( 1, 1 ) );
        matXX.push_back ( cv::Mat ( vecYt ).reshape ( 1, 1 ) );
        matXX.push_back ( cv::Mat ( cv::Mat::ones ( 1, ToInt32 ( vecTrimedLocations.size () ), CV_32FC1 ) ) );
        cv::transpose ( matXX, matXX );
        cv::Mat matYY ( vecPhaseTmp );
        cv::Mat matK;
        cv::solve ( matXX, matYY, matK, cv::DecompTypes::DECOMP_QR );
#ifdef _DEBUG
        auto vevVecK = CalcUtils::matToVector<DATA_TYPE> ( matK );
#endif
        cv::Mat matZp = matX * matK.at<DATA_TYPE> ( 0, 0 ) + matY * matK.at<DATA_TYPE> ( 1, 0 ) + matK.at<DATA_TYPE> ( 2, 0 );
        matPhase = matPhase - matZp;

        cv::Mat matHz;
        VectorOfPoint vecPointToFectch;
        vecPointToFectch.emplace_back ( COLS / 2, ROWS / 2 );
        vecPointToFectch.emplace_back ( 0, 0 );
        vecPointToFectch.emplace_back ( COLS - 1, 0 );
        vecPointToFectch.emplace_back ( 0, ROWS - 1 );
        vecPointToFectch.emplace_back ( COLS - 1, ROWS - 1 );
        for (size_t index = 0; index < vecNormaledThreshold.size () - 1; ++ index) {
            cv::Mat matBW = cv::Mat::zeros ( ROWS, COLS, CV_8UC1 );
            cv::inRange ( matNormalizedPhase, vecNormaledThreshold[index], vecNormaledThreshold[index + 1], matBW );
            cv::Mat matFirstRow ( matBW, cv::Rect ( 0, 0, COLS, 1 ) ); matFirstRow.setTo ( 0 );
            cv::Mat matFirstCol ( matBW, cv::Rect ( 0, 0, 1, ROWS ) ); matFirstCol.setTo ( 0 );
            cv::Mat matLastRow ( matBW, cv::Rect ( 0, ROWS - 1, COLS, 1 ) ); matLastRow.setTo ( 0 );
            cv::Mat matLastCol ( matBW, cv::Rect ( COLS - 1, 0, 1, ROWS ) ); matLastCol.setTo ( 0 );
            cv::erode ( matBW, matBW, matST );
            pstRpy->matDivideStepResultImg.setTo ( MATLAB_COLOR_ORDER[index], matBW );

            VectorOfPoint vecPtLocations;
            cv::findNonZero ( matBW, vecPtLocations );
            for ( int i = 0; i < 2; ++ i) {
                std::vector<DATA_TYPE> vecXt, vecYt, vecPhaseTmp;
                vecXt.reserve ( vecPtLocations.size () );
                vecYt.reserve ( vecPtLocations.size () );
                vecPhaseTmp.reserve ( vecPtLocations.size () );
                for (const auto &point : vecPtLocations) {
                    vecXt.push_back ( matX.at<DATA_TYPE> ( point ) );
                    vecYt.push_back ( matY.at<DATA_TYPE> ( point ) );
                    vecPhaseTmp.push_back ( matPhase.at<DATA_TYPE> ( point ) );
                }
                cv::Mat matXX;
                matXX.push_back ( cv::Mat ( vecXt ).reshape ( 1, 1 ) );
                matXX.push_back ( cv::Mat ( vecYt ).reshape ( 1, 1 ) );
                matXX.push_back ( cv::Mat ( cv::Mat::ones ( 1, ToInt32 ( vecPtLocations.size () ), CV_32FC1 ) ) );
                cv::transpose ( matXX, matXX );
                cv::Mat matYY ( vecPhaseTmp );
                cv::solve ( matXX, matYY, matK, cv::DecompTypes::DECOMP_SVD );
                if ( i == 0 ) {
                    cv::Mat matErr = cv::abs ( matYY - matXX * matK );
                    cv::Mat matErrSort;
                    cv::sortIdx ( matErr, matErrSort, cv::SortFlags::SORT_ASCENDING + cv::SortFlags::SORT_EVERY_COLUMN );
#ifdef _DEBUG
                    cv::Mat matErrSortTranspose;
                    cv::transpose ( matErrSort, matErrSortTranspose );
                    auto vevVecSortIndex = CalcUtils::matToVector<int>(matErrSortTranspose);
#endif
                    VectorOfPoint vecTmpPoints;
                    int nKeepPoints = ToInt32 ( std::floor ( matErrSort.rows * CALIB_HEIGHT_STEP_USEFUL_PT ) );
                    vecTmpPoints.reserve ( nKeepPoints );
                    for (int j = 0; j < nKeepPoints; ++ j)
                        vecTmpPoints.push_back ( vecPtLocations[ matErrSort.at<int>(j)] );
                    std::swap ( vecTmpPoints, vecPtLocations );
                }
            }
#ifdef _DEBUG
            auto vevVecK = CalcUtils::matToVector<DATA_TYPE> ( matK );
#endif
            matZp = matX * matK.at<DATA_TYPE> ( 0, 0 ) + matY * matK.at<DATA_TYPE> ( 1, 0 ) + matK.at<DATA_TYPE> ( 2, 0 );
            pstRpy->vecMatStepSurface.push_back ( matZp );
            std::vector<DATA_TYPE> vecValue;
            for (const auto &point : vecPointToFectch)
                vecValue.push_back ( matZp.at<DATA_TYPE> ( point ) );
            matHz.push_back ( cv::Mat ( vecValue ).reshape ( 1, 1 ) );
        }

        pstRpy->vecVecStepPhase = CalcUtils::matToVector<DATA_TYPE> ( matHz );

        {//Temparory code scope.
            cv::Mat matXp, matXpT, matXt0T, matYt0T, matHzT;
            if (pstCmd->bReverseHeight)
                matXp = CalcUtils::intervals<DATA_TYPE> ( pstCmd->nBlockStepCount, -1, 0 );
            else
                matXp = CalcUtils::intervals<DATA_TYPE> ( 0, 1, pstCmd->nBlockStepCount );
            matXp = cv::repeat ( matXp, 1, 5 );
            cv::transpose ( matXp, matXpT );
            matXpT = matXpT.reshape ( 1, 1 );

            vecXt = std::vector<DATA_TYPE> ( { ToFloat ( COLS / 2 ), 1.f, ToFloat ( COLS ), 1.f, ToFloat ( COLS ) } );
            cv::Mat matXt0 = cv::Mat ( vecXt ).reshape ( 1, 1 );
            matXt0 = cv::repeat ( matXt0, pstCmd->nBlockStepCount + 1, 1 );
            cv::transpose ( matXt0, matXt0T );
            matXt0T = matXt0T.reshape ( 1, 1 );

            vecYt = std::vector<DATA_TYPE> ( { ToFloat ( ROWS / 2 ), 1.f, 1.f, ToFloat ( ROWS ), ToFloat ( ROWS ) } );
            cv::Mat matYt0 = cv::Mat ( vecYt ).reshape ( 1, 1 );
            matYt0 = cv::repeat ( matYt0, pstCmd->nBlockStepCount + 1, 1 );
            cv::transpose ( matYt0, matYt0T );
            matYt0T = matYt0T.reshape ( 1, 1 );

            cv::transpose ( matHz, matHzT );
            matHzT = matHzT.reshape ( 1, ToInt32 ( matHzT.total() ) );

            std::vector<int> vecIndex( vecPointToFectch.size () * ( pstCmd->nBlockStepCount + 1 ) );
            std::iota ( vecIndex.begin(), vecIndex.end(), 0 );
            //Data fitting using multiple level data.
            //Z4 = k(1) * X * 4 + K(2) * Y * 4 + K(3) * 4
            //....
            //Z3 = k(1) * X * 3 + K(2) * Y * 3 + K(3) * 3
            //....
            //Z2 = k(1) * X * 2 + K(2) * Y * 2 + K(3) * 2
            //....
            //Z1 = k(1) * X * 1 + K(2) * Y * 1 + K(3) * 1
            //....
            for ( int i = 0; i < 2; ++ i ) {
                cv::Mat matXXt ( 1, ToInt32 ( vecIndex.size() ), CV_32FC1 );
                cv::Mat matXt( 1, ToInt32 ( vecIndex.size() ), CV_32FC1 );
                cv::Mat matYt( 1, ToInt32 ( vecIndex.size() ), CV_32FC1 );
                cv::Mat matYY ( ToInt32 ( vecIndex.size() ), 1, CV_32FC1 );
                for ( int j = 0; j < vecIndex.size(); ++ j ) {
                    matXXt.at<DATA_TYPE>(j) = matXpT.at<DATA_TYPE> ( vecIndex[j] );
                    matXt.at<DATA_TYPE>(j)  = matXt0T.at<DATA_TYPE>( vecIndex[j] );
                    matYt.at<DATA_TYPE>(j)  = matYt0T.at<DATA_TYPE>( vecIndex[j] );
                    matYY.at<DATA_TYPE>(j)  = matHzT.at<DATA_TYPE> ( vecIndex[j] );
                }
                matXX = cv::Mat ( matXXt.mul ( matXt ) );
                matXX.push_back ( cv::Mat ( matXXt.mul ( matYt ) ) );
                matXX.push_back ( matXXt );
                cv::transpose ( matXX, matXX );

                assert ( matXX.rows == matYY.rows );
                cv::solve ( matXX, matYY, matK, cv::DecompTypes::DECOMP_SVD );
#ifdef _DEBUG
                auto vecVecXX = CalcUtils::matToVector<DATA_TYPE> ( matXX );
                auto vecVecYY = CalcUtils::matToVector<DATA_TYPE> ( matYY );
                auto vecVecK  = CalcUtils::matToVector<DATA_TYPE> ( matK );
#endif
                if ( i == 0 ) {
                    cv::Mat matErr = cv::abs ( matYY - matXX * matK );
                    cv::Mat matErrSort;
                    cv::sortIdx ( matErr, matErrSort, cv::SortFlags::SORT_ASCENDING + cv::SortFlags::SORT_EVERY_COLUMN );
                    std::vector<int> vecTmpIndex;
                    int nKeepPoints = ToInt32 ( std::floor ( matErrSort.rows * CALIB_HEIGHT_STEP_USEFUL_PT ) );
                    vecTmpIndex.reserve ( nKeepPoints );
                    for (int j = 0; j < nKeepPoints; ++ j )
                        vecTmpIndex.push_back ( matErrSort.at<int>(j) );
                    std::swap ( vecTmpIndex, vecIndex );
                }
            }
            if ( NN == n ) {
                //Matlab code: Hz - (xp.*xt0*K(1) + xp.*yt0*K(2) + xp*K(3)
                cv::Mat matFitResultData = matXp.mul ( matXt0 ) * matK.at<DATA_TYPE> ( 0 ) + matXp.mul ( matYt0 ) * matK.at<DATA_TYPE> ( 1 ) + matXp * matK.at<DATA_TYPE> ( 2 );
                cv::Mat matAllDiff = matHz - matFitResultData;
                pstRpy->vecVecStepPhaseDiff = CalcUtils::matToVector<DATA_TYPE> ( matAllDiff );

                pstRpy->vecStepPhaseSlope.clear ();
                for (int i = 0; i < ToInt32 ( vecPointToFectch.size () ); ++ i ) {
                    cv::Mat matXX = cv::Mat ( matXp, cv::Rect ( i, 0, 1, matHz.rows ) );
                    cv::Mat matYY = cv::Mat ( matHz, cv::Rect ( i, 0, 1, matHz.rows ) );
                    cv::Mat matK;

                    assert ( matXX.rows == matYY.rows );
                    cv::solve ( matXX, matYY, matK, cv::DecompTypes::DECOMP_SVD );
                    pstRpy->vecStepPhaseSlope.push_back ( matK.at<DATA_TYPE> ( 0, 0 ) );
                }
            }
        }

        if (pstCmd->bReverseHeight)
            pstRpy->matPhaseToHeightK = - matK;
        else
            pstRpy->matPhaseToHeightK =   matK;

        cv::Mat matHeight;
        CalcUtils::meshgrid<DATA_TYPE> ( 1.f, 1.f, ToFloat(COLS), 1.f, 1.f, ToFloat(ROWS), matX, matY );
        cv::Mat matRatio = pstRpy->matPhaseToHeightK.at<DATA_TYPE> ( 0, 0 ) * matX + pstRpy->matPhaseToHeightK.at<DATA_TYPE> ( 1, 0 ) * matY + pstRpy->matPhaseToHeightK.at<DATA_TYPE> ( 2, 0 );
        cv::divide ( matPhase, matRatio, matTmpPhase );
    }
    pstRpy->matResultImg = _drawHeightGrid ( matTmpPhase, pstCmd->nResultImgGridRow, pstCmd->nResultImgGridCol, pstCmd->szMeasureWinSize );
    pstRpy->enStatus = VisionStatus::OK;
}

/*static*/ cv::Mat Unwrap::_drawHeightGrid(const cv::Mat &matHeight, int nGridRow, int nGridCol, const cv::Size &szMeasureWinSize) {
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
    int thickness = 2;
    for ( int j = 0; j < nGridRow; ++ j )
    for ( int i = 0; i < nGridCol; ++ i ) {
        cv::Rect rectROI ( i * nIntervalX + nIntervalX / 2 - szMeasureWinSize.width  / 2,
                           j * nIntervalY + nIntervalY / 2 - szMeasureWinSize.height / 2,
                           szMeasureWinSize.width, szMeasureWinSize.height );
        cv::rectangle ( matResultImg, rectROI, cv::Scalar ( 0, 255, 0 ), 3 );

        cv::Mat matROI ( matHeight, rectROI );
        cv::Mat matMask = (matROI == matROI);
        float fAverage = ToFloat ( cv::mean ( matROI, matMask )[0] );

        char strAverage[100];
        _snprintf ( strAverage, sizeof(strAverage), "%.4f", fAverage );
        int baseline = 0;
        cv::Size textSize = cv::getTextSize ( strAverage, fontFace, fontScale, thickness, &baseline );
        //The height use '+' because text origin start from left-bottom.
        cv::Point ptTextOrg ( rectROI.x + (rectROI.width - textSize.width) / 2, rectROI.y + (rectROI.height + textSize.height) / 2 );
        cv::putText ( matResultImg, strAverage, ptTextOrg, fontFace, fontScale, cv::Scalar ( 255, 0, 0 ), thickness );
    }

    int nGridLineSize = 3;
    cv::Scalar scalarCyan(255, 255, 0);
    for ( int i = 1; i < nGridCol; ++ i )
        cv::line ( matResultImg, cv::Point(i * nIntervalX, 0), cv::Point(i * nIntervalX, ROWS), scalarCyan, nGridLineSize );
    for ( int i = 1; i < nGridRow; ++ i )
        cv::line ( matResultImg, cv::Point(0, i * nIntervalY), cv::Point(COLS, i * nIntervalY), scalarCyan, nGridLineSize );

    return matResultImg;
}

/*static*/ void Unwrap::calc3DHeightDiff ( const PR_CALC_3D_HEIGHT_DIFF_CMD *const pstCmd, PR_CALC_3D_HEIGHT_DIFF_RPY *const pstRpy ) {
    std::vector<float> vecXt, vecYt, vecHeightTmp;
    auto ROWS = pstCmd->matHeight.rows;
    auto COLS = pstCmd->matHeight.cols;
    cv::Mat matX, matY;
    CalcUtils::meshgrid<DATA_TYPE> ( 1.f, 1.f, ToFloat(COLS), 1.f, 1.f, ToFloat(ROWS), matX, matY );

    cv::Mat matNan = CalcUtils::getNanMask ( pstCmd->matHeight );
    
    cv::Mat matHeight = pstCmd->matHeight;
    for (const auto &rect : pstCmd->vecRectBases ) {
        cv::Mat matMask = cv::Mat::zeros ( ROWS, COLS, CV_8UC1 );
        cv::Mat matMaskROI ( matMask, rect );
        matMaskROI.setTo ( 1 );        
        cv::Mat matNanROI ( matNan, rect );
        matMaskROI.setTo ( 0, matNanROI );         //Remove the points which is NAN.

        AOI::Vision::VectorOfPoint vecPtLocations;
        cv::findNonZero ( matMask, vecPtLocations );
        
        std::sort ( vecPtLocations.begin (), vecPtLocations.end (), [&matHeight]( const cv::Point &pt1, const cv::Point &pt2 ) {
            return matHeight.at<float> ( pt1 ) < matHeight.at<float> ( pt2 );
        } );
        AOI::Vision::VectorOfPoint vecTrimedLocations ( vecPtLocations.begin () + ToInt32 ( vecPtLocations.size () * pstCmd->fEffectHRatioStart ), vecPtLocations.begin () + ToInt32 ( vecPtLocations.size () * pstCmd->fEffectHRatioEnd ) );
        vecXt.reserve ( vecXt.size () + vecTrimedLocations.size () );
        vecYt.reserve ( vecYt.size () + vecTrimedLocations.size () );
        vecHeightTmp.reserve ( vecHeightTmp.size () + vecTrimedLocations.size () );
        for (const auto &point : vecTrimedLocations) {
            vecXt.push_back ( matX.at<float> ( point ) );
            vecYt.push_back ( matY.at<float> ( point ) );
            vecHeightTmp.push_back ( matHeight.at<float> ( point ) );
        }
    }
    cv::Mat matK;
    {
        cv::Mat matXX;
        matXX.push_back ( cv::Mat ( vecXt ).reshape ( 1, 1 ) );
        matXX.push_back ( cv::Mat ( vecYt ).reshape ( 1, 1 ) );
        matXX.push_back ( cv::Mat ( cv::Mat::ones ( 1, ToInt32 ( vecHeightTmp.size () ), CV_32FC1 ) ) );
        cv::transpose ( matXX, matXX );
        cv::Mat matYY ( vecHeightTmp );
        cv::solve ( matXX, matYY, matK, cv::DecompTypes::DECOMP_QR );
    }

    float k1 = matK.at<float> ( 0 );
    float k2 = matK.at<float> ( 1 );
    float k3 = matK.at<float> ( 2 );
    
    //Find the good points in ROI.
    cv::Mat matMask = cv::Mat::zeros ( ROWS, COLS, CV_8UC1 );
    cv::Mat matMaskROI ( matMask, pstCmd->rectROI );
    matMaskROI.setTo ( 1 );
    cv::Mat matNanROI ( matNan, pstCmd->rectROI );
    matMaskROI.setTo ( 0, matNanROI );         //Remove the points which is NAN.

    AOI::Vision::VectorOfPoint vecPtLocations;
    cv::findNonZero ( matMask, vecPtLocations );

    std::sort ( vecPtLocations.begin (), vecPtLocations.end (), [&matHeight]( const cv::Point &pt1, const cv::Point &pt2 ) {
        return matHeight.at<float> ( pt1 ) < matHeight.at<float> ( pt2 );
    } );
    AOI::Vision::VectorOfPoint vecTrimedLocations ( vecPtLocations.begin () + ToInt32 ( vecPtLocations.size () * pstCmd->fEffectHRatioStart ), vecPtLocations.begin () + ToInt32 ( vecPtLocations.size () * pstCmd->fEffectHRatioEnd ) );

    vecXt.clear();
    vecYt.clear();
    vecHeightTmp.clear();
    for (const auto &point : vecTrimedLocations) {
        vecXt.push_back ( matX.at<float> ( point ) );
        vecYt.push_back ( matY.at<float> ( point ) );
        vecHeightTmp.push_back ( matHeight.at<float> ( point ) );
    }

    cv::Mat matXX;
    matXX.push_back ( cv::Mat ( vecXt ).reshape ( 1, 1 ) );
    matXX.push_back ( cv::Mat ( vecYt ).reshape ( 1, 1 ) );
    matXX.push_back ( cv::Mat ( cv::Mat::ones ( 1, ToInt32 ( vecHeightTmp.size () ), CV_32FC1 ) ) );
    cv::transpose ( matXX, matXX );
    cv::Mat matZZ ( vecHeightTmp );

    matZZ = matZZ - matXX * matK;
    pstRpy->fHeightDiff = ToFloat ( cv::mean ( matZZ )[0] );
    pstRpy->enStatus = VisionStatus::OK;
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
    pstRpy->enStatus = VisionStatus::OK;
}

/*static*/ void Unwrap::calcPD(const PR_CALC_PD_CMD *const pstCmd, PR_CALC_PD_RPY *const pstRpy) {
    std::vector<cv::Mat> vecConvertedImgs;
    //Discard the first 4 thick texture light.
    for ( size_t index = 4; index < 12; ++ index ) {
        cv::Mat matConvert = pstCmd->vecInputImgs[index];
        if ( pstCmd->vecInputImgs[index].channels() > 1 )
            cv::cvtColor ( pstCmd->vecInputImgs[index], matConvert, CV_BGR2GRAY );
        matConvert.convertTo ( matConvert, CV_32FC1 );
        vecConvertedImgs.push_back ( matConvert );
    }

    //Filtering can reduce residues at the expense of a loss of spatial resolution.
    for( auto &mat : vecConvertedImgs )
        cv::GaussianBlur ( mat, mat, cv::Size ( pstCmd->nGaussianFilterSize, pstCmd->nGaussianFilterSize ), pstCmd->fGaussianFilterSigma, pstCmd->fGaussianFilterSigma, cv::BorderTypes::BORDER_REPLICATE );

    //if ( pstCmd->bReverseSeq ) {
    //    std::swap ( vecConvertedImgs[1], vecConvertedImgs[3] );
    //    std::swap ( vecConvertedImgs[5], vecConvertedImgs[7] );
    //}
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

#ifdef _DEBUG
    auto vecVecAlpha = CalcUtils::matToVector<DATA_TYPE> ( matAlpha );
    auto vecVecBeta  = CalcUtils::matToVector<DATA_TYPE> ( matBeta );
#endif

    const auto ROWS = matAlpha.rows;
    const auto COLS = matAlpha.cols;
    const auto MIDDLE_GRAY_SCALE = 127;

    matAlpha = _phaseUnwrapSurface ( matAlpha );
    matBeta  = _phaseUnwrapSurface ( matBeta );

    cv::Size szDlpPatternSize ( pstCmd->szDlpPatternSize );
    szDlpPatternSize.width *= 2;    //The width is double sized when do projection.
    cv::Mat matX, matY, matU, matV;
    CalcUtils::meshgrid<DATA_TYPE>(1.f, 1.f, ToFloat ( szDlpPatternSize.width ), 1.f, 1.f, ToFloat ( szDlpPatternSize.height ), matX, matY );
    cv::Mat matDlpPattern0 = ( pstCmd->fMagnitudeOfDLP / 2 ) * CalcUtils::cos<DATA_TYPE>( ( matY + matX ) * ( 2 * CV_PI ) / pstCmd->fDlpPixelCycle ) + MIDDLE_GRAY_SCALE; 
    matDlpPattern0.convertTo ( pstRpy->matCaptureRegionImg, CV_8U );
    cv::cvtColor ( pstRpy->matCaptureRegionImg, pstRpy->matCaptureRegionImg, CV_GRAY2BGR );
    cv::Mat matXX1 = ( matAlpha - matBeta ) / 4 / CV_PI * pstCmd->fDlpPixelCycle;   //Convert the unwrapped phase to DLP pattern phase.
    cv::Mat matYY1 = ( matAlpha + matBeta ) / 4 / CV_PI * pstCmd->fDlpPixelCycle;
#ifdef _DEBUG
    auto vecVecXX1 = CalcUtils::matToVector<DATA_TYPE> ( matXX1 );
    auto vecVecYY1 = CalcUtils::matToVector<DATA_TYPE> ( matYY1 );
#endif
    CalcUtils::meshgrid<DATA_TYPE>(0.f, 1.f / (COLS - 1), 1.f, 0.f, 1.f / (ROWS - 1), 1.f, matU, matV );
    matU = matU.reshape ( 1, 1 );
    matV = matV.reshape ( 1, 1 );

    cv::Mat matP1 = calcBezierCoeff ( matU );
    cv::Mat matP2 = calcBezierCoeff ( matV );
    cv::Mat matXX;
    for ( int i = 0; i < BEZIER_RANK; ++ i )
    for ( int j = 0; j < BEZIER_RANK; ++ j ) {
        matXX.push_back ( cv::Mat (  matP1.row(i).mul ( matP2.row(j) ) ) );
    }
    cv::transpose ( matXX, matXX );

    matXX1 = matXX1.reshape ( 1, ToInt32 ( matXX1.total() ) );
    matYY1 = matYY1.reshape ( 1, ToInt32 ( matYY1.total() ) );
    cv::Mat matPPx, matPPy;
    cv::solve ( matXX, matXX1, matPPx, cv::DecompTypes::DECOMP_SVD );
    cv::solve ( matXX, matYY1, matPPy, cv::DecompTypes::DECOMP_SVD );

    float fPPxMean = ToFloat ( cv::mean ( matPPx )[0] );
    float fPPyMean = ToFloat ( cv::mean ( matPPy )[0] );
    cv::Mat matPPx1 = matPPx - floor ( ( fPPxMean - szDlpPatternSize.width  / 2 ) / pstCmd->fDlpPixelCycle  ) * pstCmd->fDlpPixelCycle;
    cv::Mat matPPy1 = matPPy - floor ( ( fPPyMean - szDlpPatternSize.height / 2 ) / pstCmd->fDlpPixelCycle  ) * pstCmd->fDlpPixelCycle;
#ifdef _DEBUG
    auto vecVecPPx1 = CalcUtils::matToVector<DATA_TYPE> ( matPPx1 );
    auto vecVecPPy1 = CalcUtils::matToVector<DATA_TYPE> ( matPPy1 );
#endif

    const int SMALL_WIN_SIZE = 10;
    for ( int i = 0; i < matPPx1.rows; ++ i ) {
        cv::Point ptCtr ( ToInt32 ( matPPx1.at<DATA_TYPE>(i) ), ToInt32 ( matPPy1.at<DATA_TYPE>(i) ) );
        cv::Rect rect ( ptCtr.x - SMALL_WIN_SIZE / 2, ptCtr.y - SMALL_WIN_SIZE / 2, SMALL_WIN_SIZE, SMALL_WIN_SIZE );
        cv::rectangle ( pstRpy->matCaptureRegionImg, rect, cv::Scalar(0, 255, 0 ), CV_FILLED, CV_AA );
    }

    cv::Mat matXXt1 = matXX * matPPx1;
    cv::Mat matYYt1 = matXX * matPPy1;

    matXXt1 = matXXt1.reshape ( 1, ROWS );
    matYYt1 = matYYt1.reshape ( 1, ROWS );
#ifdef _DEBUG
    auto vecVecXXt1 = CalcUtils::matToVector<DATA_TYPE> ( matXXt1 );
    auto vecVecYYt1 = CalcUtils::matToVector<DATA_TYPE> ( matYYt1 );
#endif
    const int nInterval = 100;
    for ( int row = 0; row < ROWS; row += nInterval )
    for ( int col = 0; col < COLS; col += nInterval ) {
        cv::Point ptCtr ( ToInt32 ( matXXt1.at<DATA_TYPE>(row, col) ), ToInt32 ( matYYt1.at<DATA_TYPE>(row, col) ) );
        cv::circle ( pstRpy->matCaptureRegionImg, ptCtr, 5, cv::Scalar(255, 0, 0 ), CV_FILLED, CV_AA );
    }

    cv::Mat matError;
    {
        cv::Mat matXt(matXXt1, cv::Rect(0, 0, 1, ROWS ) );
        cv::Mat matYt(matYYt1, cv::Rect(0, 0, 1, ROWS ) );
        matError = ( matYt - matYt.at<DATA_TYPE>(0) - ( matYt.at<DATA_TYPE>(ROWS - 1) - matYt.at<DATA_TYPE>(0))/(matXt.at<DATA_TYPE>(ROWS - 1) - matXt.at<DATA_TYPE>(0))*( matXt - matXt.at<DATA_TYPE>(0)))*( ROWS - 1 )/ ( matXt.at<DATA_TYPE>(ROWS - 1) - matXt.at<DATA_TYPE>(0) );
        matError = matError.reshape(1, 1);
        pstRpy->vecDistortionLeft = CalcUtils::matToVector<DATA_TYPE>(matError)[0];
    }
    {
        cv::Mat matXt(matXXt1, cv::Rect(COLS - 1, 0, 1, ROWS ) );
        cv::Mat matYt(matYYt1, cv::Rect(COLS - 1, 0, 1, ROWS ) );
        matError = ( matYt - matYt.at<DATA_TYPE>(0) - ( matYt.at<DATA_TYPE>(ROWS - 1) - matYt.at<DATA_TYPE>(0))/(matXt.at<DATA_TYPE>(ROWS - 1) - matXt.at<DATA_TYPE>(0))*( matXt - matXt.at<DATA_TYPE>(1)))*( ROWS - 1 )/ ( matXt.at<DATA_TYPE>(ROWS - 1) - matXt.at<DATA_TYPE>(0) );
        matError = matError.reshape(1, 1);
        pstRpy->vecDistortionRight = CalcUtils::matToVector<DATA_TYPE>(matError)[0];
    }
    {
        cv::Mat matYt(matXXt1, cv::Rect(0, 0, COLS, 1 ) );
        cv::Mat matXt(matYYt1, cv::Rect(0, 0, COLS, 1 ) );
        matError = ( matYt - matYt.at<DATA_TYPE>(0) - ( matYt.at<DATA_TYPE>(COLS - 1) - matYt.at<DATA_TYPE>(0))/(matXt.at<DATA_TYPE>(COLS - 1) - matXt.at<DATA_TYPE>(0))*( matXt - matXt.at<DATA_TYPE>(0)))*( COLS - 1 )/ ( matXt.at<DATA_TYPE>(COLS - 1) - matXt.at<DATA_TYPE>(0) );
        pstRpy->vecDistortionTop = CalcUtils::matToVector<DATA_TYPE>(matError)[0];
    }
    {
        cv::Mat matYt(matXXt1, cv::Rect(0, ROWS - 1, COLS, 1 ) );
        cv::Mat matXt(matYYt1, cv::Rect(0, ROWS - 1, COLS, 1 ) );
        matError = ( matYt - matYt.at<DATA_TYPE>(0) - ( matYt.at<DATA_TYPE>(COLS - 1) - matYt.at<DATA_TYPE>(0))/(matXt.at<DATA_TYPE>(COLS - 1) - matXt.at<DATA_TYPE>(0))*( matXt - matXt.at<DATA_TYPE>(0)))*( COLS - 1 )/ ( matXt.at<DATA_TYPE>(COLS - 1) - matXt.at<DATA_TYPE>(0) );
        pstRpy->vecDistortionBottom = CalcUtils::matToVector<DATA_TYPE>(matError)[0];
    }
    pstRpy->enStatus = VisionStatus::OK;
}

}
}