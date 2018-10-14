#include "Unwrap.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "CalcUtils.h"
#include "VisionAlgorithm.h"
#include "TimeLog.h"
#include "Log.h"
#include "Config.h"
#include <iostream>
#include <numeric>
#include <limits>

#define MARK_FUNCTION_START_TIME    CStopWatch      stopWatch; __int64 functionStart = stopWatch.AbsNow()
#define MARK_FUNCTION_END_TIME      TimeLog::GetInstance()->addTimeLog( __FUNCTION__, stopWatch.AbsNow() - functionStart)

namespace AOI
{
namespace Vision
{

const DATA_TYPE PHASE_TURN_THRESHOLD_1 = 0.75f;
const DATA_TYPE PHASE_TURN_THRESHOLD_2 = 0.15f;

Unwrap::Unwrap () {
}

Unwrap::~Unwrap() {
}

/*static*/ const float Unwrap::GAUSSIAN_FILTER_SIGMA = ToFloat(pow(20, 0.5));
/*static*/ const float Unwrap::ONE_HALF_CYCLE = ToFloat(1.f);
/*static*/ const float Unwrap::ONE_CYCLE = ONE_HALF_CYCLE * 2;
/*static*/ const float Unwrap::CALIB_HEIGHT_STEP_USEFUL_PT = 0.8f;
/*static*/ const float Unwrap::LOW_BASE_PHASE = - ToFloat(CV_PI);
/*static*/ VectorOfVectorOfFloat Unwrap::vecVecAtan2Table(PR_MAX_GRAY_LEVEL * 2 + 1, VectorOfFloat(PR_MAX_GRAY_LEVEL * 2 + 1, 0));

/*static*/ void Unwrap::initAtan2Table() {
    for (int x = -PR_MAX_GRAY_LEVEL; x <= PR_MAX_GRAY_LEVEL; ++ x)
    for (int y = -PR_MAX_GRAY_LEVEL; y <= PR_MAX_GRAY_LEVEL; ++ y) {
        //The opencv cv::phase function return result is 0~2*PI, but matlab is -PI~PI, so here need to do a conversion.
        auto radian = cv::fastAtan2(ToFloat(y), ToFloat(x)) * CV_PI / 180.;
        if (radian > CV_PI)
            radian -= CV_PI * 2.;
        radian /= CV_PI;
        vecVecAtan2Table[x + PR_MAX_GRAY_LEVEL][y + PR_MAX_GRAY_LEVEL] = ToFloat(radian);
    }
}

/*static*/ cv::Mat Unwrap::_getResidualPoint(const cv::Mat &matPhaseDx, const cv::Mat &matPhaseDy, cv::Mat &matPosPole, cv::Mat &matNegPole) {
    MARK_FUNCTION_START_TIME;

    const float PI_WITH_MARGIN = ToFloat(0.9999 * ONE_HALF_CYCLE);

    cv::Mat matDB1 = cv::Mat::zeros(matPhaseDx.size(), CV_8SC1);
    cv::Mat matDB2 = cv::Mat::zeros(matPhaseDy.size(), CV_8SC1);

    //This loop same as matlab code.
    //idx = find( abs(dxPhase)>0.9999*pi );
    //db1(idx) = - sign(dxPhase(idx)); 
    //for ( int i = 0; i < matPhaseDx.total(); ++ i ) {
    //    auto value = matPhaseDx.at<DATA_TYPE>( i );
    //    if ( value > PI_WITH_MARGIN )
    //        matDB1.at<char> ( i ) = -1;
    //    else if ( value < - PI_WITH_MARGIN )
    //        matDB1.at<char>( i ) = 1;
    //}
    {
        cv::Mat matPosJump, matNegJump;
        cv::compare(matPhaseDx, cv::Scalar::all(PI_WITH_MARGIN), matPosJump, cv::CmpTypes::CMP_GT);
        cv::compare(matPhaseDx, cv::Scalar::all(-PI_WITH_MARGIN), matNegJump, cv::CmpTypes::CMP_LT);
        matDB1.setTo(-1, matPosJump);
        matDB1.setTo(1, matNegJump);
    }
    //This loop same as matlab code.
    //idx = find( abs(dyPhase)>0.9999*pi );
    //db2(idx) = - sign(dyPhase(idx));
    //for ( int i = 0; i < matPhaseDy.total(); ++ i ) {
    //    auto value = matPhaseDy.at<DATA_TYPE>( i );
    //    if ( value > PI_WITH_MARGIN )
    //        matDB2.at<char> ( i ) = -1;
    //    else if ( value < - PI_WITH_MARGIN )
    //        matDB2.at<char>( i ) = 1;
    //}
    {
        cv::Mat matPosJump, matNegJump;
        cv::compare(matPhaseDy, cv::Scalar::all(PI_WITH_MARGIN), matPosJump, cv::CmpTypes::CMP_GT);
        cv::compare(matPhaseDy, cv::Scalar::all(-PI_WITH_MARGIN), matNegJump, cv::CmpTypes::CMP_LT);
        matDB2.setTo(-1, matPosJump);
        matDB2.setTo(1, matNegJump);
    }

    TimeLog::GetInstance()->addTimeLog("Prepare take: ", stopWatch.Span());

    cv::Mat matD1 = cv::Mat(matDB1, cv::Rect(0, 0, matDB1.cols, matDB1.rows - 1));
    cv::Mat matD2 = cv::Mat(matDB2, cv::Rect(1, 0, matDB2.cols - 1, matDB2.rows));
    cv::Mat matD3 = -cv::Mat(matDB1, cv::Rect(0, 1, matDB1.cols, matDB1.rows - 1));
    cv::Mat matD4 = -cv::Mat(matDB2, cv::Rect(0, 0, matDB2.cols - 1, matDB2.rows));

    cv::Mat matDD = matD1 + matD2 + matD3 + matD4;

    TimeLog::GetInstance()->addTimeLog("Sum take: ", stopWatch.Span());

    cv::Mat matPosIndex, matNegIndex;
    cv::compare(matDD, cv::Scalar::all(0.9999), matPosPole, cv::CmpTypes::CMP_GT);
    cv::compare(matDD, cv::Scalar::all(-0.9999), matNegPole, cv::CmpTypes::CMP_LT);

    MARK_FUNCTION_END_TIME;
    return cv::Mat();
}

template<typename T>
inline std::vector<size_t> sort_indexes(const std::vector<T> &v) {
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return idx;
}

/*static*/ VectorOfPoint Unwrap::_selectLonePole(ListOfPoint &listPoint, int width, int height, size_t nLonePoneCount) {
    VectorOfPoint vecResult;
    if (nLonePoneCount > listPoint.size())
        return vecResult;

    std::vector<int> vecDistance;
    for (auto iter = listPoint.cbegin(); iter != listPoint.cend(); ++ iter) {
        cv::Point point = *iter;
        std::vector<int> vec4SideDistance;
        vec4SideDistance.push_back(point.x);
        vec4SideDistance.push_back(width - point.x);
        vec4SideDistance.push_back(point.y);
        vec4SideDistance.push_back(height - point.y);

        auto minElement = std::min_element(vec4SideDistance.begin(), vec4SideDistance.end());
        int minDistance = *minElement;
        vecDistance.push_back(minDistance);
    }

    std::vector<size_t> vecIndex = sort_indexes<int>(vecDistance);
    std::vector<ListOfPoint::const_iterator> vecIter;

    for (size_t j = 0; j < nLonePoneCount; ++ j) {
        size_t index = 0;
        for (auto iter = listPoint.cbegin(); iter != listPoint.cend(); ++ iter) {
            if (index == vecIndex[j]) {
                vecIter.push_back(iter);
                vecResult.push_back(*iter);
                ++ index;
            }
            ++ index;
        }
    }

    for (const auto &iter : vecIter)
        listPoint.erase(iter);
    return vecResult;
}

/*static*/ void Unwrap::_selectPolePair(const VectorOfPoint &vecPosPoint,
    const VectorOfPoint &vecNegPoint,
    int rows, int cols,
    VectorOfPoint &vecResult1,
    VectorOfPoint &vecResult2,
    VectorOfPoint &vecResult3) {
    const int MAX_RANGE = 20;
    ListOfPoint listPosPoint, listNegPoint;
    for (const auto &point : vecPosPoint) listPosPoint.push_back(point);
    for (const auto &point : vecNegPoint) listNegPoint.push_back(point);

    for (int range = 1; range <= MAX_RANGE; ++ range) {
        std::vector<ListOfPoint::const_iterator> vecIter1;
        for (auto iter1 = listPosPoint.cbegin(); iter1 != listPosPoint.cend(); ++ iter1) {
            std::vector<ListOfPoint::const_iterator> vecIter2;
            for (auto iter2 = listNegPoint.cbegin(); iter2 != listNegPoint.cend(); ++ iter2) {
                cv::Point point1 = *iter1, point2 = *iter2;
                if (abs(point1.x - point2.x) <= range && abs(point1.y - point2.y) <= range) {
                    vecIter1.push_back(iter1);
                    vecIter2.push_back(iter2);

                    vecResult1.push_back(point1);
                    vecResult2.push_back(point2);

                    listNegPoint.erase(iter2);
                    break;
                }
            }
        }
        if (! vecIter1.empty()) {
            for (const auto &iter : vecIter1)
                listPosPoint.erase(iter);
        }

        if (3 == range) {
            if (listPosPoint.size() > listNegPoint.size()) {
                vecResult3 = _selectLonePole(listPosPoint, cols, rows, listPosPoint.size() - listNegPoint.size());
            }
            else
            if (listPosPoint.size() < listNegPoint.size()) {
                vecResult3 = _selectLonePole(listNegPoint, cols, rows, listNegPoint.size() - listPosPoint.size());
            }
        }
    }

    for (const auto &point1 : listPosPoint) {
        double dMinDistance = 100000000000.;
        ListOfPoint::const_iterator iterMatch;
        for (auto iter2 = listNegPoint.cbegin(); iter2 != listNegPoint.cend(); ++ iter2) {
            cv::Point point2 = *iter2;
            double dDistance = sqrt((point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y));
            if (dDistance < dMinDistance) {
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

/*static*/ VectorOfPoint Unwrap::_getIntegralTree(const VectorOfPoint &vecResult1, const VectorOfPoint &vecResult2, const VectorOfPoint &vecResult3, int rows, int cols) {
    MARK_FUNCTION_START_TIME;

    VectorOfPoint vecResult;
    for (size_t i = 0; i < vecResult1.size(); ++ i) {
        const auto point1 = vecResult1[i];
        const auto point2 = vecResult2[i];

        const auto minX = std::min(point1.x, point2.x);
        const auto maxX = std::max(point1.x, point2.x);
        const auto minY = std::min(point1.y, point2.y);
        const auto maxY = std::max(point1.y, point2.y);

        std::vector<int> vecXX{point1.x, point2.x};
        std::vector<int> vecYY{point1.y, point2.y};
        auto index = std::min_element(vecYY.begin(), vecYY.end()) - vecYY.begin();

        std::vector<int> vecTrkYOne(maxY - minY + 1);
        std::iota(vecTrkYOne.begin(), vecTrkYOne.end(), minY);
        std::vector<int> vecTrkXOne(vecTrkYOne.size(), vecXX[index]);

        std::vector<int> vecTrkXTwo(maxX - minX + 1);
        std::iota(vecTrkXTwo.begin(), vecTrkXTwo.end(), minX);
        if (maxX == vecXX[index])
            std::reverse(vecTrkXTwo.begin(), vecTrkXTwo.end());
        std::vector<int> vecTrkYTwo(vecTrkXTwo.size(), maxY);

        for (size_t k = 0; k < vecTrkXOne.size(); ++ k)
            vecResult.emplace_back(vecTrkXOne[k], vecTrkYOne[k]);

        for (size_t k = 0; k < vecTrkXTwo.size(); ++ k)
            vecResult.emplace_back(vecTrkXTwo[k], vecTrkYTwo[k]);
    }

    if (! vecResult3.empty()) {
        for (const auto &point : vecResult3) {
            std::vector<int> vecDist({point.x, cols - point.x, point.y, rows - point.y});
            auto iterMin = std::min_element(vecDist.begin(), vecDist.end());
            int nIndex = ToInt32(std::distance(vecDist.begin(), iterMin));
            if (0 == nIndex) {
                for (int i = 0; i <= point.x; ++ i)
                    vecResult.emplace_back(i, point.y);
            }
            else if (1 == nIndex) {
                for (int i = point.x; i < cols; ++ i)
                    vecResult.emplace_back(i, point.y);
            }
            else if (2 == nIndex) {
                for (int i = 0; i <= point.y; ++ i)
                    vecResult.emplace_back(point.x, i);
            }
            else if (3 == nIndex) {
                for (int i = point.y; i < rows; ++ i)
                    vecResult.emplace_back(point.x, i);
            }
        }
    }

    MARK_FUNCTION_END_TIME;
    return vecResult;
}

/*static*/ void Unwrap::calib3DBase(const PR_CALIB_3D_BASE_CMD *const pstCmd, PR_CALIB_3D_BASE_RPY *const pstRpy) {
    VectorOfMat vecConvertedImgs;
    for (auto &mat : pstCmd->vecInputImgs) {
        cv::Mat matConvert = mat;
        if (mat.channels() > 1)
            cv::cvtColor(mat, matConvert, CV_BGR2GRAY);
        matConvert.convertTo(matConvert, CV_32FC1);
        vecConvertedImgs.push_back(matConvert);
    }

    _calcReverseAndProjDir(VectorOfMat(vecConvertedImgs.begin(), vecConvertedImgs.begin() + PR_GROUP_TEXTURE_IMG_COUNT), pstCmd->bEnableGaussianFilter, pstRpy);

    if (pstCmd->bEnableGaussianFilter) {
        //Filtering can reduce residues at the expense of a loss of spatial resolution.
        for (int i = 0; i < PR_GROUP_TEXTURE_IMG_COUNT; ++ i)
            cv::GaussianBlur(vecConvertedImgs[i], vecConvertedImgs[i], cv::Size(GUASSIAN_FILTER_SIZE, GUASSIAN_FILTER_SIZE), GAUSSIAN_FILTER_SIGMA, GAUSSIAN_FILTER_SIGMA, cv::BorderTypes::BORDER_REPLICATE);
    }

    for (int i = 2 * PR_GROUP_TEXTURE_IMG_COUNT; i < 3 * PR_GROUP_TEXTURE_IMG_COUNT; ++ i) {
        cv::Mat matBlur;
        cv::GaussianBlur(vecConvertedImgs[i], matBlur, cv::Size(GUASSIAN_FILTER_SIZE, GUASSIAN_FILTER_SIZE), GAUSSIAN_FILTER_SIGMA, GAUSSIAN_FILTER_SIGMA, cv::BorderTypes::BORDER_REPLICATE);
        vecConvertedImgs.push_back(matBlur);
    }
    assert(vecConvertedImgs.size() == 4 * PR_GROUP_TEXTURE_IMG_COUNT);

    if (pstRpy->bReverseSeq) {
        std::swap(vecConvertedImgs[1], vecConvertedImgs[3]);
        std::swap(vecConvertedImgs[5], vecConvertedImgs[7]);
        std::swap(vecConvertedImgs[9], vecConvertedImgs[11]);
        std::swap(vecConvertedImgs[13], vecConvertedImgs[15]);
    }
    cv::Mat mat00 = vecConvertedImgs[2] - vecConvertedImgs[0];
    cv::Mat mat01 = vecConvertedImgs[3] - vecConvertedImgs[1];

    cv::Mat mat10 = vecConvertedImgs[6] - vecConvertedImgs[4];
    cv::Mat mat11 = vecConvertedImgs[7] - vecConvertedImgs[5];

    cv::Mat mat20 = vecConvertedImgs[10] - vecConvertedImgs[8];
    cv::Mat mat21 = vecConvertedImgs[11] - vecConvertedImgs[9];

    cv::Mat mat30 = vecConvertedImgs[14] - vecConvertedImgs[12];
    cv::Mat mat31 = vecConvertedImgs[15] - vecConvertedImgs[13];

    cv::Mat matAlpha, matBeta, matGamma, matGammaRef;
    cv::phase(-mat00, mat01, matAlpha);
    cv::phase(-mat10, mat11, matBeta);
    cv::phase(-mat20, mat21, matGamma);
    cv::phase(-mat30, mat31, matGammaRef);

    //The opencv cv::phase function return result is 0~2*PI, but matlab is -PI~PI, so here need to do a conversion.
    cv::Mat matOverPi;
    cv::compare(matAlpha, cv::Scalar::all(CV_PI), matOverPi, cv::CmpTypes::CMP_GT);
    cv::subtract(matAlpha, cv::Scalar::all(2.f * CV_PI), matAlpha, matOverPi);
    cv::compare(matBeta, cv::Scalar::all(CV_PI), matOverPi, cv::CmpTypes::CMP_GT);
    cv::subtract(matBeta, cv::Scalar::all(2.f * CV_PI), matBeta, matOverPi);
    cv::compare(matGamma, cv::Scalar::all(CV_PI), matOverPi, cv::CmpTypes::CMP_GT);
    cv::subtract(matGamma, cv::Scalar::all(2.f * CV_PI), matGamma, matOverPi);
    cv::compare(matGammaRef, cv::Scalar::all(CV_PI), matOverPi, cv::CmpTypes::CMP_GT);
    cv::subtract(matGammaRef, cv::Scalar::all(2.f * CV_PI), matGammaRef, matOverPi);

    matAlpha = matAlpha / CV_PI;
    matBeta = matBeta / CV_PI;
    matGamma = matGamma / CV_PI;
    matGammaRef = matGammaRef / CV_PI;
    matAlpha = _phaseUnwrapSurface(matAlpha);
    matBeta = _phaseUnwrapSurface(matBeta);
    matGammaRef = _phaseUnwrapSurface(matGammaRef);
    matGamma = _phaseUnwrapSurfaceByRefer(matGamma, matGammaRef, cv::Mat());

    const int TOTAL = ToInt32(matAlpha.total());

    if (fabs(pstCmd->fRemoveHarmonicWaveK) > 1e-5)
        matAlpha = matAlpha + CalcUtils::sin<DATA_TYPE>(4.f * matAlpha) * pstCmd->fRemoveHarmonicWaveK;

    cv::Mat matAlphaReshape = matAlpha.reshape(1, 1);
    cv::Mat matXX = matAlphaReshape.clone();
    cv::Mat matOne = cv::Mat::ones(1, matAlpha.rows * matAlpha.cols, CV_32FC1);
    matXX.push_back(matOne);
    cv::transpose(matXX, matXX);

    cv::Mat matYY = matBeta.reshape(1, TOTAL);
    cv::solve(matXX, matYY, pstRpy->matThickToThinK, cv::DecompTypes::DECOMP_SVD);
#ifdef _DEBUG
    auto vecVecAlpha = CalcUtils::matToVector<DATA_TYPE>(matAlpha);
    auto vecVecBeta  = CalcUtils::matToVector<DATA_TYPE>(matBeta);
    auto vecVecMatK  = CalcUtils::matToVector<DATA_TYPE>(pstRpy->matThickToThinK);
#endif

    matYY = matGamma.reshape(1, TOTAL);
    cv::solve(matXX, matYY, pstRpy->matThickToThinnestK, cv::DecompTypes::DECOMP_SVD);

#ifdef _DEBUG
    vecVecMatK  = CalcUtils::matToVector<DATA_TYPE>(pstRpy->matThickToThinnestK);
#endif

    int ROWS = pstCmd->vecInputImgs[0].rows;
    int COLS = pstCmd->vecInputImgs[0].cols;
    cv::Mat matX, matY;
    CalcUtils::meshgrid<float>(1.f, 1.f, ToFloat(COLS), 1.f, 1.f, ToFloat(ROWS), matX, matY);
    cv::Mat matBaseSurfaceParam = _calculatePPz(matX, matY, matGamma);
    cv::Mat matFittedGamma = calculateBaseSurface(matBeta.rows, matBeta.cols, matBaseSurfaceParam);
    cv::Mat matFittedAlpha = (matFittedGamma - pstRpy->matThickToThinnestK.at<DATA_TYPE>(1)) / pstRpy->matThickToThinnestK.at<DATA_TYPE>(0);
    cv::Mat matFittedBeta = matFittedAlpha * pstRpy->matThickToThinK.at<DATA_TYPE>(0) + pstRpy->matThickToThinK.at<DATA_TYPE>(1); //pBeta = K1(1) * pAlpha + K1(2)
    _phaseWrap(matFittedAlpha);
    pstRpy->matBaseWrappedAlpha = matFittedAlpha;
    _phaseWrap(matFittedBeta);
    pstRpy->matBaseWrappedBeta = matFittedBeta;
    _phaseWrap(matFittedGamma);
    pstRpy->matBaseWrappedGamma = matFittedGamma;
    pstRpy->enStatus = VisionStatus::OK;
}

/*static*/ inline void Unwrap::_setBySign(cv::Mat &matInOut, DATA_TYPE value) {
    cv::Mat matPosIndex = cv::Mat::zeros(matInOut.size(), CV_8UC1);
    cv::Mat matNegIndex = cv::Mat::zeros(matInOut.size(), CV_8UC1);
    cv::compare(matInOut, cv::Scalar::all(0.0001), matPosIndex, cv::CmpTypes::CMP_GT);
    cv::compare(matInOut, cv::Scalar::all(-0.0001), matNegIndex, cv::CmpTypes::CMP_LT);

    matInOut.setTo(cv::Scalar::all(-value), matPosIndex);
    matInOut.setTo(cv::Scalar::all(value), matNegIndex);
}

/*static*/ cv::Mat Unwrap::_phaseUnwrapSurface(const cv::Mat &matPhase) {
    MARK_FUNCTION_START_TIME;

    const float OneCycle = 2.f * ONE_HALF_CYCLE;

    cv::Mat matPhaseT;
    cv::transpose(matPhase, matPhaseT);
    cv::Mat matTmpPhase = cv::Mat(matPhaseT, cv::Rect(0, 0, matPhaseT.cols, 1));
    cv::Mat matDp = CalcUtils::diff(matTmpPhase, 1, 2);

#ifdef _DEBUG
    auto vecVecDp = CalcUtils::matToVector<DATA_TYPE>( matDp );
#endif

    cv::Mat matAbsUnderPI;
    cv::compare(cv::abs(matDp), cv::Scalar::all(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT);
    matDp.setTo(0, matAbsUnderPI);

    _setBySign(matDp, OneCycle);       //Correspoinding to matlab: dp(idx) = -sign(dp(idx))*Th;
    matDp = cv::repeat(matDp, matPhase.cols, 1);

    cv::Mat matTmp(matPhaseT, cv::Rect(1, 0, matPhaseT.cols - 1, matPhaseT.rows));
    cv::Mat matCumSum = CalcUtils::cumsum<DATA_TYPE>(matDp, 2);
    matTmp += matCumSum;

    cv::Mat matPhaseResult;
    cv::transpose(matPhaseT, matPhaseResult);
#ifdef _DEBUG 
    auto vecCumsum = CalcUtils::matToVector<DATA_TYPE>( matCumSum );
    auto vecTmp = CalcUtils::matToVector<DATA_TYPE>( matTmp );
    auto vecPhaseT = CalcUtils::matToVector<DATA_TYPE>( matPhaseT );
    auto vecPhaseResult = CalcUtils::matToVector<DATA_TYPE>( matPhaseResult );
#endif

    matDp = CalcUtils::diff(matPhaseResult, 1, 2);
    cv::compare(cv::abs(matDp), cv::Scalar::all(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT);
    matDp.setTo(0, matAbsUnderPI);
    _setBySign(matDp, OneCycle);

    cv::Mat matTmpl(matPhaseResult, cv::Rect(1, 0, matPhaseResult.cols - 1, matPhaseResult.rows));
    matTmpl += CalcUtils::cumsum<DATA_TYPE>(matDp, 2);
#ifdef _DEBUG
    vecPhaseResult = CalcUtils::matToVector<DATA_TYPE>( matPhaseResult );
#endif

    MARK_FUNCTION_END_TIME;
    return matPhaseResult;
}

static inline cv::Mat calcOrder5BezierCoeff(const cv::Mat &matU) {
    cv::Mat matUSubBy1Pow1 = 1.f - matU;
    cv::Mat matUSubBy1Pow2 = matUSubBy1Pow1.mul(matUSubBy1Pow1);
    cv::Mat matUSubBy1Pow3 = matUSubBy1Pow2.mul(matUSubBy1Pow1);
    cv::Mat matUSubBy1Pow4 = matUSubBy1Pow3.mul(matUSubBy1Pow1);
    cv::Mat matUPow2 = matU.mul(matU);
    cv::Mat matUPow3 = matUPow2.mul(matU);
    cv::Mat matUPow4 = matUPow3.mul(matU);

    cv::Mat matP;
    matP = matUSubBy1Pow4;
    matP.push_back(cv::Mat(4 * matUSubBy1Pow3.mul(matU)));
    matP.push_back(cv::Mat(6 * matUSubBy1Pow2.mul(matUPow2)));
    matP.push_back(cv::Mat(4 * matUSubBy1Pow1.mul(matUPow3)));
    matP.push_back(matUPow4);
    return matP;
}

//The 5 order bezier surface has 25 parameters.
/*static*/ cv::Mat Unwrap::_calculatePPz(const cv::Mat &matX, const cv::Mat &matY, const cv::Mat &matZ) {
    int ROWS = matX.rows;
    int COLS = matX.cols;
    cv::Mat matU, matV;
    CalcUtils::meshgrid<float>(0, 1.f / (COLS - 1), 1.f, 0, 1.f / (ROWS - 1), 1.f, matU, matV);
    matU = matU.reshape(1, 1);
    matV = matV.reshape(1, 1);
    cv::Mat matZReshape = matZ.reshape(1, matZ.rows * matZ.cols);
    cv::Mat matXReshape = matX.reshape(1, 1);
    cv::Mat matYReshape = matY.reshape(1, 1);
    cv::Mat matP1 = calcOrder5BezierCoeff(matU);
    cv::Mat matP2 = calcOrder5BezierCoeff(matV);
    cv::Mat matXX;
    for (int i = 0; i < BEZIER_RANK; ++ i)
    for (int j = 0; j < BEZIER_RANK; ++ j) {
        matXX.push_back(cv::Mat(matP1.row(i).mul(matP2.row(j))));
    }
    cv::transpose(matXX, matXX);
    cv::Mat matBaseSurfaceParam;
    cv::solve(matXX, matZReshape, matBaseSurfaceParam, cv::DecompTypes::DECOMP_QR);
    return matBaseSurfaceParam;
}

/*static*/ cv::Mat Unwrap::calculateBaseSurface(int rows, int cols, const cv::Mat &matBaseSurfaceParam) {
    CStopWatch stopWatch;
    cv::Mat matU, matV;
    CalcUtils::meshgrid<float>(0, 1.f / (cols - 1), 1.f, 0, 1.f / (rows - 1), 1.f, matU, matV);
    matU = matU.reshape(1, 1);
    matV = matV.reshape(1, 1);
    cv::Mat matP1 = calcOrder5BezierCoeff(matU);
    cv::Mat matP2 = calcOrder5BezierCoeff(matV);
    cv::Mat matXX;
    for (int i = 0; i < BEZIER_RANK; ++ i)
    for (int j = 0; j < BEZIER_RANK; ++ j) {
        matXX.push_back(cv::Mat(matP1.row(i).mul(matP2.row(j))));
    }
    cv::transpose(matXX, matXX);
    cv::Mat matZP = matXX * matBaseSurfaceParam;
    matZP = matZP.reshape(1, rows);

    TimeLog::GetInstance()->addTimeLog(__FUNCTION__, stopWatch.Span());
    return matZP;
}

/*static*/ void Unwrap::calc3DHeight(const PR_CALC_3D_HEIGHT_CMD *const pstCmd, PR_CALC_3D_HEIGHT_RPY *const pstRpy) {
    CStopWatch stopWatch;
    std::vector<cv::Mat> vecConvertedImgs;
    for (auto &mat : pstCmd->vecInputImgs) {
        cv::Mat matConvert = mat;
        if (mat.channels() > 1)
            cv::cvtColor(mat, matConvert, CV_BGR2GRAY);
        vecConvertedImgs.push_back(matConvert);
    }
    TimeLog::GetInstance()->addTimeLog("Convert image to float.", stopWatch.Span());

    if (pstCmd->bReverseSeq) {
        std::swap(vecConvertedImgs[1], vecConvertedImgs[3]);
        std::swap(vecConvertedImgs[5], vecConvertedImgs[7]);
        if (pstCmd->bUseThinnestPattern)
            std::swap(vecConvertedImgs[9], vecConvertedImgs[11]);
    }

    auto size = vecConvertedImgs[0].size();
    auto total = ToInt32(vecConvertedImgs[0].total());
    cv::Mat matAlpha(size, CV_32FC1), matBeta(size, CV_32FC1), matGamma(size, CV_32FC1), matBuffer1(size, CV_32FC1), matBuffer2(size, CV_32FC1);
    cv::Mat matAvgUnderTolIndex = cv::Mat::zeros(size, CV_8UC1);
    float fMinimumAlpitudeSquare = pstCmd->fMinAmplitude * pstCmd->fMinAmplitude;

    auto ptrData0 = vecConvertedImgs[0].ptr<uchar>(0);
    auto ptrData1 = vecConvertedImgs[1].ptr<uchar>(0);
    auto ptrData2 = vecConvertedImgs[2].ptr<uchar>(0);
    auto ptrData3 = vecConvertedImgs[3].ptr<uchar>(0);
    for (int i = 0; i < total; ++ i) {
        short x = (short)ptrData0[i] - (short)ptrData2[i];
        short y = (short)ptrData3[i] - (short)ptrData1[i];
        float fAmplitudeSquare = (x*x + y*y) / 4.f;
        if (fAmplitudeSquare < fMinimumAlpitudeSquare)
            matAvgUnderTolIndex.at<uchar>(i) = PR_MAX_GRAY_LEVEL;
        matAlpha.at<float>(i) = vecVecAtan2Table[x + PR_MAX_GRAY_LEVEL][y + PR_MAX_GRAY_LEVEL];
    }

    //_fitHoleInNanMask(matAvgUnderTolIndex, cv::Size(10, 10), 2);

    ptrData0 = vecConvertedImgs[4].ptr<uchar>(0);
    ptrData1 = vecConvertedImgs[5].ptr<uchar>(0);
    ptrData2 = vecConvertedImgs[6].ptr<uchar>(0);
    ptrData3 = vecConvertedImgs[7].ptr<uchar>(0);
    for (int i = 0; i < total; ++ i) {
        short x = (short)ptrData0[i] - (short)ptrData2[i];  // x = 2*cos(beta)
        short y = (short)ptrData3[i] - (short)ptrData1[i];  // y = 2*sin(beta)        
        matBeta.at<float>(i) = vecVecAtan2Table[x + PR_MAX_GRAY_LEVEL][y + PR_MAX_GRAY_LEVEL];
    }
    TimeLog::GetInstance()->addTimeLog("2 lookup phase ", stopWatch.Span());

    if (pstCmd->bUseThinnestPattern) {
        ptrData0 = vecConvertedImgs[8].ptr<uchar>(0);
        ptrData1 = vecConvertedImgs[9].ptr<uchar>(0);
        ptrData2 = vecConvertedImgs[10].ptr<uchar>(0);
        ptrData3 = vecConvertedImgs[11].ptr<uchar>(0);
        for (int i = 0; i < total; ++i) {
            short x = (short)ptrData0[i] - (short)ptrData2[i];  // x = 2*cos(beta)
            short y = (short)ptrData3[i] - (short)ptrData1[i];  // y = 2*sin(beta)
            matGamma.at<float>(i) = vecVecAtan2Table[x + PR_MAX_GRAY_LEVEL][y + PR_MAX_GRAY_LEVEL];
        }
        matGamma = matGamma - pstCmd->matBaseWrappedGamma;
    }

    matAlpha = matAlpha - pstCmd->matBaseWrappedAlpha;
    matBeta = matBeta - pstCmd->matBaseWrappedBeta;
    TimeLog::GetInstance()->addTimeLog("2 substract take. ", stopWatch.Span());

#ifdef _DEBUG
    auto vecVecAlpha = CalcUtils::matToVector<DATA_TYPE>(matAlpha);
    auto vecVecBeta  = CalcUtils::matToVector<DATA_TYPE>(matBeta);
#endif

    _phaseWrapBuffer(matAlpha, matBuffer1, pstCmd->fPhaseShift);
    TimeLog::GetInstance()->addTimeLog("_phaseWrapBuffer.", stopWatch.Span());

    matBuffer1 = matAlpha * pstCmd->matThickToThinK.at<DATA_TYPE>(0);
    TimeLog::GetInstance()->addTimeLog("Multiply takes.", stopWatch.Span());

    _phaseWrapByRefer(matBeta, matBuffer1);
    TimeLog::GetInstance()->addTimeLog("_phaseWrapByRefer.", stopWatch.Span());

    if (pstCmd->nRemoveBetaJumpMinSpan > 0 || pstCmd->nRemoveBetaJumpMaxSpan > 0) {
        _phasePatch(matBeta, pstCmd->enProjectDir, matAvgUnderTolIndex);
        phaseCorrectionEx(matBeta, pstCmd->enProjectDir, pstCmd->enScanDir, pstCmd->nRemoveBetaJumpMinSpan, pstCmd->nRemoveBetaJumpMaxSpan);
        TimeLog::GetInstance()->addTimeLog("phaseCorrection for beta.", stopWatch.Span());
    }

    if (pstCmd->bUseThinnestPattern) {
        auto k1 = pstCmd->matThickToThinK.at<DATA_TYPE>(0);
        auto k2 = pstCmd->matThickToThinnestK.at<DATA_TYPE>(0);
        matBuffer1 = matBeta * k2 / k1;
        _phaseWrapByRefer(matGamma, matBuffer1);

        if (pstCmd->nRemoveGammaJumpSpanX > 0 || pstCmd->nRemoveGammaJumpSpanY > 0) {
            _phasePatch(matGamma, pstCmd->enProjectDir, matAvgUnderTolIndex);
            phaseCorrection(matGamma, matAvgUnderTolIndex, pstCmd->nRemoveGammaJumpSpanX, pstCmd->nRemoveGammaJumpSpanY);
            //Run 2 times, first time remove small jump, second time remove big jump. 2 pass can remove big jump with small jump on its edge.
            //phaseCorrection(matGamma, matAvgUnderTolIndex, pstCmd->nRemoveGammaJumpSpanX, pstCmd->nRemoveGammaJumpSpanY);
            TimeLog::GetInstance()->addTimeLog("phaseCorrection for gamma.", stopWatch.Span());
        }

        pstRpy->matPhase = matGamma * k1 / k2;
    }
    else
        pstRpy->matPhase = matBeta;

    cv::medianBlur(pstRpy->matPhase, pstRpy->matPhase, 5);

    if (pstCmd->bEnableGaussianFilter)
        cv::GaussianBlur(pstRpy->matPhase, pstRpy->matPhase, cv::Size(5, 5), 5, 5, cv::BorderTypes::BORDER_REPLICATE);

    pstRpy->matNanMask = matAvgUnderTolIndex;

    if (! pstCmd->matOrder3CurveSurface.empty() && ! pstCmd->matIntegratedK.empty())
        pstRpy->matHeight = _calcHeightFromPhaseBuffer(pstRpy->matPhase, pstCmd->matOrder3CurveSurface, pstCmd->matIntegratedK, matBuffer1, matBuffer2);
    else if (! pstCmd->matPhaseToHeightK.empty()) {
        cv::Mat matX, matY;
        CalcUtils::meshgrid<DATA_TYPE>(1.f, 1.f, ToFloat(pstRpy->matPhase.cols), 1.f, 1.f, ToFloat(pstRpy->matPhase.rows), matX, matY);
        cv::Mat matRatio = pstCmd->matPhaseToHeightK.at<DATA_TYPE>(0, 0) * matX + pstCmd->matPhaseToHeightK.at<DATA_TYPE>(1, 0) * matY + pstCmd->matPhaseToHeightK.at<DATA_TYPE>(2, 0);
        cv::divide(pstRpy->matPhase, matRatio, pstRpy->matHeight);
    }
    else
        pstRpy->matHeight = pstRpy->matPhase;

    TimeLog::GetInstance()->addTimeLog("_calcHeightFromPhase.", stopWatch.Span());
    pstRpy->enStatus = VisionStatus::OK;
}

/*static*/ void Unwrap::merge3DHeight(const PR_MERGE_3D_HEIGHT_CMD *const pstCmd, PR_MERGE_3D_HEIGHT_RPY *const pstRpy) {
    cv::Mat matHeightOne = pstCmd->vecMatHeight[0], matHeightTwo = pstCmd->vecMatHeight[1];
    if (pstCmd->enMethod == PR_MERGE_3D_HT_METHOD::SELECT_NEAREST_INTERSECT) {
        pstRpy->matHeight = _mergeHeightIntersect(
            matHeightOne,
            pstCmd->vecMatNanMask[0],
            matHeightTwo,
            pstCmd->vecMatNanMask[1],
            pstCmd->fHeightDiffThreshold,
            pstCmd->enProjDir);
    }else {
        pstRpy->matHeight = _mergeHeightMax(
            matHeightOne,
            pstCmd->vecMatNanMask[0],
            matHeightTwo,
            pstCmd->vecMatNanMask[1],
            pstCmd->fHeightDiffThreshold,
            pstCmd->enProjDir);
    }
    pstRpy->matNanMask = pstCmd->vecMatNanMask[0] & pstCmd->vecMatNanMask[1];
    pstRpy->enStatus = VisionStatus::OK;
}

/*static*/ cv::Mat Unwrap::_phaseUnwrapSurfaceTrk(const cv::Mat &matPhase, const cv::Mat &dxPhase, const cv::Mat &dyPhase, const cv::Mat &matBranchCut) {
    MARK_FUNCTION_START_TIME;

    int ROWS = matPhase.rows;
    int COLS = matPhase.cols;

    cv::Mat matPhaseResult;
    cv::Mat matUnwrapped = matBranchCut.clone();

    //1. Unwrap from left to right, but to accelerate the operation, transpose the matrix first,
    //and do it from top to bottom, it is the same effect as from left to right on original phase.
    cv::Mat matPhaseT;
    cv::transpose(matPhase, matPhaseT);

    TimeLog::GetInstance()->addTimeLog("cv::transpose. ", stopWatch.Span());

    cv::Mat matTmpPhase = cv::Mat(matPhaseT, cv::Rect(0, 0, matPhaseT.cols, 1));
    cv::Mat matDp = CalcUtils::diff(matTmpPhase, 1, 2);

#ifdef _DEBUG
    auto vecVecArray = CalcUtils::matToVector<DATA_TYPE>( dxPhase );
    auto vecmatTmp = CalcUtils::matToVector<DATA_TYPE>( matTmpPhase );
    auto vecVecMatDp_1 = CalcUtils::matToVector<DATA_TYPE>( matDp );
#endif

    cv::Mat matAbsUnderPI;
    cv::compare(cv::abs(matDp), cv::Scalar::all(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT);
    matDp.setTo(0, matAbsUnderPI);

    _setBySign(matDp, ONE_CYCLE);

    TimeLog::GetInstance()->addTimeLog("Before cv::repeat. ", stopWatch.Span());
    matDp = cv::repeat(matDp, matPhase.cols, 1);
    TimeLog::GetInstance()->addTimeLog("After cv::repeat. ", stopWatch.Span());

    cv::Mat matTmp(matPhaseT, cv::Rect(1, 0, matPhaseT.cols - 1, matPhaseT.rows));

    TimeLog::GetInstance()->addTimeLog("Before cumsum. ", stopWatch.Span());
    auto matCumSum = CalcUtils::cumsum<DATA_TYPE>(matDp, 2);
    TimeLog::GetInstance()->addTimeLog("After cumsum. ", stopWatch.Span());
    matTmp += matCumSum;

    //2. Unwrap from top to bottom. Because last step do a transpose, this step need to transpose back.
    cv::transpose(matPhaseT, matPhaseResult);
    matPhaseResult.setTo(cv::Scalar::all(NAN), matBranchCut);

    matDp = CalcUtils::diff(matPhaseResult, 1, 2);
    cv::compare(cv::abs(matDp), cv::Scalar::all(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT);
    matDp.setTo(0, matAbsUnderPI);
    _setBySign(matDp, ONE_CYCLE);

    cv::Mat matTmp1(matPhaseResult, cv::Rect(1, 0, matPhaseResult.cols - 1, matPhaseResult.rows));
    matCumSum = CalcUtils::cumsum<DATA_TYPE>(matDp, 2);
    matTmp1 += matCumSum;

#ifdef _DEBUG
    auto vecVecCumSum = CalcUtils::matToVector<DATA_TYPE> ( matCumSum );
#endif

    //3. Unwrap from Top to Bottom. Use the last row to unwrap next row.
    matPhaseResult.setTo(cv::Scalar(0), matBranchCut);
    cv::Mat matNan = CalcUtils::getNanMask(matPhaseResult);  //Find out points been blocked by branch cut.
    cv::Mat matIsRowNeedUnwrap;
    cv::reduce(matNan, matIsRowNeedUnwrap, 1, CV_REDUCE_MAX);
    std::vector<cv::Point> vecRowsNeedToUnwrap;
    cv::findNonZero(matIsRowNeedUnwrap, vecRowsNeedToUnwrap);
    for (const auto &point : vecRowsNeedToUnwrap) {
        int row = point.y;
        cv::Mat matOneRowIndex(matNan, cv::Range(row, row + 1));
        if (cv::sum(matOneRowIndex)[0] <= 1 || row <= 0)
            continue;
        cv::Mat matPhaseLastRow(matPhaseResult, cv::Range(row - 1, row));
        cv::Mat matPhaseDiff(dyPhase, cv::Range(row - 1, row));

        cv::Mat matZero;
        cv::compare(matPhaseLastRow, cv::Scalar(0), matZero, cv::CmpTypes::CMP_EQ);
        if (cv::sum(matZero)[0] > 0)
            matPhaseLastRow.setTo(cv::Scalar(NAN), matZero);

        matDp = matPhaseDiff;
        cv::compare(cv::abs(matDp), cv::Scalar(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT);
        matDp.setTo(0, matAbsUnderPI);
        _setBySign(matDp, ONE_CYCLE);

        cv::Mat matPhaseLastRowClone = matPhaseLastRow.clone();
        cv::accumulate(matPhaseDiff, matPhaseLastRowClone, matOneRowIndex);
        cv::accumulate(matDp, matPhaseLastRowClone, matOneRowIndex);

        cv::Mat matOneRowPhase(matPhaseResult, cv::Range(row, row + 1));
        matPhaseLastRowClone.copyTo(matOneRowPhase, matOneRowIndex);
    }

    //4. Unwrap from bottom to top. Use the next row to unwrap last row.
    matPhaseResult.setTo(cv::Scalar(0), matBranchCut);
    matNan = CalcUtils::getNanMask(matPhaseResult);  //Find out points been blocked by branch cut.

    cv::reduce(matNan, matIsRowNeedUnwrap, 1, CV_REDUCE_MAX);
    vecRowsNeedToUnwrap.clear();
    cv::findNonZero(matIsRowNeedUnwrap, vecRowsNeedToUnwrap);
    std::vector<int> vecRows;
    for (const auto &point : vecRowsNeedToUnwrap)
        vecRows.push_back(point.y);
    std::sort(vecRows.begin(), vecRows.end());
    std::reverse(vecRows.begin(), vecRows.end());
    for (const auto &row : vecRows) {
        cv::Mat matOneRowIndex(matNan, cv::Range(row, row + 1));
        if (cv::sum(matOneRowIndex)[0] <= 1 || row >= ROWS - 1)
            continue;
        cv::Mat matPhaseNextRow(matPhaseResult, cv::Range(row + 1, row + 2));
        cv::Mat matPhaseDiff(dyPhase, cv::Range(row, row + 1));

        cv::Mat matZero;
        cv::compare(matPhaseNextRow, cv::Scalar(0), matZero, cv::CmpTypes::CMP_EQ);
        if (cv::sum(matZero)[0] > 0)
            matPhaseNextRow.setTo(cv::Scalar(NAN), matZero);

        matDp = matPhaseDiff;
        cv::compare(cv::abs(matDp), cv::Scalar(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT);
        matDp.setTo(0, matAbsUnderPI);
        _setBySign(matDp, ONE_CYCLE);

        cv::Mat matPhaseNextRowClone = matPhaseNextRow.clone();
        cv::accumulate(matPhaseDiff, matPhaseNextRowClone, matOneRowIndex);
        cv::accumulate(matDp, matPhaseNextRowClone, matOneRowIndex);

        cv::Mat matOneRowPhase(matPhaseResult, cv::Range(row, row + 1));
        matPhaseNextRowClone.copyTo(matOneRowPhase, matOneRowIndex);
    }

    //5. Unwrap from right to left. One column by one column.
    matPhaseResult.setTo(cv::Scalar(0), matBranchCut);
    matNan = CalcUtils::getNanMask(matPhaseResult);

    cv::Mat matIsColNeedUnwrap;
    cv::reduce(matNan, matIsColNeedUnwrap, 0, CV_REDUCE_MAX);
    std::vector<cv::Point> vecColsNeedToUnwrap;
    cv::findNonZero(matIsColNeedUnwrap, vecColsNeedToUnwrap);
    std::vector<int> vecCols;
    for (const auto &point : vecColsNeedToUnwrap)
        vecCols.push_back(point.x);
    std::sort(vecCols.begin(), vecCols.end());
    std::reverse(vecCols.begin(), vecCols.end());
    for (const auto col : vecCols) {
        cv::Mat matOneColIndex(matNan, cv::Range::all(), cv::Range(col, col + 1));
        if (cv::sum(matOneColIndex)[0] <= 1 || col >= COLS - 1)
            continue;
        cv::Mat matPhaseNextCol(matPhaseResult, cv::Range::all(), cv::Range(col + 1, col + 2));
        cv::Mat matPhaseDiff(dxPhase, cv::Range::all(), cv::Range(col, col + 1));

        cv::Mat matZero;
        cv::compare(matPhaseNextCol, cv::Scalar(0), matZero, cv::CmpTypes::CMP_EQ);
        if (cv::sum(matZero)[0] > 0)
            matPhaseNextCol.setTo(cv::Scalar(NAN), matZero);

        matDp = matPhaseDiff;
        cv::compare(cv::abs(matDp), cv::Scalar::all(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT);
        matDp.setTo(0, matAbsUnderPI);
        _setBySign(matDp, ONE_CYCLE);

        cv::Mat matPhaseNextColClone = matPhaseNextCol.clone();
        cv::accumulate(matPhaseDiff, matPhaseNextColClone, matOneColIndex);
        cv::accumulate(matDp, matPhaseNextColClone, matOneColIndex);

        cv::Mat matOneColPhase(matPhaseResult, cv::Range::all(), cv::Range(col, col + 1));
        matPhaseNextColClone.copyTo(matOneColPhase, matOneColIndex);
    }

#ifdef _DEBUG
    matNan = CalcUtils::getNanMask ( matPhaseResult );
    cv::imwrite("./data/NanMap_2.png", matNan);
#endif

    matPhaseResult.setTo(cv::Scalar(NAN), matBranchCut);

#ifdef _DEBUG
    matNan = CalcUtils::getNanMask ( matPhaseResult );
    cv::imwrite("./data/NanMap_3.png", matNan);
#endif

    MARK_FUNCTION_END_TIME;

    return matPhaseResult;
}

/*static*/ cv::Mat Unwrap::_phaseUnwrapSurfaceTrkNew(const cv::Mat &matPhase, const cv::Mat &dxPhase, const cv::Mat &dyPhase, const cv::Mat &matBranchCut) {
    MARK_FUNCTION_START_TIME;

    int ROWS = matPhase.rows;
    int COLS = matPhase.cols;

    cv::Mat matPhaseResult = matPhase.clone();
    const float OneCycle = 2.f * ONE_HALF_CYCLE;

    //1. Unwrap from left to right, but to accelerate the operation, transpose the matrix first,
    //and do it from top to bottom, it is the same effect as from left to right on original phase.
    cv::Mat matTmpPhase = cv::Mat(matPhase, cv::Rect(0, 0, 1, ROWS));
    cv::Mat matDp = CalcUtils::diff(matTmpPhase, 1, 1);

#ifdef _DEBUG
    auto vecVecArray = CalcUtils::matToVector<DATA_TYPE>( dxPhase );
    auto vecmatTmp = CalcUtils::matToVector<DATA_TYPE>( matTmpPhase );
    auto vecVecMatDp_1 = CalcUtils::matToVector<DATA_TYPE>( matDp );
#endif

    cv::Mat matAbsUnderPI;
    cv::compare(cv::abs(matDp), cv::Scalar::all(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT);
    matDp.setTo(0, matAbsUnderPI);

    _setBySign(matDp, OneCycle);

    TimeLog::GetInstance()->addTimeLog("Before cv::repeat. ", stopWatch.Span());
    matDp = CalcUtils::repeatInX<DATA_TYPE>(matDp, matPhase.cols);
    TimeLog::GetInstance()->addTimeLog("After cv::repeat. ", stopWatch.Span());

    TimeLog::GetInstance()->addTimeLog("Before cumsum. ", stopWatch.Span());
    auto matCumSum = CalcUtils::cumsum<DATA_TYPE>(matDp, 1);
    TimeLog::GetInstance()->addTimeLog("After cumsum. ", stopWatch.Span());

    cv::Mat matTmp(matPhaseResult, cv::Rect(0, 1, COLS, ROWS - 1));
    matTmp += matCumSum;

    //2. Unwrap from top to bottom. Because last step do a transpose, this step need to transpose back.
    matPhaseResult.setTo(cv::Scalar::all(NAN), matBranchCut);

    matDp = CalcUtils::diff(matPhaseResult, 1, 2);
    cv::compare(cv::abs(matDp), cv::Scalar::all(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT);
    matDp.setTo(0, matAbsUnderPI);
    _setBySign(matDp, OneCycle);

    cv::Mat matTmp1(matPhaseResult, cv::Rect(1, 0, matPhaseResult.cols - 1, matPhaseResult.rows));
    matCumSum = CalcUtils::cumsum<DATA_TYPE>(matDp, 2);
    matTmp1 += matCumSum;

#ifdef _DEBUG
    auto vecVecCumSum = CalcUtils::matToVector<DATA_TYPE> ( matCumSum );
#endif

    //3. Unwrap from Top to Bottom. Use the last row to unwrap next row.
    matPhaseResult.setTo(cv::Scalar(0), matBranchCut);
    cv::Mat matNan = CalcUtils::getNanMask(matPhaseResult);  //Find out points been blocked by branch cut.
    cv::Mat matIsRowNeedUnwrap;
    cv::reduce(matNan, matIsRowNeedUnwrap, 1, CV_REDUCE_MAX);
    std::vector<cv::Point> vecRowsNeedToUnwrap;
    cv::findNonZero(matIsRowNeedUnwrap, vecRowsNeedToUnwrap);
    std::vector<int> vecRows;
    for (const auto &point : vecRowsNeedToUnwrap)
        vecRows.push_back(point.y);
    std::sort(vecRows.begin(), vecRows.end());
    std::reverse(vecRows.begin(), vecRows.end());
    for (const auto &row : vecRows) {
        cv::Mat matOneRowIndex(matNan, cv::Range(row, row + 1));
        if (cv::sum(matOneRowIndex)[0] <= 1 || row >= ROWS - 1)
            continue;
        cv::Mat matPhaseNextRow(matPhaseResult, cv::Range(row + 1, row + 2));
        cv::Mat matPhaseDiff(dyPhase, cv::Range(row, row + 1));

        cv::Mat matZero;
        cv::compare(matPhaseNextRow, cv::Scalar(0), matZero, cv::CmpTypes::CMP_EQ);
        if (cv::sum(matZero)[0] > 0)
            matPhaseNextRow.setTo(cv::Scalar(NAN), matZero);

        matDp = matPhaseDiff;
        cv::compare(cv::abs(matDp), cv::Scalar(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT);
        matDp.setTo(0, matAbsUnderPI);
        _setBySign(matDp, OneCycle);

        //Must use clone, and copy it back. Othereise the time is very slow.
        cv::Mat matPhaseNextRowClone = matPhaseNextRow.clone();
        cv::accumulate(matPhaseDiff, matPhaseNextRowClone, matOneRowIndex);
        cv::accumulate(matDp, matPhaseNextRowClone, matOneRowIndex);

        cv::Mat matOneRowPhase(matPhaseResult, cv::Range(row, row + 1));
        matPhaseNextRowClone.copyTo(matOneRowPhase, matOneRowIndex);
    }

    //4. Unwrap from bottom to top. Use the next row to unwrap last row.
    matPhaseResult.setTo(cv::Scalar(0), matBranchCut);
    matNan = CalcUtils::getNanMask(matPhaseResult);  //Find out points been blocked by branch cut.    
    cv::reduce(matNan, matIsRowNeedUnwrap, 1, CV_REDUCE_MAX);
    vecRowsNeedToUnwrap.clear();
    cv::findNonZero(matIsRowNeedUnwrap, vecRowsNeedToUnwrap);
    for (const auto &point : vecRowsNeedToUnwrap) {
        int row = point.y;
        cv::Mat matOneRowIndex(matNan, cv::Range(row, row + 1));
        if (cv::sum(matOneRowIndex)[0] <= 1 || row <= 0)
            continue;
        cv::Mat matPhaseLastRow(matPhaseResult, cv::Range(row - 1, row));
        cv::Mat matPhaseDiff(dyPhase, cv::Range(row - 1, row));

        cv::Mat matZero;
        cv::compare(matPhaseLastRow, cv::Scalar(0), matZero, cv::CmpTypes::CMP_EQ);
        if (cv::sum(matZero)[0] > 0)
            matPhaseLastRow.setTo(cv::Scalar(NAN), matZero);

        matDp = matPhaseDiff;
        cv::compare(cv::abs(matDp), cv::Scalar(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT);
        matDp.setTo(0, matAbsUnderPI);
        _setBySign(matDp, OneCycle);

        cv::Mat matPhaseLastRowClone = matPhaseLastRow.clone();
        cv::accumulate(matPhaseDiff, matPhaseLastRowClone, matOneRowIndex);
        cv::accumulate(matDp, matPhaseLastRowClone, matOneRowIndex);

        cv::Mat matOneRowPhase(matPhaseResult, cv::Range(row, row + 1));
        matPhaseLastRowClone.copyTo(matOneRowPhase, matOneRowIndex);
    }

    //5. Unwrap from right to left. One column by one column.
    matPhaseResult.setTo(cv::Scalar(0), matBranchCut);
    matNan = CalcUtils::getNanMask(matPhaseResult);

    cv::Mat matIsColNeedUnwrap;
    cv::reduce(matNan, matIsColNeedUnwrap, 0, CV_REDUCE_MAX);
    std::vector<cv::Point> vecColsNeedToUnwrap;
    cv::findNonZero(matIsColNeedUnwrap, vecColsNeedToUnwrap);
    std::vector<int> vecCols;
    for (const auto &point : vecColsNeedToUnwrap)
        vecCols.push_back(point.x);
    std::sort(vecCols.begin(), vecCols.end());
    std::reverse(vecCols.begin(), vecCols.end());
    for (const auto col : vecCols) {
        cv::Mat matOneColIndex(matNan, cv::Range::all(), cv::Range(col, col + 1));
        if (cv::sum(matOneColIndex)[0] <= 1 || col >= COLS - 1)
            continue;
        cv::Mat matPhaseNextCol(matPhaseResult, cv::Range::all(), cv::Range(col + 1, col + 2));
        cv::Mat matPhaseDiff(dxPhase, cv::Range::all(), cv::Range(col, col + 1));

        cv::Mat matZero;
        cv::compare(matPhaseNextCol, cv::Scalar(0), matZero, cv::CmpTypes::CMP_EQ);
        if (cv::sum(matZero)[0] > 0)
            matPhaseNextCol.setTo(cv::Scalar(NAN), matZero);

        matDp = matPhaseDiff;
        cv::compare(cv::abs(matDp), cv::Scalar::all(ONE_HALF_CYCLE), matAbsUnderPI, cv::CmpTypes::CMP_LT);
        matDp.setTo(0, matAbsUnderPI);
        _setBySign(matDp, OneCycle);

        cv::Mat matPhaseNextColClone = matPhaseNextCol.clone();
        cv::accumulate(matPhaseDiff, matPhaseNextColClone, matOneColIndex);
        cv::accumulate(matDp, matPhaseNextColClone, matOneColIndex);

        cv::Mat matOneColPhase(matPhaseResult, cv::Range::all(), cv::Range(col, col + 1));
        matPhaseNextColClone.copyTo(matOneColPhase, matOneColIndex);
    }

#ifdef _DEBUG
    matNan = CalcUtils::getNanMask ( matPhaseResult );
    cv::imwrite("./data/NanMap_2.png", matNan);
#endif

    matPhaseResult.setTo(cv::Scalar(NAN), matBranchCut);

#ifdef _DEBUG
    matNan = CalcUtils::getNanMask ( matPhaseResult );
    cv::imwrite("./data/NanMap_3.png", matNan);
#endif

    MARK_FUNCTION_END_TIME;
    return matPhaseResult;
}

/*static*/ cv::Mat Unwrap::_phaseUnwrapSurfaceByRefer(const cv::Mat &matPhase, const cv::Mat &matRef, const cv::Mat &matNan) {
    MARK_FUNCTION_START_TIME;

    cv::Mat matPhaseResult = matPhase.clone();
    if (! matNan.empty())
        matPhaseResult.setTo(NAN, matNan);

    cv::Mat matErr = matPhaseResult - matRef;
    cv::Mat matNn = matErr / ONE_CYCLE + 0.5;
    CalcUtils::floorByRef<DATA_TYPE>(matNn);
    matPhaseResult = matPhaseResult - matNn * ONE_CYCLE;

    MARK_FUNCTION_END_TIME;
    return matPhaseResult;
}

//For shadow area
/*static*/ void Unwrap::_findUnstablePoint(const std::vector<cv::Mat> &vecInputImgs, float fDiffTol, float fAvgTol, cv::Mat &matDiffUnderTolIndex, cv::Mat &matAvgUnderTolIndex) {
    MARK_FUNCTION_START_TIME;

    //int ROWS = vecInputImgs[0].rows;
    //int COLS = vecInputImgs[0].cols;
    //matDiffUnderTolIndex = cv::Mat::zeros(ROWS, COLS, CV_8UC1);
    //matAvgUnderTolIndex  = cv::Mat::zeros(ROWS, COLS, CV_8UC1);
    //const int GROUP_IMG_COUNT = 4;
    //for ( int row = 0; row < ROWS; ++ row )
    //for ( int col = 0; col < COLS; ++ col ) {
    //    std::vector<DATA_TYPE> vecData(GROUP_IMG_COUNT);
    //    for ( size_t i = 0; i < vecInputImgs.size(); ++ i )
    //        vecData[i] = vecInputImgs[i].at<DATA_TYPE>(row, col);
    //    DATA_TYPE minValue = std::numeric_limits<DATA_TYPE>::max(), maxValue = std::numeric_limits<DATA_TYPE>::min(), avgValue = 0.f;
    //    for ( auto data : vecData ) {
    //        if ( minValue > data ) minValue = data;
    //        if ( maxValue < data ) maxValue = data;
    //        avgValue += data;
    //    }
    //    avgValue /= GROUP_IMG_COUNT;
    //    if ( (maxValue - minValue) < fDiffTol )
    //        matDiffUnderTolIndex.at<uchar>(row, col) = 1;
    //    if ( avgValue < fAvgTol )
    //        matAvgUnderTolIndex.at<uchar>(row, col) = 1;
    //}

    cv::Mat matCombine;
    const int GROUP_IMG_COUNT = 4;
    int ROWS = vecInputImgs[4].rows;
    int COLS = vecInputImgs[4].cols;
    int nTotal = ToInt32(vecInputImgs[4].total());
    cv::Mat matArray[] = {vecInputImgs[4].reshape(1, 1), vecInputImgs[5].reshape(1, 1), vecInputImgs[6].reshape(1, 1), vecInputImgs[7].reshape(1, 1)};
    cv::vconcat(matArray, GROUP_IMG_COUNT, matCombine);
    //for ( int i = 0; i < GROUP_IMG_COUNT; ++ i )
    //    matCombine.push_back ( vecInputImgs[i].reshape ( 1, 1 ) );
    //cv::transpose ( matCombine, matCombine );
    TimeLog::GetInstance()->addTimeLog("_findUnstablePoint combine image. ", stopWatch.Span());
    cv::Mat matMax, matMin, matAvg;
    cv::reduce(matCombine, matMax, 0, cv::ReduceTypes::REDUCE_MAX);
    cv::reduce(matCombine, matMin, 0, cv::ReduceTypes::REDUCE_MIN);
    cv::reduce(matCombine, matAvg, 0, cv::ReduceTypes::REDUCE_AVG);
    TimeLog::GetInstance()->addTimeLog("3 cv::reduce combine image. ", stopWatch.Span());
    cv::Mat matDiff = matMax - matMin;
    cv::compare(matDiff, fDiffTol, matDiffUnderTolIndex, cv::CmpTypes::CMP_LT);
    cv::compare(matAvg, fAvgTol, matAvgUnderTolIndex, cv::CmpTypes::CMP_LT);
    matDiffUnderTolIndex = matDiffUnderTolIndex.reshape(1, ROWS);
    matAvgUnderTolIndex = matAvgUnderTolIndex.reshape(1, ROWS);
    MARK_FUNCTION_END_TIME;
}

/*static*/ void Unwrap::calib3DHeight(const PR_CALIB_3D_HEIGHT_CMD *const pstCmd, PR_CALIB_3D_HEIGHT_RPY *const pstRpy) {
    PR_CALC_3D_HEIGHT_CMD stCalcCmd;
    PR_CALC_3D_HEIGHT_RPY stCalcRpy;
    stCalcCmd.vecInputImgs = std::move(pstCmd->vecInputImgs);
    stCalcCmd.bEnableGaussianFilter = pstCmd->bEnableGaussianFilter;
    stCalcCmd.bReverseSeq = pstCmd->bReverseSeq;
    stCalcCmd.fRemoveHarmonicWaveK = pstCmd->fRemoveHarmonicWaveK;
    stCalcCmd.bUseThinnestPattern = pstCmd->bUseThinnestPattern;
    stCalcCmd.fMinAmplitude = pstCmd->fMinAmplitude;
    stCalcCmd.matThickToThinK = pstCmd->matThickToThinK;
    stCalcCmd.matThickToThinnestK = pstCmd->matThickToThinnestK;
    stCalcCmd.matBaseWrappedAlpha = pstCmd->matBaseWrappedAlpha;
    stCalcCmd.matBaseWrappedBeta = pstCmd->matBaseWrappedBeta;
    stCalcCmd.matBaseWrappedGamma = pstCmd->matBaseWrappedGamma;
    stCalcCmd.nRemoveBetaJumpMinSpan = pstCmd->nRemoveBetaJumpMinSpan;
    stCalcCmd.nRemoveBetaJumpMaxSpan = pstCmd->nRemoveBetaJumpMaxSpan;
    stCalcCmd.nRemoveGammaJumpSpanX = pstCmd->nRemoveGammaJumpSpanX;
    stCalcCmd.nRemoveGammaJumpSpanY = pstCmd->nRemoveGammaJumpSpanY;
    calc3DHeight(&stCalcCmd, &stCalcRpy);
    if (VisionStatus::OK != stCalcRpy.enStatus) {
        WriteLog("Unwrap::calc3DHeight failed.");
        pstRpy->enStatus = stCalcRpy.enStatus;
        return;
    }

    cv::Mat matPhase = stCalcRpy.matPhase;
    cv::medianBlur(matPhase, matPhase, 5);
    cv::Mat matTmpPhase = matPhase.clone();
    pstRpy->matPhase = matPhase.clone();
    int ROWS = matPhase.rows;
    int COLS = matPhase.cols;
    cv::Mat matST = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ERODE_WIN_SIZE, ERODE_WIN_SIZE));

    const int NN = 2;
    for (int n = 1; n <= NN; ++ n) {
        double dMinValue = 0, dMaxValue = 0;
        cv::Point ptMin, ptMax;
        VectorOfPoint vecPtLocations;
        cv::Mat matNormalizedPhase;
        VectorOfFloat vecNormaledThreshold;
        for (int nRetryToFindBase = 0; nRetryToFindBase < 2; ++ nRetryToFindBase) {
            cv::Mat matMask = (matTmpPhase == matTmpPhase); //Find out value is not NAN.
            cv::minMaxLoc(matTmpPhase, &dMinValue, &dMaxValue, &ptMin, &ptMax, matMask);
            matNormalizedPhase = (matTmpPhase - dMinValue) / (dMaxValue - dMinValue);
            cv::Mat matGrayScalePhase = matNormalizedPhase * PR_MAX_GRAY_LEVEL;
            matGrayScalePhase.convertTo(pstRpy->matDivideStepResultImg, CV_8U);
            cv::cvtColor(pstRpy->matDivideStepResultImg, pstRpy->matDivideStepResultImg, CV_GRAY2BGR);
            pstRpy->matDivideStepIndex = cv::Mat::zeros(pstRpy->matDivideStepResultImg.size(), CV_8SC1);
            auto vecThreshold = VisionAlgorithm::autoMultiLevelThreshold(matGrayScalePhase, matMask, pstCmd->nBlockStepCount);

            vecNormaledThreshold.push_back(0.f);
            for (const auto thresh : vecThreshold)
                vecNormaledThreshold.push_back(ToFloat(thresh) / ToFloat(PR_MAX_GRAY_LEVEL));
            vecNormaledThreshold.push_back(1.f);

            auto nKK = 0;
            if (pstCmd->bReverseHeight)
                nKK = ToInt32(vecNormaledThreshold.size()) - 2;
            cv::Mat matInRangeBaseSurface;
            cv::inRange(matNormalizedPhase, vecNormaledThreshold[nKK], vecNormaledThreshold[nKK + 1], matInRangeBaseSurface);
#ifdef _DEBUG
            auto vevVecNormalizedPhase = CalcUtils::matToVector<DATA_TYPE> ( matNormalizedPhase );
            cv::Mat matAutoThresholdResult = _drawAutoThresholdResult ( matNormalizedPhase, vecNormaledThreshold );
            cv::imwrite ( "./data/calib3DHeight_AutoThresholdResult.png", matAutoThresholdResult );
            cv::imwrite ( "./data/calib3DHeight_NormGrayScale.png", matGrayScalePhase );
#endif
            auto rectBounding = cv::boundingRect(matInRangeBaseSurface);
            cv::findNonZero(matInRangeBaseSurface, vecPtLocations);
            if (rectBounding.width > CALIB_HEIGHT_MIN_SIZE && rectBounding.height > CALIB_HEIGHT_MIN_SIZE && vecPtLocations.size() > 5000) {
                //We are good here, find a good base surface, can exit the retry and continue the next step.
                break;
            }
            else if (0 == nRetryToFindBase) {
                //The first time retry, remove the fake base surface to NAN, and retry again.
                matTmpPhase.setTo(NAN, matInRangeBaseSurface);
                vecNormaledThreshold.clear();
            }
            else if (1 == nRetryToFindBase) {
                WriteLog("The base surface size is too small to calibrate 3D height.");
                pstRpy->enStatus = VisionStatus::CALIB_3D_HEIGHT_SURFACE_TOO_SMALL;
                return;
            }
        }
        std::sort(vecPtLocations.begin(), vecPtLocations.end(), [&matNormalizedPhase](const cv::Point &pt1, const cv::Point &pt2) {
            return matNormalizedPhase.at<DATA_TYPE>(pt1) < matNormalizedPhase.at<DATA_TYPE>(pt2);
        });
        VectorOfPoint vecTrimedLocations(vecPtLocations.begin() + ToInt32(vecPtLocations.size() * 0.1), vecPtLocations.begin() + ToInt32(vecPtLocations.size() * 0.9));
        cv::Mat matX, matY;
        CalcUtils::meshgrid<DATA_TYPE>(1.f, 1.f, ToFloat(COLS), 1.f, 1.f, ToFloat(ROWS), matX, matY);
        std::vector<DATA_TYPE> vecXt, vecYt, vecPhaseTmp;
        vecXt.reserve(vecTrimedLocations.size());
        vecYt.reserve(vecTrimedLocations.size());
        vecPhaseTmp.reserve(vecTrimedLocations.size());
        for (const auto &point : vecTrimedLocations) {
            vecXt.push_back(matX.at<DATA_TYPE>(point));
            vecYt.push_back(matY.at<DATA_TYPE>(point));
            vecPhaseTmp.push_back(matPhase.at<DATA_TYPE>(point));
        }
        cv::Mat matXX;
        matXX.push_back(cv::Mat(vecXt).reshape(1, 1));
        matXX.push_back(cv::Mat(vecYt).reshape(1, 1));
        matXX.push_back(cv::Mat(cv::Mat::ones(1, ToInt32(vecTrimedLocations.size()), CV_32FC1)));
        cv::transpose(matXX, matXX);
        cv::Mat matYY(vecPhaseTmp);
        cv::Mat matK;
        cv::solve(matXX, matYY, matK, cv::DecompTypes::DECOMP_QR);
        cv::Mat matZp = matX * matK.at<DATA_TYPE>(0, 0) + matY * matK.at<DATA_TYPE>(1, 0) + matK.at<DATA_TYPE>(2, 0);
#ifdef _DEBUG
        auto vevVecK = CalcUtils::matToVector<DATA_TYPE> ( matK );
        auto vecVecZp = CalcUtils::matToVector<DATA_TYPE> ( matZp );
        auto vecVecPhase = CalcUtils::matToVector<DATA_TYPE> ( matPhase );
#endif
        auto meanValue = cv::mean(matZp)[0];
        if (meanValue > 1.) {
            WriteLog("The H 0 step is not found. Please make sure the input image has zero phase, the bReverseHeight and nBlockStepCount input value are correct.");
            pstRpy->enStatus = VisionStatus::CALIB_3D_HEIGHT_NO_BASE_STEP;
            return;
        }

        matPhase = matPhase - matZp;

        cv::Mat matHz;
        VectorOfPoint vecPointToFectch;
        vecPointToFectch.emplace_back(COLS / 2, ROWS / 2);
        vecPointToFectch.emplace_back(0, 0);
        vecPointToFectch.emplace_back(COLS - 1, 0);
        vecPointToFectch.emplace_back(0, ROWS - 1);
        vecPointToFectch.emplace_back(COLS - 1, ROWS - 1);
        for (int index = 0; index < ToInt32(vecNormaledThreshold.size()) - 1; ++ index) {
            cv::Mat matBW = cv::Mat::zeros(ROWS, COLS, CV_8UC1);
            cv::inRange(matNormalizedPhase, vecNormaledThreshold[index], vecNormaledThreshold[index + 1], matBW);
#ifdef _DEBUG
            int nCountOfPoint = cv::countNonZero ( matBW );
#endif
            cv::Mat matFirstRow(matBW, cv::Rect(0, 0, COLS, 1)); matFirstRow.setTo(0);
            cv::Mat matFirstCol(matBW, cv::Rect(0, 0, 1, ROWS)); matFirstCol.setTo(0);
            cv::Mat matLastRow(matBW, cv::Rect(0, ROWS - 1, COLS, 1)); matLastRow.setTo(0);
            cv::Mat matLastCol(matBW, cv::Rect(COLS - 1, 0, 1, ROWS)); matLastCol.setTo(0);
            cv::erode(matBW, matBW, matST);
            pstRpy->matDivideStepResultImg.setTo(MATLAB_COLOR_ORDER[index], matBW);

            if (NN == n) {
                if (pstCmd->bReverseHeight)
                    pstRpy->matDivideStepIndex.setTo(index - pstCmd->nBlockStepCount - 1, matBW);
                else
                    pstRpy->matDivideStepIndex.setTo(index + 1, matBW);
            }

            VectorOfPoint vecPtLocations;
            cv::findNonZero(matBW, vecPtLocations);
            if (vecPtLocations.size() < 10) {
                WriteLog("The step area is too small to calibrate 3D height. Please check if the input step count is correct.");
                pstRpy->enStatus = VisionStatus::CALIB_3D_HEIGHT_SURFACE_TOO_SMALL;
                return;
            }
            for (int i = 0; i < 2; ++ i) {
                std::vector<DATA_TYPE> vecXt, vecYt, vecPhaseTmp;
                vecXt.reserve(vecPtLocations.size());
                vecYt.reserve(vecPtLocations.size());
                vecPhaseTmp.reserve(vecPtLocations.size());
                for (const auto &point : vecPtLocations) {
                    vecXt.push_back(matX.at<DATA_TYPE>(point));
                    vecYt.push_back(matY.at<DATA_TYPE>(point));
                    vecPhaseTmp.push_back(matPhase.at<DATA_TYPE>(point));
                }
                cv::Mat matXX;
                matXX.push_back(cv::Mat(vecXt).reshape(1, 1));
                matXX.push_back(cv::Mat(vecYt).reshape(1, 1));
                matXX.push_back(cv::Mat(cv::Mat::ones(1, ToInt32(vecPtLocations.size()), CV_32FC1)));
                cv::transpose(matXX, matXX);
                cv::Mat matYY(vecPhaseTmp);
                cv::solve(matXX, matYY, matK, cv::DecompTypes::DECOMP_SVD);
                if (i == 0) {
                    cv::Mat matErr = cv::abs(matYY - matXX * matK);
                    cv::Mat matErrSort;
                    cv::sortIdx(matErr, matErrSort, cv::SortFlags::SORT_ASCENDING + cv::SortFlags::SORT_EVERY_COLUMN);
#ifdef _DEBUG
                    cv::Mat matErrSortTranspose;
                    cv::transpose ( matErrSort, matErrSortTranspose );
                    auto vevVecSortIndex = CalcUtils::matToVector<int>(matErrSortTranspose);
#endif
                    VectorOfPoint vecTmpPoints;
                    int nKeepPoints = ToInt32(std::floor(matErrSort.rows * CALIB_HEIGHT_STEP_USEFUL_PT));
                    vecTmpPoints.reserve(nKeepPoints);
                    for (int j = 0; j < nKeepPoints; ++ j)
                        vecTmpPoints.push_back(vecPtLocations[matErrSort.at<int>(j)]);
                    std::swap(vecTmpPoints, vecPtLocations);
                }
            }
#ifdef _DEBUG
            auto vevVecK = CalcUtils::matToVector<DATA_TYPE> ( matK );
#endif
            matZp = matX * matK.at<DATA_TYPE>(0, 0) + matY * matK.at<DATA_TYPE>(1, 0) + matK.at<DATA_TYPE>(2, 0);
            pstRpy->vecMatStepSurface.push_back(matZp);
            std::vector<DATA_TYPE> vecValue;
            for (const auto &point : vecPointToFectch)
                vecValue.push_back(matZp.at<DATA_TYPE>(point));
            matHz.push_back(cv::Mat(vecValue).reshape(1, 1));
        }

        pstRpy->vecVecStepPhase = CalcUtils::matToVector<DATA_TYPE>(matHz);

        {//Temparory code scope.
            cv::Mat matXp, matXpT, matXt0T, matYt0T, matHzT;
            if (pstCmd->bReverseHeight)
                matXp = CalcUtils::intervals<DATA_TYPE>(pstCmd->nBlockStepCount, -1, 0);
            else
                matXp = CalcUtils::intervals<DATA_TYPE>(0, 1, pstCmd->nBlockStepCount);
            matXp = cv::repeat(matXp, 1, 5);
            cv::transpose(matXp, matXpT);
            matXpT = matXpT.reshape(1, 1);

            vecXt = std::vector<DATA_TYPE>({ToFloat(COLS / 2), 1.f, ToFloat(COLS), 1.f, ToFloat(COLS)});
            cv::Mat matXt0 = cv::Mat(vecXt).reshape(1, 1);
            matXt0 = cv::repeat(matXt0, pstCmd->nBlockStepCount + 1, 1);
            cv::transpose(matXt0, matXt0T);
            matXt0T = matXt0T.reshape(1, 1);

            vecYt = std::vector<DATA_TYPE>({ToFloat(ROWS / 2), 1.f, 1.f, ToFloat(ROWS), ToFloat(ROWS)});
            cv::Mat matYt0 = cv::Mat(vecYt).reshape(1, 1);
            matYt0 = cv::repeat(matYt0, pstCmd->nBlockStepCount + 1, 1);
            cv::transpose(matYt0, matYt0T);
            matYt0T = matYt0T.reshape(1, 1);

            cv::transpose(matHz, matHzT);
            matHzT = matHzT.reshape(1, ToInt32(matHzT.total()));

            std::vector<int> vecIndex(vecPointToFectch.size() * (pstCmd->nBlockStepCount + 1));
            std::iota(vecIndex.begin(), vecIndex.end(), 0);
            //Data fitting using multiple level data.
            //Z4 = k(1) * X * 4 + K(2) * Y * 4 + K(3) * 4
            //....
            //Z3 = k(1) * X * 3 + K(2) * Y * 3 + K(3) * 3
            //....
            //Z2 = k(1) * X * 2 + K(2) * Y * 2 + K(3) * 2
            //....
            //Z1 = k(1) * X * 1 + K(2) * Y * 1 + K(3) * 1
            //....
            for (int i = 0; i < 2; ++ i) {
                cv::Mat matXXt(1, ToInt32(vecIndex.size()), CV_32FC1);
                cv::Mat matXt(1, ToInt32(vecIndex.size()), CV_32FC1);
                cv::Mat matYt(1, ToInt32(vecIndex.size()), CV_32FC1);
                cv::Mat matYY(ToInt32(vecIndex.size()), 1, CV_32FC1);
                for (int j = 0; j < ToInt32(vecIndex.size()); ++ j) {
                    matXXt.at<DATA_TYPE>(j) = matXpT.at<DATA_TYPE>(vecIndex[j]);
                    matXt.at<DATA_TYPE>(j) = matXt0T.at<DATA_TYPE>(vecIndex[j]);
                    matYt.at<DATA_TYPE>(j) = matYt0T.at<DATA_TYPE>(vecIndex[j]);
                    matYY.at<DATA_TYPE>(j) = matHzT.at<DATA_TYPE>(vecIndex[j]);
                }
                matXX = cv::Mat(matXXt.mul(matXt));
                matXX.push_back(cv::Mat(matXXt.mul(matYt)));
                matXX.push_back(matXXt);
                cv::transpose(matXX, matXX);

                assert(matXX.rows == matYY.rows);
                cv::solve(matXX, matYY, matK, cv::DecompTypes::DECOMP_SVD);
#ifdef _DEBUG
                auto vecVecXX = CalcUtils::matToVector<DATA_TYPE> ( matXX );
                auto vecVecYY = CalcUtils::matToVector<DATA_TYPE> ( matYY );
                auto vecVecK  = CalcUtils::matToVector<DATA_TYPE> ( matK );
#endif
                //Remove the points with the largest fitting error, in the second pass, fit again with better data.
                if (i == 0) {
                    cv::Mat matErr = cv::abs(matYY - matXX * matK);
                    cv::Mat matErrSort;
                    cv::sortIdx(matErr, matErrSort, cv::SortFlags::SORT_ASCENDING + cv::SortFlags::SORT_EVERY_COLUMN);
                    std::vector<int> vecTmpIndex;
                    int nKeepPoints = ToInt32(std::floor(matErrSort.rows * CALIB_HEIGHT_STEP_USEFUL_PT));
                    vecTmpIndex.reserve(nKeepPoints);
                    for (int j = 0; j < nKeepPoints; ++ j)
                        vecTmpIndex.push_back(matErrSort.at<int>(j));
                    std::swap(vecTmpIndex, vecIndex);
                }
            }
            if (NN == n) {
                //Matlab code: Hz - (xp.*xt0*K(1) + xp.*yt0*K(2) + xp*K(3)
                cv::Mat matFitH = matXp.mul(matXt0) * matK.at<DATA_TYPE>(0) + matXp.mul(matYt0) * matK.at<DATA_TYPE>(1) + matXp * matK.at<DATA_TYPE>(2);
                cv::Mat matAllDiff = matHz - matFitH;
                pstRpy->vecVecStepPhaseDiff = CalcUtils::matToVector<DATA_TYPE>(matAllDiff);

                pstRpy->vecStepPhaseSlope.clear();
                for (int i = 0; i < ToInt32(vecPointToFectch.size()); ++ i) {
                    cv::Mat matXX = cv::Mat(matXp, cv::Rect(i, 0, 1, matHz.rows));
                    cv::Mat matYY = cv::Mat(matHz, cv::Rect(i, 0, 1, matHz.rows));
                    cv::Mat matSlope;

                    assert(matXX.rows == matYY.rows);
                    cv::solve(matXX, matYY, matSlope, cv::DecompTypes::DECOMP_SVD);
                    pstRpy->vecStepPhaseSlope.push_back(matSlope.at<DATA_TYPE>(0, 0));
                }
            }
        }

        if (pstCmd->bReverseHeight)
            pstRpy->matPhaseToHeightK = - matK;
        else
            pstRpy->matPhaseToHeightK = matK;

        cv::Mat matHeight;
        CalcUtils::meshgrid<DATA_TYPE>(1.f, 1.f, ToFloat(COLS), 1.f, 1.f, ToFloat(ROWS), matX, matY);
        cv::Mat matRatio = pstRpy->matPhaseToHeightK.at<DATA_TYPE>(0, 0) * matX + pstRpy->matPhaseToHeightK.at<DATA_TYPE>(1, 0) * matY + pstRpy->matPhaseToHeightK.at<DATA_TYPE>(2, 0);
        cv::divide(matPhase, matRatio, matTmpPhase);
    }
    VectorOfFloat vecGridHeights;
    pstRpy->matResultImg = _drawHeightGrid(matTmpPhase, pstCmd->nResultImgGridRow, pstCmd->nResultImgGridCol, pstCmd->szMeasureWinSize, vecGridHeights);
    pstRpy->enStatus = VisionStatus::OK;
}

/*static*/ cv::Mat Unwrap::_drawHeightGrid(const cv::Mat &matHeight, int nGridRow, int nGridCol, const cv::Size &szMeasureWinSize, VectorOfFloat &vecGridHeights) {
    vecGridHeights.clear();
    double dMinValue = 0, dMaxValue = 0;
    cv::Mat matMask = (matHeight == matHeight);
    cv::minMaxIdx(matHeight, &dMinValue, &dMaxValue, 0, 0, matMask);

    cv::Mat matNewPhase = matHeight - dMinValue;

    float dRatio = 255.f / ToFloat(dMaxValue - dMinValue);
    matNewPhase = matNewPhase * dRatio;

    cv::Mat matResultImg;
    matNewPhase.convertTo(matResultImg, CV_8UC1);
    cv::cvtColor(matResultImg, matResultImg, CV_GRAY2BGR);

    int ROWS = matNewPhase.rows;
    int COLS = matNewPhase.cols;
    int nIntervalX = matNewPhase.cols / nGridCol;
    int nIntervalY = matNewPhase.rows / nGridRow;

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    int thickness = 2;
    for (int j = 0; j < nGridRow; ++ j)
    for (int i = 0; i < nGridCol; ++ i) {
        cv::Rect rectROI(i * nIntervalX + nIntervalX / 2 - szMeasureWinSize.width / 2,
            j * nIntervalY + nIntervalY / 2 - szMeasureWinSize.height / 2,
            szMeasureWinSize.width, szMeasureWinSize.height);
        cv::rectangle(matResultImg, rectROI, cv::Scalar(0, 255, 0), 3);

        cv::Mat matROI(matHeight, rectROI);
        cv::Mat matMask = (matROI == matROI);
        float fAverage = ToFloat(cv::mean(matROI, matMask)[0]);
        vecGridHeights.push_back(fAverage);

        char strAverage[100];
        _snprintf(strAverage, sizeof(strAverage), "%.4f", fAverage);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(strAverage, fontFace, fontScale, thickness, &baseline);
        //The height use '+' because text origin start from left-bottom.
        cv::Point ptTextOrg(rectROI.x + (rectROI.width - textSize.width) / 2, rectROI.y + (rectROI.height + textSize.height) / 2);
        cv::putText(matResultImg, strAverage, ptTextOrg, fontFace, fontScale, cv::Scalar(255, 0, 0), thickness);
    }

    int nGridLineSize = 3;
    cv::Scalar scalarCyan(255, 255, 0);
    for (int i = 1; i < nGridCol; ++ i)
        cv::line(matResultImg, cv::Point(i * nIntervalX, 0), cv::Point(i * nIntervalX, ROWS), scalarCyan, nGridLineSize);
    for (int i = 1; i < nGridRow; ++ i)
        cv::line(matResultImg, cv::Point(0, i * nIntervalY), cv::Point(COLS, i * nIntervalY), scalarCyan, nGridLineSize);

    return matResultImg;
}

/*static*/ void Unwrap::calc3DHeightDiff(const PR_CALC_3D_HEIGHT_DIFF_CMD *const pstCmd, PR_CALC_3D_HEIGHT_DIFF_RPY *const pstRpy) {
    VectorOfRect vecRect = pstCmd->vecRectBases;
    vecRect.push_back(pstCmd->rectROI);
    cv::Rect rectBounding = CalcUtils::boundingRect(vecRect);
    cv::Mat matHeight(pstCmd->matHeight, rectBounding);
    cv::Mat matMask;
    if (!pstCmd->matMask.empty())
        matMask = cv::Mat(pstCmd->matMask, rectBounding);

    std::vector<float> vecXt, vecYt, vecHeightTmp;
    auto ROWS = matHeight.rows;
    auto COLS = matHeight.cols;
    cv::Mat matX, matY;
    CalcUtils::meshgrid<DATA_TYPE>(1.f, 1.f, ToFloat(COLS), 1.f, 1.f, ToFloat(ROWS), matX, matY);

    cv::Mat matNonNanMask = CalcUtils::getNonNanMask(matHeight);
    if (!matMask.empty()) {
        cv::Mat matMaskReverse = PR_MAX_GRAY_LEVEL - matMask;
        matNonNanMask.setTo(0, matMaskReverse);
    }

    for (auto rect : pstCmd->vecRectBases) {
        rect.x -= rectBounding.x;
        rect.y -= rectBounding.y;
        cv::Mat matTmpNonNanMask = matNonNanMask.clone();
        CalcUtils::setToExcept(matTmpNonNanMask, 0, rect);

        AOI::Vision::VectorOfPoint vecPtLocations;
        cv::findNonZero(matTmpNonNanMask, vecPtLocations);

        std::sort(vecPtLocations.begin(), vecPtLocations.end(), [&matHeight](const cv::Point &pt1, const cv::Point &pt2) {
            return matHeight.at<float>(pt1) < matHeight.at<float>(pt2);
        });
        AOI::Vision::VectorOfPoint vecTrimedLocations(vecPtLocations.begin() + ToInt32(vecPtLocations.size() * pstCmd->fEffectHRatioStart), vecPtLocations.begin() + ToInt32(vecPtLocations.size() * pstCmd->fEffectHRatioEnd));
        vecXt.reserve(vecXt.size() + vecTrimedLocations.size());
        vecYt.reserve(vecYt.size() + vecTrimedLocations.size());
        vecHeightTmp.reserve(vecHeightTmp.size() + vecTrimedLocations.size());
        for (const auto &point : vecTrimedLocations) {
            vecXt.push_back(matX.at<float>(point));
            vecYt.push_back(matY.at<float>(point));
            vecHeightTmp.push_back(matHeight.at<float>(point));
        }
    }

    if (vecXt.empty() || vecYt.empty()) {
        pstRpy->enStatus = VisionStatus::CALC_3D_HEIGHT_DIFF_NO_BASE_POINT;
        WriteLog("No base points to calculate 3D height difference, all base points are masked or are NAN.");
        return;
    }

    cv::Mat matK;
    {
        cv::Mat matXX;
        matXX.push_back(cv::Mat(vecXt).reshape(1, 1));
        matXX.push_back(cv::Mat(vecYt).reshape(1, 1));
        matXX.push_back(cv::Mat(cv::Mat::ones(1, ToInt32(vecHeightTmp.size()), CV_32FC1)));
        cv::transpose(matXX, matXX);
        cv::Mat matYY(vecHeightTmp);
        cv::solve(matXX, matYY, matK, cv::DecompTypes::DECOMP_QR);
    }

    float k1 = matK.at<float>(0);
    float k2 = matK.at<float>(1);
    float k3 = matK.at<float>(2);

    //Find the good points in ROI.
    if (!matMask.empty())
        matNonNanMask = CalcUtils::getNonNanMask(matHeight);
    auto rectROI = pstCmd->rectROI;
    rectROI.x -= rectBounding.x;
    rectROI.y -= rectBounding.y;
    CalcUtils::setToExcept(matNonNanMask, 0, rectROI);

    AOI::Vision::VectorOfPoint vecPtLocations;
    cv::findNonZero(matNonNanMask, vecPtLocations);

    std::sort(vecPtLocations.begin(), vecPtLocations.end(), [&matHeight](const cv::Point &pt1, const cv::Point &pt2) {
        return matHeight.at<float>(pt1) < matHeight.at<float>(pt2);
    });
    AOI::Vision::VectorOfPoint vecTrimedLocations(vecPtLocations.begin() + ToInt32(vecPtLocations.size() * pstCmd->fEffectHRatioStart), vecPtLocations.begin() + ToInt32(vecPtLocations.size() * pstCmd->fEffectHRatioEnd));

    vecXt.clear();
    vecYt.clear();
    vecHeightTmp.clear();
    for (const auto &point : vecTrimedLocations) {
        vecXt.push_back(matX.at<float>(point));
        vecYt.push_back(matY.at<float>(point));
        vecHeightTmp.push_back(matHeight.at<float>(point));
    }

    cv::Mat matXX;
    matXX.push_back(cv::Mat(vecXt).reshape(1, 1));
    matXX.push_back(cv::Mat(vecYt).reshape(1, 1));
    matXX.push_back(cv::Mat(cv::Mat::ones(1, ToInt32(vecHeightTmp.size()), CV_32FC1)));
    cv::transpose(matXX, matXX);
    cv::Mat matZZ(vecHeightTmp);

    matZZ = matZZ - matXX * matK;
    pstRpy->fHeightDiff = ToFloat(cv::mean(matZZ)[0]);
    pstRpy->enStatus = VisionStatus::OK;
}

/*static*/ void Unwrap::calcMTF(const PR_CALC_MTF_CMD *const pstCmd, PR_CALC_MTF_RPY *const pstRpy) {
    std::vector<cv::Mat> vecConvertedImgs;
    for (auto &mat : pstCmd->vecInputImgs) {
        cv::Mat matConvert = mat;
        if (mat.channels() > 1)
            cv::cvtColor(mat, matConvert, CV_BGR2GRAY);
        matConvert.convertTo(matConvert, CV_32FC1);
        vecConvertedImgs.push_back(matConvert);
    }

    pstRpy->vecVecAbsMtfH.clear();
    pstRpy->vecVecRelMtfH.clear();
    pstRpy->vecVecAbsMtfV.clear();
    pstRpy->vecVecRelMtfV.clear();
    int TOTAL_GROUP = ToInt32(pstCmd->vecInputImgs.size()) / PR_GROUP_TEXTURE_IMG_COUNT;
    for (int nGroup = 0; nGroup < TOTAL_GROUP; ++ nGroup) {
        cv::Mat mat00 = vecConvertedImgs[nGroup * PR_GROUP_TEXTURE_IMG_COUNT + 0] - vecConvertedImgs[nGroup * PR_GROUP_TEXTURE_IMG_COUNT + 2];
        cv::Mat mat01 = vecConvertedImgs[nGroup * PR_GROUP_TEXTURE_IMG_COUNT + 3] - vecConvertedImgs[nGroup * PR_GROUP_TEXTURE_IMG_COUNT + 1];

        cv::Mat matPhase;
        cv::phase(mat00, mat01, matPhase);
        //The opencv cv::phase function return result is 0~2*PI, but matlab is -PI~PI, so here need to do a conversion.
        cv::Mat matOverPi;
        cv::compare(matPhase, cv::Scalar::all(CV_PI), matOverPi, cv::CmpTypes::CMP_GT);
        cv::subtract(matPhase, cv::Scalar::all(2.f * CV_PI), matPhase, matOverPi);

        matPhase = _phaseUnwrapSurface(matPhase);
        cv::Mat matSin = CalcUtils::sin<DATA_TYPE>(matPhase);
        cv::Mat matCos = CalcUtils::cos<DATA_TYPE>(matPhase);
        std::vector<float> vecMtfAbsH(matPhase.rows, 1.f);
        std::vector<float> vecMtfRelH(matPhase.rows, 1.f);
        for (int row = 0; row < matPhase.rows; ++ row) {
            cv::Rect rectROI(0, row, matPhase.cols, 1);
            cv::Mat matYY(mat00, rectROI);
            cv::Mat matYYT;
            cv::transpose(matYY, matYYT);
            cv::Mat matXX;
            matXX.push_back(cv::Mat(matSin, rectROI).clone());
            matXX.push_back(cv::Mat(matCos, rectROI).clone());
            matXX.push_back(cv::Mat(cv::Mat::ones(1, matPhase.cols, matPhase.type())));
            cv::transpose(matXX, matXX);
            cv::Mat matK;
            cv::solve(matXX, matYYT, matK, cv::DecompTypes::DECOMP_SVD);
            vecMtfAbsH[row] = std::sqrt(matK.at<DATA_TYPE>(0, 0) * matK.at<DATA_TYPE>(0, 0) + matK.at<DATA_TYPE>(1, 0) * matK.at<DATA_TYPE>(1, 0)) / pstCmd->fMagnitudeOfDLP;
            if (nGroup > 0)
                vecMtfRelH[row] = vecMtfAbsH[row] / pstRpy->vecVecAbsMtfH[0][row];
        }
        pstRpy->vecVecAbsMtfH.push_back(vecMtfAbsH);
        pstRpy->vecVecRelMtfH.push_back(vecMtfRelH);

        cv::transpose(matPhase, matPhase);
        cv::transpose(mat00, mat00);
        cv::transpose(matSin, matSin);
        cv::transpose(matCos, matCos);

        std::vector<float> vecMtfAbsV(matPhase.rows, 1.f);
        std::vector<float> vecMtfRelV(matPhase.rows, 1.f);
        for (int row = 0; row < matPhase.rows; ++ row) {
            cv::Rect rectROI(0, row, matPhase.cols, 1);
            cv::Mat matYY(mat00, rectROI);
            cv::Mat matYYT;
            cv::transpose(matYY, matYYT);
            cv::Mat matXX;
            matXX.push_back(cv::Mat(matSin, rectROI).clone());
            matXX.push_back(cv::Mat(matCos, rectROI).clone());
            matXX.push_back(cv::Mat(cv::Mat::ones(1, matPhase.cols, matPhase.type())));
            cv::transpose(matXX, matXX);
            cv::Mat matK;
            cv::solve(matXX, matYYT, matK, cv::DecompTypes::DECOMP_SVD);
            vecMtfAbsV[row] = std::sqrt(matK.at<DATA_TYPE>(0, 0) * matK.at<DATA_TYPE>(0, 0) + matK.at<DATA_TYPE>(1, 0) * matK.at<DATA_TYPE>(1, 0)) / pstCmd->fMagnitudeOfDLP;
            if (nGroup > 0)
                vecMtfRelV[row] = vecMtfAbsV[row] / pstRpy->vecVecAbsMtfV[0][row];
        }
        pstRpy->vecVecAbsMtfV.push_back(vecMtfAbsV);
        pstRpy->vecVecRelMtfV.push_back(vecMtfRelV);
    }
    pstRpy->enStatus = VisionStatus::OK;
}

/*static*/ void Unwrap::calcPD(const PR_CALC_PD_CMD *const pstCmd, PR_CALC_PD_RPY *const pstRpy) {
    std::vector<cv::Mat> vecConvertedImgs;
    //Discard the first 4 thick texture light.
    for (size_t index = 4; index < 12; ++ index) {
        cv::Mat matConvert = pstCmd->vecInputImgs[index];
        if (pstCmd->vecInputImgs[index].channels() > 1)
            cv::cvtColor(pstCmd->vecInputImgs[index], matConvert, CV_BGR2GRAY);
        matConvert.convertTo(matConvert, CV_32FC1);
        vecConvertedImgs.push_back(matConvert);
    }

    //Filtering can reduce residues at the expense of a loss of spatial resolution.
    for (auto &mat : vecConvertedImgs)
        cv::GaussianBlur(mat, mat, cv::Size(pstCmd->nGaussianFilterSize, pstCmd->nGaussianFilterSize), pstCmd->fGaussianFilterSigma, pstCmd->fGaussianFilterSigma, cv::BorderTypes::BORDER_REPLICATE);

    //if ( pstCmd->bReverseSeq ) {
    //    std::swap ( vecConvertedImgs[1], vecConvertedImgs[3] );
    //    std::swap ( vecConvertedImgs[5], vecConvertedImgs[7] );
    //}
    cv::Mat mat00 = vecConvertedImgs[2] - vecConvertedImgs[0];
    cv::Mat mat01 = vecConvertedImgs[3] - vecConvertedImgs[1];

    cv::Mat mat10 = vecConvertedImgs[6] - vecConvertedImgs[4];
    cv::Mat mat11 = vecConvertedImgs[7] - vecConvertedImgs[5];

    cv::Mat matAlpha, matBeta;
    cv::phase(-mat00, mat01, matAlpha);
    //The opencv cv::phase function return result is 0~2*PI, but matlab is -PI~PI, so here need to do a conversion.
    cv::Mat matOverPi;
    cv::compare(matAlpha, cv::Scalar::all(CV_PI), matOverPi, cv::CmpTypes::CMP_GT);
    cv::subtract(matAlpha, cv::Scalar::all(2.f * CV_PI), matAlpha, matOverPi);

    cv::phase(-mat10, mat11, matBeta);
    cv::compare(matBeta, cv::Scalar::all(CV_PI), matOverPi, cv::CmpTypes::CMP_GT);
    cv::subtract(matBeta, cv::Scalar::all(2.f * CV_PI), matBeta, matOverPi);

#ifdef _DEBUG
    auto vecVecAlpha = CalcUtils::matToVector<DATA_TYPE> ( matAlpha );
    auto vecVecBeta  = CalcUtils::matToVector<DATA_TYPE> ( matBeta );
#endif

    const auto ROWS = matAlpha.rows;
    const auto COLS = matAlpha.cols;
    const auto MIDDLE_GRAY_SCALE = 127;

    matAlpha = _phaseUnwrapSurface(matAlpha);
    matBeta = _phaseUnwrapSurface(matBeta);

    cv::Size szDlpPatternSize(pstCmd->szDlpPatternSize);
    szDlpPatternSize.width *= 2;    //The width is double sized when do projection.
    cv::Mat matX, matY, matU, matV;
    CalcUtils::meshgrid<DATA_TYPE>(1.f, 1.f, ToFloat(szDlpPatternSize.width), 1.f, 1.f, ToFloat(szDlpPatternSize.height), matX, matY);
    cv::Mat matDlpPattern0 = (pstCmd->fMagnitudeOfDLP / 2) * CalcUtils::cos<DATA_TYPE>((matY + matX) * (2 * CV_PI) / pstCmd->fDlpPixelCycle) + MIDDLE_GRAY_SCALE;
    matDlpPattern0.convertTo(pstRpy->matCaptureRegionImg, CV_8U);
    cv::cvtColor(pstRpy->matCaptureRegionImg, pstRpy->matCaptureRegionImg, CV_GRAY2BGR);
    cv::Mat matXX1 = (matAlpha - matBeta) / 4 / CV_PI * pstCmd->fDlpPixelCycle;   //Convert the unwrapped phase to DLP pattern phase.
    cv::Mat matYY1 = (matAlpha + matBeta) / 4 / CV_PI * pstCmd->fDlpPixelCycle;   //divide 4 actually means divide (2*2*PI), a cycle is 2*PI.
#ifdef _DEBUG
    auto vecVecXX1 = CalcUtils::matToVector<DATA_TYPE> ( matXX1 );
    auto vecVecYY1 = CalcUtils::matToVector<DATA_TYPE> ( matYY1 );
#endif
    CalcUtils::meshgrid<DATA_TYPE>(0.f, 1.f / (COLS - 1), 1.f, 0.f, 1.f / (ROWS - 1), 1.f, matU, matV);
    matU = matU.reshape(1, 1);
    matV = matV.reshape(1, 1);

    cv::Mat matP1 = calcOrder5BezierCoeff(matU);
    cv::Mat matP2 = calcOrder5BezierCoeff(matV);
    cv::Mat matXX;
    for (int i = 0; i < BEZIER_RANK; ++ i)
    for (int j = 0; j < BEZIER_RANK; ++ j) {
        matXX.push_back(cv::Mat(matP1.row(i).mul(matP2.row(j))));
    }
    cv::transpose(matXX, matXX);

    matXX1 = matXX1.reshape(1, ToInt32(matXX1.total()));
    matYY1 = matYY1.reshape(1, ToInt32(matYY1.total()));
    cv::Mat matPPx, matPPy;
    cv::solve(matXX, matXX1, matPPx, cv::DecompTypes::DECOMP_SVD);
    cv::solve(matXX, matYY1, matPPy, cv::DecompTypes::DECOMP_SVD);

    float fPPxMean = ToFloat(cv::mean(matPPx)[0]);
    float fPPyMean = ToFloat(cv::mean(matPPy)[0]);
    cv::Mat matPPx1 = matPPx - floor((fPPxMean - szDlpPatternSize.width / 2) / pstCmd->fDlpPixelCycle) * pstCmd->fDlpPixelCycle;
    cv::Mat matPPy1 = matPPy - floor((fPPyMean - szDlpPatternSize.height / 2) / pstCmd->fDlpPixelCycle) * pstCmd->fDlpPixelCycle;
#ifdef _DEBUG
    auto vecVecPPx1 = CalcUtils::matToVector<DATA_TYPE> ( matPPx1 );
    auto vecVecPPy1 = CalcUtils::matToVector<DATA_TYPE> ( matPPy1 );
#endif

    const int SMALL_WIN_SIZE = 10;
    for (int i = 0; i < matPPx1.rows; ++ i) {
        cv::Point ptCtr(ToInt32(matPPx1.at<DATA_TYPE>(i)), ToInt32(matPPy1.at<DATA_TYPE>(i)));
        cv::Rect rect(ptCtr.x - SMALL_WIN_SIZE / 2, ptCtr.y - SMALL_WIN_SIZE / 2, SMALL_WIN_SIZE, SMALL_WIN_SIZE);
        cv::rectangle(pstRpy->matCaptureRegionImg, rect, cv::Scalar(0, 255, 0), CV_FILLED, CV_AA);
    }

    cv::Mat matXXt1 = matXX * matPPx1;
    cv::Mat matYYt1 = matXX * matPPy1;

    matXXt1 = matXXt1.reshape(1, ROWS);
    matYYt1 = matYYt1.reshape(1, ROWS);
#ifdef _DEBUG
    auto vecVecXXt1 = CalcUtils::matToVector<DATA_TYPE> ( matXXt1 );
    auto vecVecYYt1 = CalcUtils::matToVector<DATA_TYPE> ( matYYt1 );
#endif
    const int nInterval = 100;
    for (int row = 0; row < ROWS; row += nInterval)
    for (int col = 0; col < COLS; col += nInterval) {
        cv::Point ptCtr(ToInt32(matXXt1.at<DATA_TYPE>(row, col)), ToInt32(matYYt1.at<DATA_TYPE>(row, col)));
        cv::circle(pstRpy->matCaptureRegionImg, ptCtr, 5, cv::Scalar(255, 0, 0), CV_FILLED, CV_AA);
    }

    cv::Mat matError;
    bool bSwap = false;
    {
        cv::Mat matXt(matXXt1, cv::Rect(0, 0, 1, ROWS));
        cv::Mat matYt(matYYt1, cv::Rect(0, 0, 1, ROWS));
#ifdef _DEBUG
        auto vecVecXt = CalcUtils::matToVector<DATA_TYPE> ( matXt );
        auto vecVecYt = CalcUtils::matToVector<DATA_TYPE> ( matYt );
#endif
        if (fabs(matYt.at<DATA_TYPE>(0) - matYt.at<DATA_TYPE>(ROWS - 1)) > fabs(matXt.at<DATA_TYPE>(0) - matXt.at<DATA_TYPE>(ROWS - 1)))
            bSwap = true;
        if (bSwap)
            std::swap(matXt, matYt);

        matError = (matYt - matYt.at<DATA_TYPE>(0) - (matYt.at<DATA_TYPE>(ROWS - 1) - matYt.at<DATA_TYPE>(0)) / (matXt.at<DATA_TYPE>(ROWS - 1) - matXt.at<DATA_TYPE>(0))*(matXt - matXt.at<DATA_TYPE>(0)))*(ROWS - 1) / (matXt.at<DATA_TYPE>(ROWS - 1) - matXt.at<DATA_TYPE>(0));
        matError = matError.reshape(1, 1);
        pstRpy->vecDistortionLeft = CalcUtils::matToVector<DATA_TYPE>(matError)[0];
    }
    {
        cv::Mat matXt(matXXt1, cv::Rect(COLS - 1, 0, 1, ROWS));
        cv::Mat matYt(matYYt1, cv::Rect(COLS - 1, 0, 1, ROWS));
        if (bSwap)
            std::swap(matXt, matYt);
        matError = (matYt - matYt.at<DATA_TYPE>(0) - (matYt.at<DATA_TYPE>(ROWS - 1) - matYt.at<DATA_TYPE>(0)) / (matXt.at<DATA_TYPE>(ROWS - 1) - matXt.at<DATA_TYPE>(0))*(matXt - matXt.at<DATA_TYPE>(1)))*(ROWS - 1) / (matXt.at<DATA_TYPE>(ROWS - 1) - matXt.at<DATA_TYPE>(0));
        matError = matError.reshape(1, 1);
        pstRpy->vecDistortionRight = CalcUtils::matToVector<DATA_TYPE>(matError)[0];
    }
    {
        cv::Mat matYt(matXXt1, cv::Rect(0, 0, COLS, 1));
        cv::Mat matXt(matYYt1, cv::Rect(0, 0, COLS, 1));
#ifdef _DEBUG
        auto vecVecXt = CalcUtils::matToVector<DATA_TYPE> ( matXt );
        auto vecVecYt = CalcUtils::matToVector<DATA_TYPE> ( matYt );
#endif
        if (bSwap)
            std::swap(matXt, matYt);
        matError = (matYt - matYt.at<DATA_TYPE>(0) - (matYt.at<DATA_TYPE>(COLS - 1) - matYt.at<DATA_TYPE>(0)) / (matXt.at<DATA_TYPE>(COLS - 1) - matXt.at<DATA_TYPE>(0))*(matXt - matXt.at<DATA_TYPE>(0)))*(COLS - 1) / (matXt.at<DATA_TYPE>(COLS - 1) - matXt.at<DATA_TYPE>(0));
        pstRpy->vecDistortionTop = CalcUtils::matToVector<DATA_TYPE>(matError)[0];
    }
    {
        cv::Mat matYt(matXXt1, cv::Rect(0, ROWS - 1, COLS, 1));
        cv::Mat matXt(matYYt1, cv::Rect(0, ROWS - 1, COLS, 1));
        if (bSwap)
            std::swap(matXt, matYt);
        matError = (matYt - matYt.at<DATA_TYPE>(0) - (matYt.at<DATA_TYPE>(COLS - 1) - matYt.at<DATA_TYPE>(0)) / (matXt.at<DATA_TYPE>(COLS - 1) - matXt.at<DATA_TYPE>(0))*(matXt - matXt.at<DATA_TYPE>(0)))*(COLS - 1) / (matXt.at<DATA_TYPE>(COLS - 1) - matXt.at<DATA_TYPE>(0));
        pstRpy->vecDistortionBottom = CalcUtils::matToVector<DATA_TYPE>(matError)[0];
    }
    pstRpy->enStatus = VisionStatus::OK;
}

static inline cv::Mat prepareOrder3SurfaceFit(const cv::Mat &matX, const cv::Mat &matY) {
    assert(matX.cols == 1 && matY.cols == 1);
    cv::Mat matXPow2 = matX.mul(matX);
    cv::Mat matXPow3 = matXPow2.mul(matX);
    cv::Mat matYPow2 = matY.mul(matY);
    cv::Mat matYPow3 = matYPow2.mul(matY);
    cv::Mat matArray[] = {
        matXPow3,
        matYPow3,
        matXPow2.mul(matY),
        matYPow2.mul(matX),
        matXPow2,
        matYPow2,
        matX.mul(matY),
        matX,
        matY,
        cv::Mat::ones(matX.size(), matX.type()),
    };
    const int RANK_3_SURFACE_COEFF_COL = 10;
    static_assert (sizeof(matArray) / sizeof(cv::Mat) == RANK_3_SURFACE_COEFF_COL, "Size of array not correct");
    cv::Mat matResult;
    cv::hconcat(matArray, RANK_3_SURFACE_COEFF_COL, matResult);
    return matResult;
}

static inline cv::Mat calcOrder3Surface(const cv::Mat &matX, const cv::Mat &matY, const cv::Mat &matK) {
    cv::Mat matXPow2 = matX.mul(matX);
    cv::Mat matXPow3 = matXPow2.mul(matX);
    cv::Mat matYPow2 = matY.mul(matY);
    cv::Mat matYPow3 = matYPow2.mul(matY);
    cv::Mat matArray[] = {
        matXPow3 * matK.at<DATA_TYPE>(0),
        matYPow3 * matK.at<DATA_TYPE>(1),
        matXPow2.mul(matY) * matK.at<DATA_TYPE>(2),
        matYPow2.mul(matX) * matK.at<DATA_TYPE>(3),
        matXPow2 * matK.at<DATA_TYPE>(4),
        matYPow2 * matK.at<DATA_TYPE>(5),
        matX.mul(matY) * matK.at<DATA_TYPE>(6),
        matX * matK.at<DATA_TYPE>(7),
        matY * matK.at<DATA_TYPE>(8),
        cv::Mat::ones(matX.size(), matX.type()) * matK.at<DATA_TYPE>(9),
    };
    const int RANK_3_SURFACE_COEFF_COL = 10;
    static_assert (sizeof(matArray) / sizeof(cv::Mat) == RANK_3_SURFACE_COEFF_COL, "Size of array not correct");
    cv::Mat matResult = matArray[0];
    for (int i = 1; i < RANK_3_SURFACE_COEFF_COL; ++ i)
        matResult = matResult + matArray[i];
    return matResult;
}

/*static*/ cv::Mat Unwrap::_calcHeightFromPhase(const cv::Mat &matPhase, const cv::Mat &matHtt, const cv::Mat &matK) {
    CStopWatch stopWatch;
    cv::Mat matPhasePow2 = matPhase.mul(matPhase);
    cv::Mat matPhasePow3 = matPhasePow2.mul(matPhase);
    TimeLog::GetInstance()->addTimeLog("_calcHeightFromPhase 2 mul take: ", stopWatch.Span());
    cv::Mat matLeft = matPhase + matPhasePow2 * matK.at<DATA_TYPE>(10) + matPhasePow3 * matK.at<DATA_TYPE>(11);
    TimeLog::GetInstance()->addTimeLog("_calcHeightFromPhase add and muliply take: ", stopWatch.Span());
    cv::Mat matHeight;
    cv::divide(matLeft, matHtt, matHeight);
    TimeLog::GetInstance()->addTimeLog("cv::divide take: ", stopWatch.Span());
    return matHeight;
}

/*static*/ cv::Mat Unwrap::_calcHeightFromPhaseBuffer(const cv::Mat &matPhase, const cv::Mat &matHtt, const cv::Mat &matK, cv::Mat &matBuffer1, cv::Mat &matBuffer2) {
    CStopWatch stopWatch;
    matBuffer1 = matPhase.mul(matPhase);     //Pow 2
    TimeLog::GetInstance()->addTimeLog("_calcHeightFromPhase mul take: ", stopWatch.Span());
    auto K10 = matK.at<DATA_TYPE>(10);
    auto K11 = matK.at<DATA_TYPE>(11);

    if (K11 > 1e-5) {
        matBuffer2 = matBuffer1.mul(matPhase);   //Pow 3
        matBuffer1 = matPhase + matBuffer1 * K10 + matBuffer2 * K11;
    }
    else {
        matBuffer1 = matPhase + matBuffer1 * K10;
    }

    TimeLog::GetInstance()->addTimeLog("_calcHeightFromPhase add and muliply take: ", stopWatch.Span());
    matBuffer1 = matBuffer1 / matHtt;
    TimeLog::GetInstance()->addTimeLog("cv::divide take: ", stopWatch.Span());
    return matBuffer1;
}

/*static*/ void Unwrap::integrate3DCalib(const PR_INTEGRATE_3D_CALIB_CMD *const pstCmd, PR_INTEGRATE_3D_CALIB_RPY *const pstRpy) {
    cv::Mat matX, matY;
    CalcUtils::meshgrid<DATA_TYPE>(1.f, 1.f, ToFloat(pstCmd->vecCalibData[0].matPhase.cols), 1.f, 1.f, ToFloat(pstCmd->vecCalibData[0].matPhase.rows), matX, matY);
    std::vector<DATA_TYPE> vecXt, vecYt, vecPhase, vecBP;
    for (const auto &stCalibData : pstCmd->vecCalibData) {
        bool bPositivePhase = cv::sum(stCalibData.matDivideStepIndex)[0] > 0;
        cv::Mat matAbsDivideStepIndex;
        cv::Mat(cv::abs(stCalibData.matDivideStepIndex)).convertTo(matAbsDivideStepIndex, CV_8UC1);
        VectorOfPoint vecNonzeroPoints;
        cv::findNonZero(matAbsDivideStepIndex, vecNonzeroPoints);
        vecXt.reserve(vecXt.size() + vecNonzeroPoints.size());
        vecYt.reserve(vecYt.size() + vecNonzeroPoints.size());
        vecPhase.reserve(vecPhase.size() + vecNonzeroPoints.size());
        for (const auto &point : vecNonzeroPoints) {
            vecXt.push_back(matX.at<DATA_TYPE>(point));
            vecYt.push_back(matY.at<DATA_TYPE>(point));
            vecPhase.push_back(stCalibData.matPhase.at<DATA_TYPE>(point));
            if (bPositivePhase)
                vecBP.push_back(ToFloat(stCalibData.matDivideStepIndex.at<char>(point) - 1));
            else
                vecBP.push_back(ToFloat(stCalibData.matDivideStepIndex.at<char>(point) + 1));
        }
    }
    if (! pstCmd->matTopSurfacePhase.empty()) {
        cv::Mat matTopSurfacePhase = pstCmd->matTopSurfacePhase;
        cv::medianBlur(matTopSurfacePhase, matTopSurfacePhase, 5);
        cv::Mat matNonNan = matTopSurfacePhase == matTopSurfacePhase;
        VectorOfPoint vecEffectPoints;
        cv::findNonZero(matNonNan, vecEffectPoints);
        vecXt.reserve(vecXt.size() + vecEffectPoints.size());
        vecYt.reserve(vecYt.size() + vecEffectPoints.size());
        vecPhase.reserve(vecPhase.size() + vecEffectPoints.size());
        vecBP.reserve(vecBP.size() + vecEffectPoints.size());
        for (const auto &point : vecEffectPoints) {
            vecXt.push_back(matX.at<DATA_TYPE>(point));
            vecYt.push_back(matY.at<DATA_TYPE>(point));
            vecPhase.push_back(matTopSurfacePhase.at<DATA_TYPE>(point));
            vecBP.push_back(pstCmd->fTopSurfaceHeight);
        }
    }
    cv::Mat matXt(vecXt), matYt(vecYt), matPhase(vecPhase), matBP(vecBP);
    cv::Mat matXX = prepareOrder3SurfaceFit(matXt, matYt);
    cv::Mat matBPRepeat = cv::repeat(matBP, 1, matXX.cols);
    matXX = matXX.mul(matBPRepeat);
    cv::Mat matPhasePow2 = matPhase.mul(matPhase);
    cv::Mat matPhasePow3 = matPhasePow2.mul(matPhase);
    cv::Mat matArray[] = {
        matXX,
        - matPhasePow2,
        - matPhasePow3,
    };
    cv::hconcat(matArray, 3, matXX);
    cv::Mat matK;
    cv::solve(matXX, matPhase, matK, cv::DecompTypes::DECOMP_SVD);
#ifdef _DEBUG
    auto vecVecK = CalcUtils::matToVector<DATA_TYPE> ( matK );
#endif
    pstRpy->matOrder3CurveSurface = calcOrder3Surface(matX, matY, matK);
    for (const auto &stCalibData : pstCmd->vecCalibData) {
        cv::Mat matHeight = _calcHeightFromPhase(stCalibData.matPhase, pstRpy->matOrder3CurveSurface, matK);
        VectorOfFloat vecGridHeights;
        cv::Mat matHeightGridImg = _drawHeightGrid(matHeight, pstCmd->nResultImgGridRow, pstCmd->nResultImgGridCol, pstCmd->szMeasureWinSize, vecGridHeights);
        pstRpy->vecMatResultImg.push_back(matHeightGridImg);
    }
    if (! pstCmd->matTopSurfacePhase.empty()) {
        cv::Mat matHeight = _calcHeightFromPhase(pstCmd->matTopSurfacePhase, pstRpy->matOrder3CurveSurface, matK);
        VectorOfFloat vecGridHeights;
        cv::Mat matHeightGridImg = _drawHeightGrid(matHeight, pstCmd->nResultImgGridRow, pstCmd->nResultImgGridCol, pstCmd->szMeasureWinSize, vecGridHeights);
        pstRpy->vecMatResultImg.push_back(matHeightGridImg);
    }
    pstRpy->matIntegratedK = matK;
    pstRpy->enStatus = VisionStatus::OK;
}

/*static*/ void Unwrap::motorCalib3D(const PR_MOTOR_CALIB_3D_CMD *const pstCmd, PR_MOTOR_CALIB_3D_RPY *const pstRpy) {
    cv::Mat matX1, matY1, matX, matY, matXX, matXXt, matPhase, matZ2, matOnes, matSum1, matSum2;
    CalcUtils::meshgrid<DATA_TYPE>(1.f, 1.f, ToFloat(pstCmd->vecPairHeightPhase[0].second.cols), 1.f, 1.f, ToFloat(pstCmd->vecPairHeightPhase[0].second.rows), matX1, matY1);
    const int TOTAL = ToInt32(matX1.total());
    matX = matX1.reshape(1, TOTAL);
    matY = matY1.reshape(1, TOTAL);
    matOnes = cv::Mat::ones(TOTAL, 1, CV_32FC1);
    cv::Mat matXX1 = prepareOrder3SurfaceFit(matX, matY);
    for (const auto &pairHeightPhase : pstCmd->vecPairHeightPhase) {
        matPhase = pairHeightPhase.second.reshape(1, TOTAL);
        matZ2 = matOnes * pairHeightPhase.first;

        matZ2 = cv::repeat(matZ2, 1, matXX1.cols);
        matXX = matXX1.mul(matZ2);

        cv::Mat matPhasePow2 = matPhase.mul(matPhase);
        cv::Mat matPhasePow3 = matPhasePow2.mul(matPhase);
        cv::Mat matArray[] = {
            matXX,
            - matPhasePow2,
            - matPhasePow3,
        };
        cv::hconcat(matArray, 3, matXX);
        cv::transpose(matXX, matXXt);
        matSum1 = matSum1 + matXXt * matXX;
        matSum2 = matSum2 + matXXt * matPhase;
    }

    cv::Mat matK;
    cv::solve(matSum1, matSum2, matK, cv::DecompTypes::DECOMP_NORMAL);

#ifdef _DEBUG
    auto vecVecSum1 = CalcUtils::matToVector<DATA_TYPE> ( matSum1 );
    auto vecVecSum2 = CalcUtils::matToVector<DATA_TYPE> ( matSum2 );
    auto vecVecK = CalcUtils::matToVector<DATA_TYPE> ( matK );
#endif
    pstRpy->matIntegratedK = matK;
    pstRpy->matOrder3CurveSurface = calcOrder3Surface(matX1, matY1, matK);
    pstRpy->vecMatResultImg.clear();
    pstRpy->vecHeightCalibResult.clear();
    for (const auto &pairHeightPhase : pstCmd->vecPairHeightPhase) {
        cv::Mat matHeight = _calcHeightFromPhase(pairHeightPhase.second, pstRpy->matOrder3CurveSurface, matK);
        pstRpy->vecHeightCalibResult.push_back(PairHeightCalibResult(pairHeightPhase.first, matHeight));
        VectorOfFloat vecGridHeights;
        cv::Mat matHeightGridImg = _drawHeightGrid(matHeight, pstCmd->nResultImgGridRow, pstCmd->nResultImgGridCol, pstCmd->szMeasureWinSize, vecGridHeights);
        pstRpy->vecHeightGridHeights.push_back(PairHeightGridHeights(pairHeightPhase.first, vecGridHeights));
        pstRpy->vecMatResultImg.push_back(matHeightGridImg);
    }
    pstRpy->enStatus = VisionStatus::OK;
}

/*static*/ void Unwrap::phaseCorrection(cv::Mat &matPhase, const cv::Mat &matIdxNan, int nJumpSpanX, int nJumpSpanY) {
    CStopWatch stopWatch;
    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;

    //X direction
    if (nJumpSpanX > 0) {
        cv::Mat matPhaseDiff = CalcUtils::diff(matPhase, 1, 2);
#ifdef _DEBUG
        //CalcUtils::saveMatToCsv ( matPhase, "./data/HaoYu_20171114/test1/NewLens2/BeforePhaseCorrectionX.csv");
        auto vecVecPhase = CalcUtils::matToVector<float>(matPhase);
        auto vecVecPhaseDiff = CalcUtils::matToVector<float>(matPhaseDiff);
#endif
        cv::Mat matDiffSign = cv::Mat::zeros(ROWS, COLS, CV_8SC1);
        cv::Mat matDiffAmpl = cv::Mat::zeros(ROWS, COLS, CV_8SC1);
        std::vector<int> vecRowsWithJump;
        for (int row = 0; row < matPhaseDiff.rows; ++ row) {
            bool bRowWithPosJump = false, bRowWithNegJump = false;
            for (int col = 0; col < matPhaseDiff.cols; ++col) {
                auto value = matPhaseDiff.at<DATA_TYPE>(row, col);
                if (value > ONE_HALF_CYCLE) {
                    matDiffSign.at<char>(row, col) = 1;
                    bRowWithPosJump = true;

                    char nJumpAmplitude = static_cast<char> (std::ceil(std::abs(value) / 2.f) * 2);
                    matDiffAmpl.at<char>(row, col) = nJumpAmplitude;
                }
                else if (value < -ONE_HALF_CYCLE) {
                    matDiffSign.at<char>(row, col) = -1;
                    bRowWithNegJump = true;

                    char nJumpAmplitude = static_cast<char> (std::ceil(std::abs(value) / 2.f) * 2);
                    matDiffAmpl.at<char>(row, col) = nJumpAmplitude;
                }
            }
            if (bRowWithPosJump && bRowWithNegJump)
                vecRowsWithJump.push_back(row);
        }

        for (auto row : vecRowsWithJump) {
            char *ptrSignOfRow = matDiffSign.ptr<char>(row);
            char *ptrAmplOfRow = matDiffAmpl.ptr<char>(row);

            for (int kk = 0; kk < 2; ++ kk) {
                std::vector<int> vecJumpCol, vecJumpSpan, vecJumpIdxNeedToHandle;
                vecJumpCol.reserve(20);
                for (int col = 0; col < COLS; ++col) {
                    if (ptrSignOfRow[col] != 0)
                        vecJumpCol.push_back(col);
                }
                if (vecJumpCol.size() < 2)
                    continue;
                vecJumpSpan.reserve(vecJumpCol.size() - 1);
                for (size_t i = 1; i < vecJumpCol.size(); ++ i)
                    vecJumpSpan.push_back(vecJumpCol[i] - vecJumpCol[i - 1]);
                auto vecSortedJumpSpanIdx = CalcUtils::sort_index_value<int>(vecJumpSpan);
                for (int i = 0; i < ToInt32(vecJumpSpan.size()); ++i) {
                    if (vecJumpSpan[i] < nJumpSpanX)
                        vecJumpIdxNeedToHandle.push_back(i);
                }

                for (size_t jj = 0; jj < vecJumpIdxNeedToHandle.size(); ++ jj) {
                    auto nStart = vecJumpCol[vecSortedJumpSpanIdx[jj]];
                    auto nEnd = vecJumpCol[vecSortedJumpSpanIdx[jj] + 1];
                    char chSignFirst = ptrSignOfRow[nStart];        //The index is hard to understand. Use the sorted span index to find the original column.
                    char chSignSecond = ptrSignOfRow[nEnd];

                    char chAmplFirst = ptrAmplOfRow[nStart];
                    char chAmplSecond = ptrAmplOfRow[nEnd];
                    char chTurnAmpl = std::min(chAmplFirst, chAmplSecond) / 2;
                    if (chSignFirst * chSignSecond == -1) { //it is a pair
                        char chAmplNew = chAmplFirst - 2 * chTurnAmpl;
                        ptrAmplOfRow[nStart] = chAmplNew;
                        if (chAmplNew <= 0)
                            ptrSignOfRow[nStart] = 0;  // Remove the sign of jump flag.

                        chAmplNew = chAmplSecond - 2 * chTurnAmpl;
                        ptrAmplOfRow[nEnd] = chAmplNew;
                        if (chAmplNew <= 0)
                            ptrSignOfRow[nEnd] = 0;

                        auto startValue = matPhase.at<DATA_TYPE>(row, nStart);
                        for (int col = nStart + 1; col <= nEnd; ++ col) {
                            auto &value = matPhase.at<DATA_TYPE>(row, col);
                            value -= chSignFirst * ONE_CYCLE * chTurnAmpl;
                            if (chSignFirst > 0 && value < startValue) { //Jump up, need to roll down, but can not over roll
                                value = startValue;
                            }
                            else if (chSignFirst < 0 && value > startValue) { //Jump down, need to roll up
                                value = startValue;
                            }
                            else {
                                auto i = value; //Just for debug, no usage.
                            }
                        }
                    }
                }
            }
        }

        TimeLog::GetInstance()->addTimeLog("phaseCorrection in x direction.", stopWatch.Span());
    }

    // Y direction
    if (nJumpSpanY > 0) {
        cv::Mat matPhaseDiff = CalcUtils::diff(matPhase, 1, 1);
#ifdef _DEBUG
        //CalcUtils::saveMatToCsv ( matPhase, "./data/HaoYu_20171114/test5/NewLens2/BeforePhaseCorrectionY.csv");
        auto vecVecPhase = CalcUtils::matToVector<float>(matPhase);
        auto vecVecPhaseDiff = CalcUtils::matToVector<float>(matPhaseDiff);
#endif
        cv::Mat matDiffSign = cv::Mat::zeros(ROWS, COLS, CV_8SC1);
        cv::Mat matDiffAmpl = cv::Mat::zeros(ROWS, COLS, CV_8SC1);
        std::vector<int> vecColsWithJump;
        for (int col = 0; col < matPhaseDiff.cols; ++ col) {
            bool bColWithPosJump = false, bColWithNegJump = false;
            for (int row = 0; row < matPhaseDiff.rows; ++row) {
                auto value = matPhaseDiff.at<DATA_TYPE>(row, col);
                if (value > ONE_HALF_CYCLE) {
                    matDiffSign.at<char>(row, col) = 1;
                    bColWithPosJump = true;
                }
                else if (value < -ONE_HALF_CYCLE) {
                    matDiffSign.at<char>(row, col) = -1;
                    bColWithNegJump = true;
                }

                if (std::abs(value) > ONE_HALF_CYCLE) {
                    char nJumpAmplitude = static_cast<char> (std::ceil(std::abs(value) / 2.f) * 2);
                    matDiffAmpl.at<char>(row, col) = nJumpAmplitude;
                }
            }
            if (bColWithPosJump && bColWithNegJump)
                vecColsWithJump.push_back(col);
        }

        for (auto col : vecColsWithJump) {
            cv::Mat matSignOfCol = cv::Mat(matDiffSign, cv::Range::all(), cv::Range(col, col + 1)).clone();
            char *ptrSignOfCol = matSignOfCol.ptr<char>(0);
            cv::Mat matAmplOfCol = cv::Mat(matDiffAmpl, cv::Range::all(), cv::Range(col, col + 1)).clone();
            char *ptrAmplOfCol = matAmplOfCol.ptr<char>(0);

            for (int kk = 0; kk < 2; ++ kk) {
                std::vector<int> vecJumpRow, vecJumpSpan, vecJumpIdxNeedToHandle;
                vecJumpRow.reserve(20);
                for (int row = 0; row < ROWS; ++row) {
                    if (ptrSignOfCol[row] != 0)
                        vecJumpRow.push_back(row);
                }
                if (vecJumpRow.size() < 2)
                    continue;
                vecJumpSpan.reserve(vecJumpRow.size() - 1);
                for (size_t i = 1; i < vecJumpRow.size(); ++ i)
                    vecJumpSpan.push_back(vecJumpRow[i] - vecJumpRow[i - 1]);
                auto vecSortedJumpSpanIdx = CalcUtils::sort_index_value<int>(vecJumpSpan);
                for (int i = 0; i < ToInt32(vecJumpSpan.size()); ++i) {
                    if (vecJumpSpan[i] < nJumpSpanY)
                        vecJumpIdxNeedToHandle.push_back(i);
                }

                for (size_t jj = 0; jj < vecJumpIdxNeedToHandle.size(); ++ jj) {
                    auto nStart = vecJumpRow[vecSortedJumpSpanIdx[jj]];
                    auto nEnd = vecJumpRow[vecSortedJumpSpanIdx[jj] + 1];
                    char chSignFirst = ptrSignOfCol[nStart];        //The index is hard to understand. Use the sorted span index to find the original column.
                    char chSignSecond = ptrSignOfCol[nEnd];

                    char chAmplFirst = ptrAmplOfCol[nStart];
                    char chAmplSecond = ptrAmplOfCol[nEnd];
                    char chTurnAmpl = std::min(chAmplFirst, chAmplSecond) / 2;
                    if (chSignFirst * chSignSecond == -1) { //it is a pair
                        char chAmplNew = chAmplFirst - 2 * chTurnAmpl;
                        ptrAmplOfCol[nStart] = chAmplNew;
                        if (chAmplNew <= 0)
                            ptrSignOfCol[nStart] = 0;  // Remove the sign of jump flag.

                        chAmplNew = chAmplSecond - 2 * chTurnAmpl;
                        ptrAmplOfCol[nEnd] = chAmplNew;
                        if (chAmplNew <= 0)
                            ptrSignOfCol[nEnd] = 0;

                        auto startValue = matPhase.at<DATA_TYPE>(nStart, col);
                        for (int row = nStart + 1; row <= nEnd; ++row) {
                            auto &value = matPhase.at<DATA_TYPE>(row, col);
                            value -= chSignFirst * ONE_CYCLE * chTurnAmpl;
                            if (chSignFirst > 0 && value < startValue) { //Jump up, need to roll down, but can not over roll
                                value = startValue;
                            }
                            else if (chSignFirst < 0 && value > startValue) { //Jump down, need to roll up
                                value = startValue;
                            }
                            else {
                                auto i = value; //Just for debug, no usage.
                            }
                        }
                    }
                }
            }
        }
        TimeLog::GetInstance()->addTimeLog("phaseCorrection in y direction.", stopWatch.Span());
    }
}

void Unwrap::_turnPhase(cv::Mat &matPhase, cv::Mat &matPhaseDiff, char *ptrSignOfRow, char *ptrAmplOfRow, int row, int nStart, int nEnd) {
    char chSignFirst = ptrSignOfRow[nStart];
    char chAmplFirst = ptrAmplOfRow[nStart], chAmplSecond = ptrAmplOfRow[nEnd]; //Map to matlab code n3i, n4i.
    char chTurnAmpl = std::min(chAmplFirst, chAmplSecond) / 2;  //Map to matlab code n5i.    

    float dp1 = matPhaseDiff.at<DATA_TYPE>(row, nStart);
    float dp2 = matPhaseDiff.at<DATA_TYPE>(row, nEnd);

    matPhaseDiff.at<DATA_TYPE>(row, nStart) = dp1 - ptrSignOfRow[nStart] * ONE_CYCLE * chTurnAmpl; // clear trigger condition.
    matPhaseDiff.at<DATA_TYPE>(row, nEnd)   = dp2 - ptrSignOfRow[nEnd]   * ONE_CYCLE * chTurnAmpl; // clear trigger condition.

    if (fabs(dp1) >= (ONE_CYCLE - PHASE_TURN_THRESHOLD_1)) {
        char chAmplNew = chAmplFirst - 2 * chTurnAmpl;
        ptrAmplOfRow[nStart] = chAmplNew;
        if (chAmplNew <= 0)
            ptrSignOfRow[nStart] = 0;  // Remove the sign of jump flag.
    }else {
        ptrSignOfRow[nStart] = dp1 >= 0 ? 1 : - 1;
        ptrSignOfRow[nStart] = 2;
    }

    if (fabs(dp2) >= (ONE_CYCLE - PHASE_TURN_THRESHOLD_1)) {
        char chAmplNew = chAmplSecond - 2 * chTurnAmpl;
        ptrAmplOfRow[nEnd] = chAmplNew;
        if (chAmplNew <= 0)
            ptrSignOfRow[nEnd] = 0;
    }else {
        ptrSignOfRow[nEnd] = dp2 >= 0 ? 1 : - 1;
        ptrSignOfRow[nEnd] = 2;
    }

    for (int col = nStart + 1; col <= nEnd; ++ col)
        matPhase.at<DATA_TYPE>(row, col) -= chSignFirst * ONE_CYCLE * chTurnAmpl;
}

/*static*/ void Unwrap::_turnPhaseConditionOne(cv::Mat &matPhase, cv::Mat &matPhaseDiff, cv::Mat &matDiffSign, cv::Mat &matDiffAmpl, int row, int nMaxSpan) {
    bool bRowWithPosJump = false, bRowWithNegJump = false;
    for (int col = 0; col < matPhaseDiff.cols; ++col) {
        auto value = matPhaseDiff.at<DATA_TYPE>(row, col);
        if (value > PHASE_TURN_THRESHOLD_1) {
            matDiffSign.at<char>(row, col) = 1;
            bRowWithPosJump = true;

            char nJumpAmplitude = static_cast<char> (std::ceil((std::abs(value) - PHASE_TURN_THRESHOLD_1) / 2.f) * 2);
            matDiffAmpl.at<char>(row, col) = nJumpAmplitude;
        }
        else if (value < -PHASE_TURN_THRESHOLD_1) {
            matDiffSign.at<char>(row, col) = -1;
            bRowWithNegJump = true;

            char nJumpAmplitude = static_cast<char> (std::ceil((std::abs(value) - PHASE_TURN_THRESHOLD_1) / 2.f) * 2);
            matDiffAmpl.at<char>(row, col) = nJumpAmplitude;
        }
    }

    if (!bRowWithPosJump || !bRowWithNegJump)
        return;

    char *ptrSignOfRow = matDiffSign.ptr<char>(row);
    char *ptrAmplOfRow = matDiffAmpl.ptr<char>(row);
    auto ptrPhaseDiffOfRow = matPhaseDiff.ptr<DATA_TYPE>(row);

    for (int kk = 0; kk < 2; ++ kk) {
        std::vector<int> vecJumpCol, vecJumpSpan, vecJumpIdxNeedToHandle;
        vecJumpCol.reserve(20);
        for (int col = 0; col < matPhase.cols; ++col) {
            if (ptrSignOfRow[col] != 0)
                vecJumpCol.push_back(col);
        }
        if (vecJumpCol.size() < 2)
            continue;
        vecJumpSpan.reserve(vecJumpCol.size() - 1);
        for (size_t i = 1; i < vecJumpCol.size(); ++i)
            vecJumpSpan.push_back(vecJumpCol[i] - vecJumpCol[i - 1]);
        auto vecSortedJumpSpanIdx = CalcUtils::sort_index_value<int>(vecJumpSpan);
        for (int i = 0; i < ToInt32(vecJumpSpan.size()); ++i) {
            if (vecJumpSpan[i] < nMaxSpan)
                vecJumpIdxNeedToHandle.push_back(i);
        }

        for (size_t jj = 0; jj < vecJumpIdxNeedToHandle.size(); ++jj) {
            auto nStart = vecJumpCol[vecSortedJumpSpanIdx[jj]];
            auto nEnd = vecJumpCol[vecSortedJumpSpanIdx[jj] + 1];
            //The index is hard to understand. Use the sorted span index to find the original column.
            char chSignFirst = ptrSignOfRow[nStart], chSignSecond = ptrSignOfRow[nEnd];
            DATA_TYPE fStartPhase = ptrPhaseDiffOfRow[nStart], fEndPhase = ptrPhaseDiffOfRow[nEnd];
            if (chSignFirst * chSignSecond == -1 && fabs(fStartPhase - fEndPhase) > ONE_CYCLE) { //it is a pair
                if (row == 0) {
                    _turnPhase(matPhase, matPhaseDiff, ptrSignOfRow, ptrAmplOfRow, row, nStart, nEnd);
                }
                else {
                    char chAmplFirst = ptrAmplOfRow[nStart], chAmplSecond = ptrAmplOfRow[nEnd];
                    char chTurnAmpl = std::min(chAmplFirst, chAmplSecond) / 2;
                    float fOriginalDiff = 0.f, fTurnedDiff = 0.f;

                    for (int col = nStart + 1; col <= nEnd; ++col) {
                        auto fOriginalValue = matPhase.at<DATA_TYPE>(row, col);
                        auto fTurnedValue = fOriginalValue - chSignFirst * ONE_CYCLE * chTurnAmpl;
                        auto fLastRowValue = matPhase.at<DATA_TYPE>(row - 1, col);
                        fOriginalDiff += fabs(fOriginalValue - fLastRowValue);
                        fTurnedDiff += fabs(fTurnedValue - fLastRowValue);
                    }

                    if (fOriginalDiff > fTurnedDiff)
                        _turnPhase(matPhase, matPhaseDiff, ptrSignOfRow, ptrAmplOfRow, row, nStart, nEnd);
                }
            }
        }
    }
}

/*static*/ void Unwrap::phaseCorrectionEx(cv::Mat &matPhase, PR_DIRECTION enProjDir, PR_DIRECTION enScanDir, int nMinSpan, int nMaxSpan) {
    CStopWatch stopWatch;
    _phaseSwitch(matPhase, enProjDir, enScanDir);
    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;
    TimeLog::GetInstance()->addTimeLog("_phaseSwitch.", stopWatch.Span());

    cv::Mat matPhaseDiff = CalcUtils::diff(matPhase, 1, 2);
#ifdef _DEBUG
    //CalcUtils::saveMatToCsv ( matPhase, "./data/HaoYu_20171114/test1/NewLens2/BeforePhaseCorrectionX.csv");
    auto vecVecPhase = CalcUtils::matToVector<float>(matPhase);
    auto vecVecPhaseDiff = CalcUtils::matToVector<float>(matPhaseDiff);
#endif

    cv::Mat matDiffSign = cv::Mat::zeros(ROWS, COLS, CV_8SC1);
    cv::Mat matDiffAmpl = cv::Mat::zeros(ROWS, COLS, CV_8SC1);

    std::vector<std::vector<DATA_TYPE>> vecTurnedPhase;
    for (int row = 0; row < matPhaseDiff.rows; ++ row) {
        _turnPhaseConditionOne(matPhase, matPhaseDiff, matDiffSign, matDiffAmpl, row, nMaxSpan);
        if(row == 0)
            continue;

        std::vector<int> vecIndex1, vecIndex2, vecIndex3;
        auto ptrPhaseDiffOfRow = matPhaseDiff.ptr<DATA_TYPE>(row);
        for (int col = 0; col < COLS; ++ col) {
            auto phaseDiff = ptrPhaseDiffOfRow[col];
            if (fabs(phaseDiff) > 2 - PHASE_TURN_THRESHOLD_1)
                vecIndex1.push_back(col);

            if (phaseDiff > PHASE_TURN_THRESHOLD_2 && phaseDiff < 2.f - PHASE_TURN_THRESHOLD_1)
                vecIndex2.push_back(col);
            if (phaseDiff < -PHASE_TURN_THRESHOLD_2 && phaseDiff > -(2.f - PHASE_TURN_THRESHOLD_1))
                vecIndex3.push_back(col);
        }

        for (auto col : vecIndex1) {
            auto phaseDiff = ptrPhaseDiffOfRow[col];
            std::vector<int> vecIndexToCheck;
            char sign1;
            if (phaseDiff > 0) {
                vecIndexToCheck = vecIndex3;
                sign1 = 1;
            }else {
                vecIndexToCheck = vecIndex2;
                sign1 = -1;
            }

            std::vector<int> vecMatchIndex;
            std::vector<DATA_TYPE> vecMatchPhaseDiff;
            for (const auto i : vecIndexToCheck) {
                if (nMaxSpan > abs(col - i) && abs(col - i) > nMinSpan) {
                    vecMatchIndex.push_back(i);
                    vecMatchPhaseDiff.push_back(ptrPhaseDiffOfRow[i]);
                }
            }

            if (vecMatchIndex.empty())
                continue;

            // Process the largest phase difference first, the sort default is ascent, need to do a reverse to make it descent.
            auto vecSortedIdx = CalcUtils::sort_index_value(vecMatchPhaseDiff);
            std::reverse(vecSortedIdx.begin(), vecSortedIdx.end());
            std::reverse(vecMatchPhaseDiff.begin(), vecMatchPhaseDiff.end());
            if (vecSortedIdx.size() > 3) {
                vecSortedIdx.resize(3);
                vecMatchPhaseDiff.resize(3);
            }

            for (const auto index : vecSortedIdx) {
                int matchIndex = vecMatchIndex[index];
                char sign2;
                if (matchIndex < col) {
                    std::swap(matchIndex, col);
                    sign2 = -1;
                }else 
                    sign2 = 1;

                float fOriginalDiff = 0.f, fTurnedDiff = 0.f;

                for (int i = col + 1; i <= matchIndex; ++ i) {
                    auto fOriginalValue = matPhase.at<DATA_TYPE>(row, i);
                    auto fTurnedValue = fOriginalValue - ONE_CYCLE * sign1 * sign2;
                    auto fLastRowValue = matPhase.at<DATA_TYPE>(row - 1, i);
                    fOriginalDiff += fabs(fOriginalValue - fLastRowValue);
                    fTurnedDiff += fabs(fTurnedValue - fLastRowValue);
                }

                if (fOriginalDiff > fTurnedDiff) {
                    for (int i = col + 1; i <= matchIndex; ++ i)
                        matPhase.at<DATA_TYPE>(row, i) -= ONE_CYCLE * sign1 * sign2;

                    vecIndexToCheck.erase(std::remove(vecIndexToCheck.begin(), vecIndexToCheck.end(), vecMatchIndex[index]), vecIndexToCheck.end());
                    if (sign1 == 1)
                        vecIndex3 = vecIndexToCheck;
                    else
                        vecIndex2 = vecIndexToCheck;
                }
            }
        }
    }
    TimeLog::GetInstance()->addTimeLog("_phase correction new.", stopWatch.Span());

    _phaseSwitchBack(matPhase, enProjDir, enScanDir);
    TimeLog::GetInstance()->addTimeLog("_phaseSwitchBack.", stopWatch.Span());
}

/*static*/ cv::Mat Unwrap::_drawAutoThresholdResult(const cv::Mat &matInput, const VectorOfFloat &vecRanges) {
    cv::Mat matResult = cv::Mat::ones(matInput.size(), CV_8UC3);
    for (size_t i = 0; i < vecRanges.size() - 1 && i < MATLAB_COLOR_COUNT; ++ i) {
        cv::Mat matRangeMask;
        cv::inRange(matInput, vecRanges[i], vecRanges[i + 1], matRangeMask);
        matResult.setTo(MATLAB_COLOR_ORDER[i], matRangeMask);
    }
    return matResult;
}

void removeBlob(cv::Mat &matThreshold, cv::Mat &matBlob, float fAreaLimit) {
    VectorOfVectorOfPoint contours, effectiveContours;
    cv::findContours(matThreshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (const auto &contour : contours) {
        auto area = cv::contourArea(contour);
        if (area > fAreaLimit) {
            effectiveContours.push_back(contour);
        }
    }
    if (effectiveContours.size() > 0)
        cv::fillPoly(matBlob, effectiveContours, cv::Scalar::all(0));
}

/*static*/ void Unwrap::removeJumpArea(cv::Mat &matHeight, float fAreaLimit) {
    int nRepeatTimes = 0;
    cv::Mat matNormHeight, matThresholdNeg, matBlobNeg, matThresholdPos, matBlobPos, matBlob;

    do {
        double dLowerThreshold = 60;
        if (nRepeatTimes >= 2)
            dLowerThreshold = 10;
        cv::normalize(matHeight, matNormHeight, 0, 255, cv::NormTypes::NORM_MINMAX, CV_8UC1);
        cv::threshold(matNormHeight, matThresholdNeg, dLowerThreshold, 255, cv::ThresholdTypes::THRESH_BINARY_INV);
        cv::threshold(matNormHeight, matThresholdPos, 200, 255, cv::ThresholdTypes::THRESH_BINARY);
        cv::imwrite("./data/HaoYu_20171114/test5/Threhold" + std::to_string(nRepeatTimes) + "_1.png", matThresholdNeg);
        cv::imwrite("./data/HaoYu_20171114/test5/Threhold" + std::to_string(nRepeatTimes) + "_2.png", matThresholdPos);

        matBlobNeg = matThresholdNeg.clone();
        removeBlob(matThresholdNeg, matBlobNeg, fAreaLimit);
        cv::imwrite("./data/HaoYu_20171114/test5/Threhold" + std::to_string(nRepeatTimes) + "_3.png", matBlobNeg);

        matBlobPos = matThresholdPos.clone();
        removeBlob(matThresholdPos, matBlobPos, fAreaLimit);
        cv::imwrite("./data/HaoYu_20171114/test5/Threhold" + std::to_string(nRepeatTimes) + "_4.png", matBlobPos);

        matBlob = matBlobNeg | matBlobPos;
        cv::dilate(matBlob, matBlob, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
        cv::imwrite("./data/HaoYu_20171114/test5/Threhold" + std::to_string(nRepeatTimes) + "_5.png", matBlob);
        VectorOfPoint vecNanPoints;
        cv::findNonZero(matBlob, vecNanPoints);
        std::cout << "Remove neg noise points " << vecNanPoints.size() << std::endl;
        for (const auto &point : vecNanPoints) {
            if (point.x > 0)
                matHeight.at<DATA_TYPE>(point) = matHeight.at<DATA_TYPE>(point.y, point.x - 1);
            else if (point.x == 0 && point.y > 0)
                matHeight.at<DATA_TYPE>(point) = matHeight.at<DATA_TYPE>(point.y - 1, point.x);
        }
        ++ nRepeatTimes;
    } while (nRepeatTimes < 6);
}

/*static*/ void Unwrap::_fitHoleInNanMask(cv::Mat &matMask, const cv::Size &szMorphKernel, Int16 nMorphIteration) {
    cv::Mat matKernal = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, szMorphKernel);
    cv::MorphTypes morphType = cv::MORPH_CLOSE;
    cv::morphologyEx(matMask, matMask, morphType, matKernal, cv::Point(-1, -1), nMorphIteration);
}

/*static*/ void Unwrap::_phasePatch(cv::Mat &matPhase, PR_DIRECTION enProjDir, const cv::Mat &matNanMask) {
    VectorOfPoint vecNanPoints;
    cv::findNonZero(matNanMask, vecNanPoints);
    if (vecNanPoints.empty())
        return;

    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;

    if (vecNanPoints.front() == cv::Point(0,0)) {
        int ii = 1;
        while (ii < ToInt32(vecNanPoints.size()) && vecNanPoints[ii] == cv::Point(ii, 0))
            ++ ii;
        if (ii < COLS)
            matPhase.at<DATA_TYPE>(0, 0) = matPhase.at<DATA_TYPE>(0, ii);
    }

    if (vecNanPoints.back() == cv::Point(COLS - 1, ROWS - 1)) {
        int ii = ToInt32(vecNanPoints.size() - 2);
        while (ii >= 0 && vecNanPoints[ii] == cv::Point(COLS - (ToInt32(vecNanPoints.size()) - ii), 0))
            -- ii;
        int nonNanCol = COLS - (ToInt32(vecNanPoints.size()) - ii);
        if (nonNanCol < COLS)
            matPhase.at<DATA_TYPE>(ROWS - 1, COLS - 1) = matPhase.at<DATA_TYPE>(ROWS - 1, nonNanCol);
    }

    // Phase Patch by direction
    // Projection up to down, patch down to up
    if (PR_DIRECTION::UP == enProjDir) {
        for (VectorOfPoint::reverse_iterator iter = vecNanPoints.rbegin(); iter != vecNanPoints.rend(); ++ iter) {
            auto point = *iter;
            if (point.y < ROWS - 1)
                matPhase.at<DATA_TYPE>(point) = matPhase.at<DATA_TYPE>(point.y + 1, point.x);
            else if (point.x < COLS - 1)
                matPhase.at<DATA_TYPE>(point) = matPhase.at<DATA_TYPE>(point.y, point.x + 1);
        }
    }
    // Projection down to up, patch up to down
    else if (PR_DIRECTION::DOWN == enProjDir) {
        for (const auto &point : vecNanPoints) {
            if (point.y > 0)
                matPhase.at<DATA_TYPE>(point) = matPhase.at<DATA_TYPE>(point.y - 1, point.x);
            else if (point.x > 0)
                matPhase.at<DATA_TYPE>(point) = matPhase.at<DATA_TYPE>(point.y, point.x - 1);
        }
    }
    // Projection left to right, patch right to left
    else if (PR_DIRECTION::LEFT == enProjDir) {
        for (VectorOfPoint::reverse_iterator iter = vecNanPoints.rbegin(); iter != vecNanPoints.rend(); ++ iter) {
            auto point = *iter;
            if (point.x < COLS - 1)
                matPhase.at<DATA_TYPE>(point) = matPhase.at<DATA_TYPE>(point.y, point.x + 1);
            else if (point.y < ROWS - 1)
                matPhase.at<DATA_TYPE>(point) = matPhase.at<DATA_TYPE>(point.y + 1, point.x);
        }
    }
    // Projection right to left, patch left to right
    else if (PR_DIRECTION::RIGHT == enProjDir) {
        for (const auto &point : vecNanPoints) {
            if (point.x > 0)
                matPhase.at<DATA_TYPE>(point) = matPhase.at<DATA_TYPE>(point.y, point.x - 1);
            else if (point.y > 0)
                matPhase.at<DATA_TYPE>(point) = matPhase.at<DATA_TYPE>(point.y - 1, point.x);
        }
    }else {
        throw std::runtime_error("Invalid project direction passed in.");
    }
}

/*static*/ void Unwrap::_phaseSwitch(cv::Mat &matPhase, PR_DIRECTION enProjDir, PR_DIRECTION enScanDir) {
    if (PR_DIRECTION::UP == enProjDir|| PR_DIRECTION::DOWN == enProjDir)
        cv::transpose(matPhase, matPhase);
    
    // Flip around y-axis
    if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::LEFT == enProjDir)
        cv::flip(matPhase, matPhase, 1);

    // Flip around x-axis
    if (PR_DIRECTION::DOWN == enScanDir || PR_DIRECTION::RIGHT == enScanDir)
        cv::flip(matPhase, matPhase, 0);
}

/*static*/ void Unwrap::_phaseSwitchBack(cv::Mat &matPhase, PR_DIRECTION enProjDir, PR_DIRECTION enScanDir) {
    if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::LEFT == enProjDir)
        cv::flip(matPhase, matPhase, 1);

    if (PR_DIRECTION::DOWN == enScanDir || PR_DIRECTION::RIGHT == enScanDir)
        cv::flip(matPhase, matPhase, 0);

    if (PR_DIRECTION::UP == enProjDir|| PR_DIRECTION::DOWN == enProjDir)
        cv::transpose(matPhase, matPhase);
}

/*static*/ void Unwrap::_calcReverseAndProjDir(const VectorOfMat &vecMat, bool bEnableGaussianFilter, PR_CALIB_3D_BASE_RPY *const pstRpy) {
    assert(vecMat.size() == PR_GROUP_TEXTURE_IMG_COUNT);

    std::vector<cv::Mat> vecConvertedImgs;
    for (const auto &mat : vecMat) {
        cv::Mat matConvert = mat;
        if (mat.channels() > 1)
            cv::cvtColor(mat, matConvert, CV_BGR2GRAY);
        matConvert.convertTo(matConvert, CV_32FC1);
        vecConvertedImgs.push_back(matConvert);
    }

    if (bEnableGaussianFilter) {
        //Filtering can reduce residues at the expense of a loss of spatial resolution.
        for (int i = 0; i < PR_GROUP_TEXTURE_IMG_COUNT; ++ i)
            cv::GaussianBlur(vecConvertedImgs[i], vecConvertedImgs[i], cv::Size(GUASSIAN_FILTER_SIZE, GUASSIAN_FILTER_SIZE), GAUSSIAN_FILTER_SIGMA, GAUSSIAN_FILTER_SIGMA, cv::BorderTypes::BORDER_REPLICATE);
    }

    cv::Mat mat00 = vecConvertedImgs[2] - vecConvertedImgs[0];
    cv::Mat mat01 = vecConvertedImgs[3] - vecConvertedImgs[1];

    cv::Mat matAlpha;
    cv::phase(-mat00, mat01, matAlpha);

    matAlpha = matAlpha / CV_PI;

    matAlpha = _phaseUnwrapSurface(matAlpha);

    const auto ROWS = matAlpha.rows;
    const auto COLS = matAlpha.cols;

    auto A1 = cv::mean(cv::Mat(matAlpha, cv::Rect(0, 0, CHECK_RS_WIN, CHECK_RS_WIN)))[0];
    auto A2 = cv::mean(cv::Mat(matAlpha, cv::Rect(COLS - CHECK_RS_WIN, 0, CHECK_RS_WIN, CHECK_RS_WIN)))[0];
    auto A3 = cv::mean(cv::Mat(matAlpha, cv::Rect(0, ROWS - CHECK_RS_WIN, CHECK_RS_WIN, CHECK_RS_WIN)))[0];
    auto A4 = cv::mean(cv::Mat(matAlpha, cv::Rect(COLS - CHECK_RS_WIN, ROWS - CHECK_RS_WIN, CHECK_RS_WIN, CHECK_RS_WIN)))[0];

    auto H1 = A2 - A1;
    auto H2 = A4 - A3;
    auto V1 = A3 - A1;
    auto V2 = A4 - A2;

    bool ReverseSeq = false;

    if (fabs(H1 + H2) < fabs(V1 + V2)) {
        if (abs(V1) > abs(V2)) {
            pstRpy->enProjectDir = PR_DIRECTION::LEFT;
            if (H1 > 0)
                ReverseSeq = 0;
            else
                ReverseSeq = 1;
        }
        else {
            pstRpy->enProjectDir = PR_DIRECTION::RIGHT;
            if (H1 > 0)
                ReverseSeq = 1;
            else
                ReverseSeq = 0;
        }

        if (!ReverseSeq) {
            if (V1 > 0)
                pstRpy->enScanDir = PR_DIRECTION::UP;
            else
                pstRpy->enScanDir = PR_DIRECTION::DOWN;
        }
        else {
            if (V1 < 0)
                pstRpy->enScanDir = PR_DIRECTION::UP;
            else
                pstRpy->enScanDir = PR_DIRECTION::DOWN;
        }
    }
    else {
        if (fabs(H1) > fabs(H2)) {
            pstRpy->enProjectDir = PR_DIRECTION::UP;
            if (V1 > 0)
                ReverseSeq = 0;
            else
                ReverseSeq = 1;
        }
        else {
            pstRpy->enProjectDir =  PR_DIRECTION::DOWN;
            if (V1 > 0)
                ReverseSeq = 1;
            else
                ReverseSeq = 0;
        }

        if (!ReverseSeq) {
            if (H1 > 0)
                pstRpy->enScanDir = PR_DIRECTION::LEFT;
            else
                pstRpy->enScanDir = PR_DIRECTION::RIGHT;
        }else {
            if (H1 < 0)
                pstRpy->enScanDir = PR_DIRECTION::LEFT;
            else
                pstRpy->enScanDir = PR_DIRECTION::RIGHT;
        }
    }
    pstRpy->bReverseSeq = ReverseSeq;
}

void _saveAsGray(const cv::Mat &matHeight, const std::string &strFilePath) {
    PR_HEIGHT_TO_GRAY_CMD stCovertCmd;
    PR_HEIGHT_TO_GRAY_RPY stConvertRpy;
    stCovertCmd.matHeight = matHeight;
    VisionAlgorithm::heightToGray(&stCovertCmd, &stConvertRpy);
    cv::imwrite(strFilePath, stConvertRpy.matGray);
}

// Merge height based on H1, H2, (H1+H2)/2. If the difference between 2 height is big, the area choose the nearest one from three of them.
/*static*/ cv::Mat Unwrap::_mergeHeightIntersect(cv::Mat &matHeightOne, const cv::Mat &matNanMaskOne, cv::Mat &matHeightTwo, const cv::Mat &matNanMaskTwo, float fDiffThreshold, PR_DIRECTION enProjDir) {
    CStopWatch stopWatch;
    matHeightTwo.copyTo(matHeightOne, matNanMaskOne);
    matHeightOne.copyTo(matHeightTwo, matNanMaskTwo);

    if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::DOWN == enProjDir) {
        cv::transpose(matHeightOne, matHeightOne);
        cv::transpose(matHeightTwo, matHeightTwo);
    }

    cv::Mat matHeightTre = (matHeightOne + matHeightTwo) / 2.f; // Tre means three
    
    cv::Mat matAbsDiff, matBigDiffMask;
    cv::absdiff(matHeightOne, matHeightTwo, matAbsDiff);
    matBigDiffMask = matAbsDiff > fDiffThreshold;

    cv::Mat matHeightFor = matHeightTre.clone();    // For means four
    matHeightFor.setTo(NAN, matBigDiffMask);

#ifdef _DEBUG
    auto vecVecHeightOne = CalcUtils::matToVector<DATA_TYPE>(matHeightOne);
    auto vecVecHeightTwo = CalcUtils::matToVector<DATA_TYPE>(matHeightTwo);
    auto vecVecHeightTre = CalcUtils::matToVector<DATA_TYPE>(matHeightTre);
    auto vecVecHeightFor = CalcUtils::matToVector<DATA_TYPE>(matHeightFor);
#endif

    for (int row = 0; row < matHeightFor.rows; ++ row) {
        cv::Mat matRowOne = matHeightOne.row(row);
        cv::Mat matRowTwo = matHeightTwo.row(row);
        cv::Mat matRowTre = matHeightTre.row(row);
        cv::Mat matRowFor = matHeightFor.row(row);

        cv::Mat matNanMasks = CalcUtils::getNanMask(matRowFor);
        matNanMasks.convertTo(matNanMasks, CV_32FC1);
        cv::Mat matDiffAbs = cv::abs(CalcUtils::diff(matNanMasks, 1, CalcUtils::DIFF_ON_X_DIR));
        std::deque<int> dequeIndex;
        for (int col = 0; col < matDiffAbs.cols; ++ col)
            if (matDiffAbs.at<float>(col) > 0)
                dequeIndex.push_back(col);
        if (matNanMasks.at<DATA_TYPE>(0) > 0)
            dequeIndex.push_front(0);
        if (matNanMasks.at<DATA_TYPE>(matNanMasks.cols - 1) > 0)
            dequeIndex.push_back(matNanMasks.cols - 1);

        for (int i = 0; i < dequeIndex.size() / 2; ++ i) {
            int startIndex = dequeIndex[i*2];
            int endIndex = dequeIndex[i*2+1];
            VectorOfFloat vecTargetH;
            if (0 == startIndex && endIndex < matNanMasks.cols - 1)
                vecTargetH = std::vector<float>(endIndex - startIndex, matRowFor.at<DATA_TYPE>(endIndex + 1));
            else if (matNanMasks.cols - 1 == endIndex && startIndex >= 0)
                vecTargetH = std::vector<float>(endIndex - startIndex, matRowFor.at<DATA_TYPE>(startIndex));
            else if (startIndex >= 0 && endIndex < matNanMasks.cols - 1) {
                float fInterval = (matRowFor.at<DATA_TYPE>(endIndex + 1) - matRowFor.at<DATA_TYPE>(startIndex)) / (endIndex - startIndex + 1);
                for (int j = 0; j < endIndex - startIndex; ++ j) {
                    vecTargetH.push_back(matRowFor.at<DATA_TYPE>(startIndex) + fInterval);
                }
            }

            float fAbsDiffSumOne = 0.f, fAbsDiffSumTwo = 0.f, fAbsDiffSumTre = 0.f;
            for (int index = startIndex, k = 0; index < endIndex; ++ index, ++ k) {
                fAbsDiffSumOne += fabs(matRowOne.at<DATA_TYPE>(index) - vecTargetH[k]);
                fAbsDiffSumTwo += fabs(matRowTwo.at<DATA_TYPE>(index) - vecTargetH[k]);
                fAbsDiffSumTre += fabs(matRowTre.at<DATA_TYPE>(index) - vecTargetH[k]);
            }

            cv::Mat matChoose;
            if (fAbsDiffSumOne <= fAbsDiffSumTwo && fAbsDiffSumOne <= fAbsDiffSumTre)
                matChoose = matRowOne;
            else if (fAbsDiffSumTwo <= fAbsDiffSumOne && fAbsDiffSumTwo <= fAbsDiffSumTre)
                matChoose = matRowTwo;
            else
                matChoose = matRowTre;

            cv::Mat matTarget(matHeightFor, cv::Range(row, row + 1), cv::Range(startIndex, endIndex + 1));
            cv::Mat matSrc(matChoose, cv::Range(0, 1), cv::Range(startIndex, endIndex + 1));
            matSrc.copyTo(matTarget);
        }
    }

    if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::DOWN == enProjDir)
        cv::transpose(matHeightFor, matHeightFor);

    return matHeightFor;
}

/*static*/ cv::Mat Unwrap::_mergeHeightMax(cv::Mat &matHeightOne, const cv::Mat &matNanMaskOne, cv::Mat &matHeightTwo, const cv::Mat &matNanMaskTwo, float fDiffThreshold, PR_DIRECTION enProjDir) {
    CStopWatch stopWatch;
    cv::Mat matHeight = cv::Mat::zeros(matHeightOne.size(), matHeightOne.type());
    cv::Mat matNonNanMaskOne = PR_MAX_GRAY_LEVEL - matNanMaskOne;
    cv::Mat matNonNanMaskTwo = PR_MAX_GRAY_LEVEL - matNanMaskTwo;

    cv::Mat matAbsDiff, matIdxH0, matIdxH1, matIdxH2;
    cv::absdiff(matHeightOne, matHeightTwo, matAbsDiff);
    matIdxH0 = matAbsDiff < fDiffThreshold;
    matIdxH1 = (matHeightTwo - matHeightOne) > fDiffThreshold;
    matIdxH2 = (matHeightOne - matHeightTwo) > fDiffThreshold;

    matIdxH1 = (matIdxH1 | matNanMaskOne) & matNonNanMaskTwo;
    matIdxH2 = (matIdxH2 | matNanMaskTwo) & matNonNanMaskOne;
    
    matIdxH0 = matIdxH0 & matNonNanMaskOne & matNonNanMaskTwo;

    matHeightTwo.copyTo(matHeight, matIdxH1);
    matHeightOne.copyTo(matHeight, matIdxH2);    

    cv::add(matHeightOne / 2.f, matHeightTwo / 2.f, matHeight, matIdxH0);
    TimeLog::GetInstance()->addTimeLog("Merge height.", stopWatch.Span());

    cv::medianBlur(matHeight, matHeight, 5);
    TimeLog::GetInstance()->addTimeLog("medianBlur.", stopWatch.Span());
    return matHeight;
}

/*static*/ void Unwrap::insp3DSolder(const PR_INSP_3D_SOLDER_CMD *pstCmd, const cv::Mat &matBaseMask, PR_INSP_3D_SOLDER_RPY *const pstRpy) {
    cv::Mat matHeightROI = cv::Mat(pstCmd->matHeight, pstCmd->rectDeviceROI).clone();

    AOI::Vision::VectorOfPoint vecPtLocations;
    cv::findNonZero(matBaseMask, vecPtLocations);

    std::sort(vecPtLocations.begin(), vecPtLocations.end(), [&matHeightROI](const cv::Point &pt1, const cv::Point &pt2) {
        return matHeightROI.at<float>(pt1) < matHeightROI.at<float>(pt2);
    });

    std::vector<float> vecXt, vecYt, vecHeightTmp;
    auto ROWS = matHeightROI.rows;
    auto COLS = matHeightROI.cols;
    cv::Mat matX, matY;
    CalcUtils::meshgrid<DATA_TYPE>(1.f, 1.f, ToFloat(COLS), 1.f, 1.f, ToFloat(ROWS), matX, matY);

    const float fEffectHRatioStart = 0.15f, fEffectHRatioEnd = 0.9f;
    AOI::Vision::VectorOfPoint vecTrimedLocations(vecPtLocations.begin() + ToInt32(vecPtLocations.size() * fEffectHRatioStart), vecPtLocations.begin() + ToInt32(vecPtLocations.size() * fEffectHRatioEnd));
    vecXt.reserve(vecXt.size() + vecTrimedLocations.size());
    vecYt.reserve(vecYt.size() + vecTrimedLocations.size());
    vecHeightTmp.reserve(vecHeightTmp.size() + vecTrimedLocations.size());
    for (const auto &point : vecTrimedLocations) {
        vecXt.push_back(matX.at<float>(point));
        vecYt.push_back(matY.at<float>(point));
        vecHeightTmp.push_back(matHeightROI.at<float>(point));
    }

    if (vecXt.empty() || vecYt.empty()) {
        pstRpy->enStatus = VisionStatus::CALC_3D_HEIGHT_DIFF_NO_BASE_POINT;
        WriteLog("No base points to calculate 3D height difference, all base points are masked or are NAN.");
        return;
    }

    cv::Mat matK;
    {
        cv::Mat matXX;
        matXX.push_back(cv::Mat(vecXt).reshape(1, 1));
        matXX.push_back(cv::Mat(vecYt).reshape(1, 1));
        matXX.push_back(cv::Mat(cv::Mat::ones(1, ToInt32(vecHeightTmp.size()), CV_32FC1)));
        cv::transpose(matXX, matXX);
        cv::Mat matYY(vecHeightTmp);
        cv::solve(matXX, matYY, matK, cv::DecompTypes::DECOMP_QR);
    }

    float k1 = matK.at<float>(0);
    float k2 = matK.at<float>(1);
    float k3 = matK.at<float>(2);

    matHeightROI = matHeightROI - (matX * k1 + matY * k2 + k3);

    bool bIsVertialROI = CalcUtils::isVerticalROI(pstCmd->vecRectCheckROIs);
    auto vecRectCheckROIs = pstCmd->vecRectCheckROIs;
    for (auto &rectCheckROI : vecRectCheckROIs) {
        rectCheckROI.x -= pstCmd->rectDeviceROI.x;
        rectCheckROI.y -= pstCmd->rectDeviceROI.y;
    }

    if (bIsVertialROI) {
        cv::transpose(matHeightROI, matHeightROI);
        cv::flip(matHeightROI, matHeightROI, 1); //transpose+flip(1)=CW
        for (auto &rectCheckROI : vecRectCheckROIs)
            rectCheckROI = cv::Rect(rectCheckROI.y, rectCheckROI.x, rectCheckROI.height, rectCheckROI.width);
    }

    const float COVERAGE = 0.75f;
    for (const auto &rectCheckROI : vecRectCheckROIs) {
        cv::Mat matCheckROI(matHeightROI, rectCheckROI), matHighMask, matMidMask, matLowMask, matBW;
        auto matCheckROIClone = matCheckROI.clone();
        VectorOfFloat vecThreshold;
        int nTopY = 0, nBtmY = 0;
        bool bResult = _extractSolder(matCheckROIClone, COVERAGE, vecThreshold, matHighMask, matMidMask, matLowMask, matBW, nTopY, nBtmY);
        if (!bResult)
            continue;
        if (ConfigInstance->getDebugMode() == PR_DEBUG_MODE::SHOW_IMAGE) {
            VisionAlgorithm::showImage("High Mask", matHighMask);
            VisionAlgorithm::showImage("Mid Mask",  matMidMask);
            VisionAlgorithm::showImage("Low Mask",  matLowMask);
        }

        VectorOfPoint vecHighPoints;
        cv::findNonZero(matHighMask, vecHighPoints);
        if (vecHighPoints.empty()) {
            pstRpy->enStatus = VisionStatus::FAIL_TO_EXTRACT_3D_SOLDER;
            return;
        }
        float fSumX = 0.f, fSumY = 0.f;
        for (const auto &point : vecHighPoints) {
            fSumX += point.x;
            fSumY += point.y;
        }
        cv::Point ptCenter(ToInt32(fSumX / vecHighPoints.size()), ToInt32(fSumY / vecHighPoints.size()));
        float fHeightSum = 0.f;
        int nCount = 0;
        for (int row = ptCenter.y - 1; row <= ptCenter.y + 1; ++ row)
        for (int col = ptCenter.x - 1; col <= ptCenter.x + 1; ++ col) {
            fHeightSum += matCheckROIClone.at<DATA_TYPE>(row, col);
            ++ nCount;
        }
        bool bLeft = (ptCenter.x + rectCheckROI.x - pstCmd->rectDeviceROI.x) <  pstCmd->rectDeviceROI.width / 2;
        cv::Rect rectCalc;
        auto fSolderHeight = _calcSolderHeightTri(matCheckROIClone, matMidMask, pstCmd->nWettingWidth, bLeft, nTopY, nBtmY, rectCalc);

        PR_INSP_3D_SOLDER_RPY::RESULT result;
        result.fComponentHeight = fHeightSum / nCount;
        result.fSolderHeight = fSolderHeight;
        result.fSolderArea = ToFloat(cv::countNonZero(matMidMask));
        result.fSolderRatio = result.fSolderArea / rectCheckROI.size().area();
        pstRpy->vecResults.push_back(result);

        if (!pstRpy->matResultImg.empty()) {
            cv::Mat matDrawROI(pstRpy->matResultImg, pstCmd->rectDeviceROI);
            cv::Mat matDrawROIClone = matDrawROI.clone();
            if (bIsVertialROI) {
                cv::transpose(matDrawROIClone, matDrawROIClone);
                cv::flip(matDrawROIClone, matDrawROIClone, 1); //transpose+flip(1)=CW
            }
            cv::Mat matResultROI(matDrawROIClone, rectCheckROI);
            matResultROI.setTo(VisionAlgorithm::YELLOW_SCALAR, matHighMask);
            matResultROI.setTo(VisionAlgorithm::BLUE_SCALAR, matMidMask);
            matResultROI.setTo(VisionAlgorithm::RED_SCALAR, matLowMask);
            rectCalc.x += rectCheckROI.x;
            rectCalc.y += rectCheckROI.y;
            cv::rectangle(matDrawROIClone, rectCalc, VisionAlgorithm::GREEN_SCALAR, 1);
            if (bIsVertialROI) {
                cv::transpose(matDrawROIClone, matDrawROIClone);
                cv::flip(matDrawROIClone, matDrawROIClone, 0); //transpose+flip(0)=CCW
            }
            matDrawROIClone.copyTo(matDrawROI);
        }
    }
}

/*static*/ bool Unwrap::_extractSolder(const cv::Mat &matCheckROI, float fCoverage, VectorOfFloat &vecThreshold, cv::Mat &matHighMask, cv::Mat &matMidMask, cv::Mat &matLowMask, cv::Mat &matBW, int &nTopY, int &nBtmY) {
    cv::Mat matCheckOneRow = matCheckROI.clone();
    matCheckOneRow = matCheckOneRow.reshape(1, 1);
    cv::sort(matCheckOneRow, matCheckOneRow, cv::SortFlags::SORT_ASCENDING + cv::SortFlags::SORT_EVERY_ROW);
    const float fEffectHRatioStart = 0.05f, fEffectHRatioEnd = 0.95f;
    int nStart = ToInt32(fEffectHRatioStart * matCheckOneRow.cols);
    int nEnd   = ToInt32(fEffectHRatioEnd   * matCheckOneRow.cols);
    cv::Mat matFetch(matCheckOneRow, cv::Range(0, 1), cv::Range(nStart, nEnd));
    matFetch = matFetch.clone();
    int filterSize = ToInt32(matFetch.cols * 0.05);
    cv::Mat matFilter;
    cv::blur(matFetch, matFilter, cv::Size(filterSize, 1), cv::Point(-1, -1), cv::BorderTypes::BORDER_REPLICATE);
    cv::Mat matDiff = CalcUtils::diff(matFilter, 1, CalcUtils::DIFF_ON_X_DIR).clone(), matDiffFilter;
    cv::blur(matDiff, matDiffFilter, cv::Size(filterSize, 1), cv::Point(-1, -1), cv::BorderTypes::BORDER_REPLICATE);
#ifdef _DEBUG
    auto vecVecCheckOneRow = CalcUtils::matToVector<DATA_TYPE>(matCheckOneRow);
    auto vecVecFetch = CalcUtils::matToVector<DATA_TYPE>(matFetch);
    auto vecVecFetchFilter = CalcUtils::matToVector<DATA_TYPE>(matFilter);
    auto vecVecDiff = CalcUtils::matToVector<DATA_TYPE>(matDiff);
    auto vecVecDiffFilter = CalcUtils::matToVector<DATA_TYPE>(matDiffFilter);
#endif

    double dMinValue = 0., dMaxValue = 0.;
    cv::Point ptMin, ptMax;
    cv::minMaxLoc(matDiffFilter, &dMinValue, &dMaxValue, &ptMin, &ptMax);
    float fMinValue = ToFloat(dMinValue), fMaxValue = ToFloat(dMaxValue);
    float mH = (fMinValue + fMaxValue) / 2.f;
    float fTotalSum = ToFloat(cv::sum(matDiffFilter)[0]);
    float fOverRatio = 0.f;
    int N = 0;
    do
    {
        mH = (fMinValue + fMaxValue) / 2.f;
        float fOverSum = 0;
        for (int i = 0; i < matDiffFilter.cols; ++ i) {
            auto value = matDiffFilter.at<DATA_TYPE>(i);
            if (value > mH)
                fOverSum += value;
        }
        fOverRatio = fOverSum / fTotalSum;
        if (fabs(fOverRatio - fCoverage) < 0.001f)
            break;

        if (fOverRatio > fCoverage)
            fMinValue = mH;
        else
            fMaxValue = mH;

        ++ N;
    }while(fabs(fOverRatio - fCoverage) > 0.001f && N < 20);

    std::deque<int> vecIndex1, vecIndex2;
    for (int col = 0; col < matDiffFilter.cols - 1; ++ col) {
        auto value1 = matDiffFilter.at<DATA_TYPE>(col);
        auto value2 = matDiffFilter.at<DATA_TYPE>(col + 1);
        if (value1 <= mH && value2 > mH)
            vecIndex1.push_back(col);
        if (value1 >= mH && value2 < mH)
            vecIndex2.push_back(col);
    }
    if (matDiffFilter.at<DATA_TYPE>(0) > mH)
        vecIndex1.push_front(0);
    if (matDiffFilter.at<DATA_TYPE>(matDiffFilter.cols - 1) > mH)
        vecIndex2.push_back(matDiffFilter.cols - 1);

    if (vecIndex1.empty() || vecIndex2.empty()) {
        WriteLog("Cannot find solder move up points.");
        return false;
    }

    int size = ToInt32(std::min(vecIndex1.size(), vecIndex2.size()));
    int nMaxGap = std::numeric_limits<int>::min();
    int nFindIndex = 0;
    for (int i = 0; i < size; ++ i) {
        if (vecIndex2[i] - vecIndex1[i] > nMaxGap)
            nFindIndex = i;
    }

    auto thres1 = matFetch.at<DATA_TYPE>(vecIndex1[nFindIndex]);
    auto thres2 = matFetch.at<DATA_TYPE>(vecIndex2[nFindIndex]);

    vecThreshold = VectorOfFloat{thres1, thres2};

    matBW = cv::Mat::zeros(matCheckROI.size(), CV_8UC1);
    matHighMask = matCheckROI >= thres2;
    matMidMask = matCheckROI >= thres1 & matCheckROI < thres2;
    matLowMask = matCheckROI < thres1;

    cv::Mat matHeightMaskFloat, matHeightMaskReduce;
    matHighMask.convertTo(matHeightMaskFloat, CV_32FC1, 1. / 255.);
    cv::reduce(matHeightMaskFloat, matHeightMaskReduce, 1, cv::ReduceTypes::REDUCE_SUM);
    float fMeanWidth = ToFloat(cv::mean(matHeightMaskReduce)[0]);
    float fEffectWidth = 0.8f * fMeanWidth;
    nTopY = 0, nBtmY = 0;
    // To find first and last row with values.
    for (int row = 0; row < matHeightMaskReduce.rows; ++ row) {
        auto value = matHeightMaskReduce.at<DATA_TYPE>(row);
        if (value > fMeanWidth) {
            nBtmY = row;
            if (nTopY == 0)
                nTopY = row;
        }
    }

    cv::Mat matMask = cv::Mat::zeros(matCheckROI.size(), CV_8UC1);
    cv::Mat matTopMask(matMask, cv::Range(0, nTopY), cv::Range::all());
    matTopMask.setTo(PR_MAX_GRAY_LEVEL);
    cv::Mat matBtmMask(matMask, cv::Range(nBtmY + 1, matCheckROI.rows), cv::Range::all());
    matBtmMask.setTo(PR_MAX_GRAY_LEVEL);

    // Clear the values over the range of device.
    matHighMask.setTo(0, matMask);
    matMidMask.setTo(0, matMask);
    matLowMask.setTo(0, matMask);

    matBW.setTo(3, matHighMask);
    matBW.setTo(2, matMidMask);
    matBW.setTo(1, matLowMask);

    return true;
}

/*static*/ float Unwrap::_calcSolderHeightXSG(const cv::Mat &matCheckROI, const cv::Mat &matMidMask) {
    auto matCheckROIClone = matCheckROI.clone();
    cv::Mat matReverseMask = PR_MAX_GRAY_LEVEL - matMidMask;
    matCheckROIClone.setTo(0, matReverseMask);
    cv::Mat matSolderHeightSum;
    cv::reduce(matCheckROIClone, matSolderHeightSum, 0, cv::ReduceTypes::REDUCE_SUM);
#ifdef _DEBUG
    auto vecVecMidMask = CalcUtils::matToVector<uchar>(matMidMask);
    auto vecVecReverseMidMask = CalcUtils::matToVector<uchar>(matReverseMask);
    auto vecVecHeight = CalcUtils::matToVector<DATA_TYPE>(matCheckROIClone);
    auto vecVecSolderHeightSum = CalcUtils::matToVector<DATA_TYPE>(matSolderHeightSum);
#endif
    cv::Mat matMidMaskOneRow;
    cv::reduce(matMidMask, matMidMaskOneRow, 0, cv::ReduceTypes::REDUCE_SUM, CV_32FC1);
    matMidMaskOneRow = matMidMaskOneRow / PR_MAX_GRAY_LEVEL;
    int nSolderStart = 0, nSolderEnd = 0;
    float startCount = 0, endCount = 0;
    for (int col = 0; col < matMidMaskOneRow.cols; ++col) {
        auto value = matMidMaskOneRow.at<DATA_TYPE>(col);
        if (value > matMidMask.rows / 2) {
            nSolderEnd = col;
            endCount = value;
            if (nSolderStart == 0) {
                nSolderStart = col;
                startCount = value;
            }
        }
    }

    float fSolderHeight = 0.f;
    if (nSolderStart > 0 && nSolderEnd > 0) {
        float fStartHeight = matSolderHeightSum.at<DATA_TYPE>(nSolderStart) / startCount;
        float fEndHeight = matSolderHeightSum.at<DATA_TYPE>(nSolderEnd) / endCount;
        fSolderHeight = fabs(fEndHeight - fStartHeight);
    }

    return fSolderHeight;
}

/*static*/ float Unwrap::_calcSolderHeightTri(const cv::Mat &matCheckROI, const cv::Mat &matMidMask, int nWettingWidth, bool bLeft, int nTopY, int nBtmY, cv::Rect &rectCalc) {
    auto matCheckROIClone = matCheckROI.clone();
    cv::Mat matReverseMask = PR_MAX_GRAY_LEVEL - matMidMask;
    matCheckROIClone.setTo(0, matReverseMask);
    cv::Mat matSolderHeightSum;
    cv::reduce(matCheckROIClone, matSolderHeightSum, 0, cv::ReduceTypes::REDUCE_SUM);
#ifdef _DEBUG
    auto vecVecMidMask = CalcUtils::matToVector<uchar>(matMidMask);
    auto vecVecReverseMidMask = CalcUtils::matToVector<uchar>(matReverseMask);
    auto vecVecHeight = CalcUtils::matToVector<DATA_TYPE>(matCheckROIClone);
    auto vecVecSolderHeightSum = CalcUtils::matToVector<DATA_TYPE>(matSolderHeightSum);
#endif
    cv::Mat matMidMaskOneRow;
    cv::reduce(matMidMask, matMidMaskOneRow, 0, cv::ReduceTypes::REDUCE_SUM, CV_32FC1);
    matMidMaskOneRow = matMidMaskOneRow / PR_MAX_GRAY_LEVEL;
    int nSolderStart = 0, nSolderEnd = 0;
    float startCount = 0, endCount = 0;
    for (int col = 0; col < matMidMaskOneRow.cols; ++col) {
        auto value = matMidMaskOneRow.at<DATA_TYPE>(col);
        if (value > matMidMask.rows / 6) {
            nSolderEnd = col;
            endCount = value;
            if (nSolderStart == 0) {
                nSolderStart = col;
                startCount = value;
            }
        }
    }

    if (nSolderStart == nSolderEnd) {
        std::stringstream ss;
        ss << "Cannot find solder area, please check input area.";
        WriteLog(ss.str());
        return 0;
    }

    int nHeight = nBtmY - nTopY;
    int nShrink = nHeight / 10;
    int nCheckTopY = nTopY + nShrink;
    int nCheckBtmY = nBtmY - nShrink;
    if (nBtmY <= nTopY) {
        std::stringstream ss;
        ss << "Calculate solder height goes wrong, the check range of Y (" << nTopY << ", " << nBtmY << ") is not correct.";
        WriteLog(ss.str());
        return 0;
    }

    if (bLeft) {
        if (nSolderEnd - 2 - nWettingWidth < 0) {
            std::stringstream ss;
            ss << "Calculate solder height goes wrong, the left side solder end positoin " << nSolderEnd <<  " is too small.";
            WriteLog(ss.str());
            return 0;
        }
        rectCalc = cv::Rect(nSolderEnd - 2 - nWettingWidth, nCheckTopY, nWettingWidth, nCheckBtmY - nCheckTopY);
    }else {
        if (nSolderStart + 2 + nWettingWidth > matCheckROI.cols) {
            std::stringstream ss;
            ss << "Calculate solder height goes wrong, the right side solder start positoin " << nSolderStart <<  " is too big.";
            WriteLog(ss.str());
            return 0;
        }

        rectCalc = cv::Rect(nSolderStart + 2, nCheckTopY, nWettingWidth, nCheckBtmY - nCheckTopY);
    }

    cv::Mat matCalc(matCheckROI, rectCalc);
    matCalc = matCalc.clone();
    matCalc = matCalc.reshape(1, 1);
    auto vecHeights = CalcUtils::matToVector<DATA_TYPE>(matCalc)[0];
    std::sort(vecHeights.begin(), vecHeights.end());
    vecHeights = VectorOfFloat(vecHeights.begin() + ToInt32(0.5f * vecHeights.size()), vecHeights.begin() + ToInt32(0.9f * vecHeights.size()));
    float fHeightSum = 0.f;
    for (const auto height : vecHeights)
        fHeightSum += height;
    float fMeanHeight = fHeightSum / vecHeights.size();
    return fMeanHeight;
}

}
}
