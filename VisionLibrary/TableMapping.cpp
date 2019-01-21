#include "TableMapping.h"
#include "CalcUtils.hpp"
#include "Log.h"

namespace AOI
{
namespace Vision
{

/*static*/ cv::Mat TableMapping::_fitModel(const PR_TABLE_MAPPING_CMD::VectorOfFramePoint &framePoints, cv::Mat& matXX, cv::Mat& matYY) {
    std::vector<float> vecTargetX, vecTargetY, vecActualX, vecActualY;
    vecTargetX.reserve(framePoints.size());
    vecTargetX.reserve(framePoints.size());
    vecActualX.reserve(framePoints.size());
    vecActualY.reserve(framePoints.size());
    for (const auto& mappingPoint : framePoints) {
        vecTargetX.push_back(mappingPoint.targetPoint.x);
        vecTargetY.push_back(mappingPoint.targetPoint.y);

        vecActualX.push_back(mappingPoint.actualPoint.x);
        vecActualY.push_back(mappingPoint.actualPoint.y);
    }

    matXX.release();
    matYY.release();
    matXX.push_back(cv::Mat(vecActualX).reshape(1, 1));
    matXX.push_back(cv::Mat(vecActualY).reshape(1, 1));
    matXX.push_back(cv::Mat(cv::Mat::ones(1, ToInt32(framePoints.size()), CV_32FC1)));
    cv::transpose(matXX, matXX);
    
    matYY.push_back(cv::Mat(vecTargetX).reshape(1, 1));
    matYY.push_back(cv::Mat(vecTargetY).reshape(1, 1));
    cv::transpose(matYY, matYY);

    cv::Mat matK;
    cv::solve(matXX, matYY, matK, cv::DecompTypes::DECOMP_QR);
    return matK;
}

/*static*/ PR_TABLE_MAPPING_CMD::VectorOfFramePoint TableMapping::_preHandleData(const PR_TABLE_MAPPING_CMD::VectorOfFramePoint &framePoints) {
    PR_TABLE_MAPPING_CMD::VectorOfFramePoint rmZeroFramePoints;
    for (const auto& mappingPoint : framePoints)
        if (mappingPoint.actualPoint.x != 0 && mappingPoint.actualPoint.y != 0)
            rmZeroFramePoints.push_back(mappingPoint);

    cv::Mat matXX, matYY;
    cv::Mat matK = _fitModel(rmZeroFramePoints, matXX, matYY);
    cv::Mat matY1 = matXX * matK;
    cv::Mat matAbsErr = cv::abs(matY1 - matYY);

    PR_TABLE_MAPPING_CMD::VectorOfFramePoint rmResultPoints;
    for (int row = 0; row < matAbsErr.rows; ++ row)
        if (matAbsErr.at<float>(row, 0) < 1 && matAbsErr.at<float>(row, 1) < 1)
            rmResultPoints.push_back(rmZeroFramePoints[row]);

    return rmResultPoints;
}

/*static*/ float TableMapping::_calcTransR1RSum(const PR_TABLE_MAPPING_CMD::VectorOfFramePoints& vecFramePoints,
    const cv::Mat& matD1, const cv::Mat& matWeight, cv::Mat& matR1, cv::Mat& matDR1) {
    // D1 = [a, b, c, theta11, Tx1, Ty1, theta12, Tx2, Ty2];
    auto a = matD1.at<float>(0), b = matD1.at<float>(1), c = matD1.at<float>(2);

    //Matlab T1   = [a, c, 0; 0, b, 0; 0, 0, 1];
    cv::Mat T1 = cv::Mat::zeros(3, 3, CV_32FC1);
    T1.at<float>(0, 0) = a; T1.at<float>(0, 1) = c; T1.at<float>(1, 1) = b; T1.at<float>(2, 2) = 1;

    // Matlab dT1a = [1, 0, 0; 0, 0, 0; 0, 0, 0]; It is the derivative of T1 against a
    cv::Mat dT1a = cv::Mat::zeros(3, 3, CV_32FC1);
    dT1a.at<float>(0, 0) = 1.f;

    // Matlab dT1b = [0, 0, 0; 0, 1, 0; 0, 0, 0]; It is the derivative of T1 against b
    cv::Mat dT1b = cv::Mat::zeros(3, 3, CV_32FC1);
    dT1b.at<float>(1, 1) = 1.f;

    // Matlab dT1c = [0, 1, 0; 0, 0, 0; 0, 0, 0]; It is the derivative of T1 against c
    cv::Mat dT1c = cv::Mat::zeros(3, 3, CV_32FC1);
    dT1c.at<float>(0, 1) = 1.f;

    int totalLen = 0;
    for (const auto& framePoints : vecFramePoints)
        totalLen += ToInt32(framePoints.size());

    VectorOfFloat vecTargetX, vecTargetY;
    cv::Mat Axy1 = cv::Mat::zeros(2, totalLen, CV_32FC1); // The calculated target position from real position.
    cv::Mat dRa  = cv::Mat::zeros(2, totalLen, CV_32FC1); // The result of derivative multiply the actual position
    cv::Mat dRb  = cv::Mat::zeros(2, totalLen, CV_32FC1);
    cv::Mat dRc  = cv::Mat::zeros(2, totalLen, CV_32FC1);

    cv::Mat dRTxy = cv::Mat::zeros(3 * ToInt32(vecFramePoints.size()) * 2, totalLen, CV_32FC1);

    int nPos = 0;
    for (int i = 0; i < vecFramePoints.size(); ++ i) {
        auto theta1i = matD1.at<float>(i * 3 + 3);
        auto Txi     = matD1.at<float>(i * 3 + 4);
        auto Tyi     = matD1.at<float>(i * 3 + 5);

        VectorOfVectorOfFloat vecT2{
            VectorOfFloat{cos(theta1i), -sin(theta1i), Txi},
            VectorOfFloat{sin(theta1i), cos(theta1i), Tyi},
            VectorOfFloat{0, 0, 1}};
        cv::Mat T2 = CalcUtils::vectorToMat(vecT2);

        VectorOfVectorOfFloat vecDerivTheta{
            VectorOfFloat{-sin(theta1i), -cos(theta1i), 0},
            VectorOfFloat{cos(theta1i), -sin(theta1i), 0},
            VectorOfFloat{0, 0, 0}};
        cv::Mat derivTheta = CalcUtils::vectorToMat(vecDerivTheta);

        // Matlab dTx = [0, 0, 1; 0, 0, 0; 0, 0, 0]; dTx is the derivative against Tx
        cv::Mat derivTx = cv::Mat::zeros(3, 3, CV_32FC1);
        derivTx.at<float>(0, 2) = 1.f;

        // Matlab dTy = [0, 0, 0; 0, 0, 1; 0, 0, 0];
        cv::Mat derivTy = cv::Mat::zeros(3, 3, CV_32FC1);
        derivTy.at<float>(1, 2) = 1.f;

        // Matlab: Mxyi = [Mxy(:, idx); ones(1, leni)];
        const auto& framePoints = vecFramePoints[i];
        cv::Mat matXYi;
        VectorOfFloat vecActualX, vecActualY;
        vecActualX.reserve(framePoints.size());
        vecActualY.reserve(framePoints.size());
        for (const auto& mappingPoint : framePoints) {
            vecActualX.push_back(mappingPoint.actualPoint.x);
            vecActualY.push_back(mappingPoint.actualPoint.y);

            vecTargetX.push_back(mappingPoint.targetPoint.x);
            vecTargetY.push_back(mappingPoint.targetPoint.y);
        }
        matXYi.push_back(cv::Mat(vecActualX).reshape(1, 1));
        matXYi.push_back(cv::Mat(vecActualY).reshape(1, 1));
        matXYi.push_back(cv::Mat(cv::Mat::ones(1, ToInt32(framePoints.size()), CV_32FC1)));

        // Matlab Axy1(:, idx) = Axyi(1:2, :);
        cv::Mat AXYi = T2 * T1 * matXYi;
        {
            cv::Mat matCopySrc(AXYi, cv::Range(0, 2), cv::Range::all());
            cv::Mat matAxy1ROI(Axy1, cv::Range::all(), cv::Range(nPos, nPos + AXYi.cols));
            matCopySrc.copyTo(matAxy1ROI);
        }
#ifdef _DEBUG
        auto vecVecAXYi = CalcUtils::matToVector<float>(AXYi);
#endif

        // Matlab dRa(:, idx)  = dRai(1:2, :);
        cv::Mat dRai = T2 * dT1a * matXYi;
        {
            cv::Mat matCopySrc(dRai, cv::Range(0, 2), cv::Range::all());
            cv::Mat matRaROI(dRa, cv::Range::all(), cv::Range(nPos, nPos + AXYi.cols));
            matCopySrc.copyTo(matRaROI);
        }

        // Matlab dRb(:, idx)  = dRbi(1:2, :);
        cv::Mat dRbi = T2 * dT1b * matXYi;
        {
            cv::Mat matCopySrc(dRbi, cv::Range(0, 2), cv::Range::all());
            cv::Mat matRbROI(dRb, cv::Range::all(), cv::Range(nPos, nPos + AXYi.cols));
            matCopySrc.copyTo(matRbROI);
        }

        // Matlab dRc(:, idx)  = dRci(1:2, :);
        cv::Mat dRci = T2 * dT1c * matXYi;
        {
            cv::Mat matCopySrc(dRci, cv::Range(0, 2), cv::Range::all());
            cv::Mat matRcROI(dRc, cv::Range::all(), cv::Range(nPos, nPos + AXYi.cols));
            matCopySrc.copyTo(matRcROI);
        }

        cv::Mat dThetai = derivTheta * T1 * matXYi;
        {
            cv::Mat matCopySrc(dThetai, cv::Range(0,2), cv::Range::all());
            cv::Mat dRTxyROI(dRTxy, cv::Range(i * 6, i * 6 + 2), cv::Range(nPos, nPos + AXYi.cols));
            matCopySrc.copyTo(dRTxyROI);
        }

        cv::Mat dTxi = derivTx * T1 * matXYi;
        {
            cv::Mat matCopySrc(dTxi, cv::Range(0, 2), cv::Range::all());
            cv::Mat dRTxyROI(dRTxy, cv::Range(i * 6 + 2, i * 6 + 4), cv::Range(nPos, nPos + AXYi.cols));
            matCopySrc.copyTo(dRTxyROI);
        }

        cv::Mat dTyi = derivTy * T1 * matXYi;
        {
            cv::Mat matCopySrc(dTyi, cv::Range(0, 2), cv::Range::all());
            cv::Mat dRTxyROI(dRTxy, cv::Range(i * 6 + 4, i * 6 + 6), cv::Range(nPos, nPos + AXYi.cols));
            matCopySrc.copyTo(dRTxyROI);
#ifdef _DEBUG
        auto vecVecdTyi = CalcUtils::matToVector<float>(dTyi);
#endif
        }

        nPos += AXYi.cols;
    }

    cv::Mat Axy;
    Axy.push_back(cv::Mat(vecTargetX).reshape(1, 1));
    Axy.push_back(cv::Mat(vecTargetY).reshape(1, 1));

    //const cv::Mat matWeightConcat = matWeight.reshape(1, 1).clone();
    cv::Mat matWeightConcatRepeat = cv::repeat(matWeight.clone(), 2, 1);

    matR1 = Axy - Axy1;
    cv::multiply(matR1, matWeightConcatRepeat, matR1);
    matR1 = matR1.reshape(1, ToInt32(matR1.total()));   // Reshape for outside calculation
    float normR1 = ToFloat(cv::norm(matR1));

    dRa = dRa.reshape(1, 1);
    dRb = dRb.reshape(1, 1);
    dRc = dRc.reshape(1, 1);
    dRTxy = dRTxy.reshape(1, 3 * ToInt32(vecFramePoints.size()));

    matDR1.release();
    matDR1.push_back(dRa);
    matDR1.push_back(dRb);
    matDR1.push_back(dRc);
    matDR1.push_back(dRTxy);
    cv::transpose(matDR1, matDR1);

    matWeightConcatRepeat = cv::repeat(matWeightConcatRepeat.reshape(1, 1), matDR1.cols, 1);
    cv::transpose(matWeightConcatRepeat, matWeightConcatRepeat);

    cv::multiply(matDR1, matWeightConcatRepeat, matDR1);
    matDR1 = -matDR1;

    return normR1;
}

/*static*/ int TableMapping::_findMatchPoint(const VectorOfFloat& vecTargetX, const VectorOfFloat& vecTargetY, const cv::Point2f& ptSrch) {
    assert(vecTargetX.size() == vecTargetY.size());
    for (size_t i = 0; i < vecTargetX.size(); ++ i) {
        if (fabs(vecTargetX[i] - ptSrch.x) < 1 && fabs(vecTargetY[i] - ptSrch.y) < 1)
            return ToInt32(i);
    }

    return -1;
}

/*static*/ cv::Mat TableMapping::_calculatePPz(const cv::Mat& matX, const cv::Mat& matY, const cv::Mat& matZ,
        const VectorOfFloat& vecXyMinMax, int mm, int nn) {
    cv::Mat matX1, matY1, matZ1;
    for (int row = 0; row < matZ.rows; ++ row)
    for (int col = 0; col < matZ.cols; ++ col){
        if (!std::isnan(matZ.at<float>(row, col))) {
            matX1.push_back(matX.at<float>(row, col));
            matY1.push_back(matY.at<float>(row, col));
            matZ1.push_back(matZ.at<float>(row, col));
        }
    }

#ifdef _DEBUG
    cv::Mat matX1T; cv::transpose(matX1, matX1T);
    auto vecVecX1T = CalcUtils::matToVector<float>(matX1T);
    cv::Mat matY1T; cv::transpose(matY1, matY1T);
    auto vecVecY1T = CalcUtils::matToVector<float>(matY1T);
    cv::Mat matZ1T; cv::transpose(matZ1, matZ1T);
    auto vecVecZ1T = CalcUtils::matToVector<float>(matZ1T);
#endif

    cv::Mat matXX = CalcUtils::generateBezier(matX1, matY1, vecXyMinMax, mm, nn);

    //Matlab x'*x
    cv::Mat matXXT; cv::transpose(matXX, matXXT);
    cv::Mat matLeft = matXXT * matXX;

    // Matlab xx'*Z1
    cv::Mat matRight = matXXT * matZ1;

    cv::Mat PPz;
    cv::solve(matLeft, matRight, PPz, cv::DecompTypes::DECOMP_SVD);
    return PPz;
}

/*static*/ cv::Mat TableMapping::_calculateSurface(const cv::Mat& matX, const cv::Mat& matY, const cv::Mat& matPPz, const VectorOfFloat& vecXyMinMax, int mm, int nn) {
    cv::Mat matXInRow = matX.reshape(1, matX.rows * matX.cols);
    cv::Mat matYInRow = matY.reshape(1, matY.rows * matY.cols);

    cv::Mat matXX = CalcUtils::generateBezier(matXInRow, matYInRow, vecXyMinMax, mm, nn);

    cv::Mat zp1 = matXX * matPPz;
    zp1 = zp1.reshape(1, matX.rows);
    return zp1;
}

/*static*/ ByteVector TableMapping::_findCrossBorderPoint(const PR_TABLE_MAPPING_CMD::VectorOfFramePoints& vecProcessedFramePoints) {
	ByteVector vecResult;
    const float BORDER_MARGIN = 5.f;
    const auto& firstFramePoints = vecProcessedFramePoints.front();
    const auto& lastFramePoints  = vecProcessedFramePoints.back();
    float fLft = firstFramePoints.front().actualPoint.x, fTop = firstFramePoints.front().actualPoint.y,
          fRgt = lastFramePoints.back().actualPoint.x,   fBtm = lastFramePoints.back().actualPoint.y;
    for (size_t i = 0; i < vecProcessedFramePoints.size(); ++ i) {
        const auto& frameLft = vecProcessedFramePoints[i].front().actualPoint.x;
        const auto& frameTop = vecProcessedFramePoints[i].front().actualPoint.y;
        const auto& frameRgt = vecProcessedFramePoints[i].back().actualPoint.x;
        const auto& frameBtm = vecProcessedFramePoints[i].back().actualPoint.y;
        for (size_t j = 0; j < vecProcessedFramePoints[i].size(); ++ j) {
            const auto& ptCurrentPoint = vecProcessedFramePoints[i][j].actualPoint;
			Byte result = 0;
            if (fabs(ptCurrentPoint.x - frameLft) < BORDER_MARGIN && fabs(ptCurrentPoint.x - fLft) > BORDER_MARGIN)
				result = 1;
            if (fabs(ptCurrentPoint.y - frameTop) < BORDER_MARGIN && fabs(ptCurrentPoint.y - fTop) > BORDER_MARGIN)
				result = 1;
            if (fabs(ptCurrentPoint.x - frameRgt) < BORDER_MARGIN && fabs(ptCurrentPoint.x - fRgt) > BORDER_MARGIN)
				result = 1;
            if (fabs(ptCurrentPoint.y - frameBtm) < BORDER_MARGIN && fabs(ptCurrentPoint.y - fBtm) > BORDER_MARGIN)
				result = 1;
			vecResult.push_back(result);
        }
    }

	return vecResult;
}

/*static*/ void TableMapping::_processMultiFrameData(const PR_TABLE_MAPPING_CMD::VectorOfFramePoints& vecProcessedFramePoints,
    const VectorOfVectorOfFloat& vecVecTransM, cv::Mat& xtt, cv::Mat& ytt, cv::Mat& dxt, cv::Mat& dyt,
        const PR_TABLE_MAPPING_CMD *const pstCmd, PR_TABLE_MAPPING_RPY *const pstRpy) {
    cv::Mat matTransM = CalcUtils::vectorToMat(vecVecTransM);
    cv::Mat matMeanTransM;
    cv::reduce(matTransM, matMeanTransM, 0, cv::ReduceTypes::REDUCE_AVG);
    auto k1 = matMeanTransM.at<float>(0);
    auto k2 = matMeanTransM.at<float>(1);
    auto theta2 = matMeanTransM.at<float>(2);

    auto vecBorderPointMask = _findCrossBorderPoint(vecProcessedFramePoints);
	cv::Mat matWeight = cv::Mat::ones(1, ToInt32(vecBorderPointMask.size()), CV_32FC1);
	for (int i = 0; i < vecBorderPointMask.size(); ++ i) {
		if (vecBorderPointMask[i] > 0)
			matWeight.at<float>(i) = pstCmd->fFrameBorderPointWeight;
    }

#ifdef _DEBUG
    auto vecVecWeight = CalcUtils::matToVector<float>(matWeight);
#endif

    auto a =  k1;
    auto c = -k2 * sin(theta2);
    auto b =  k2 * cos(theta2);

    cv::Mat matTxy(matTransM, cv::Range::all(), cv::Range(3,6));
    matTxy = matTxy.clone();
    matTxy = matTxy.reshape(1, 1);

    auto vecVecTxy = CalcUtils::matToVector<float>(matTxy);
    VectorOfFloat vecD1 = {a, b,c};
    vecD1.insert(vecD1.end(), vecVecTxy[0].begin(), vecVecTxy[0].end());

    // adding test error
    //VectorOfFloat dd{0.f, 0.f, 0.f, 0.001f, -5.34f, 6.79f, -0.01f, 20.8f, 4.9f};
    ////VectorOfFloat dd{0.f, 0.f, 0.f, 0.001f, -5.34f, 6.79f};
    //for (size_t i = 0; i < dd.size(); ++ i)
    //    vecD1[i] += dd[i];
    cv::Mat matD1(vecD1);

    cv::Mat matR1, matDR1;
    float Rsum1 = _calcTransR1RSum(vecProcessedFramePoints, matD1, matWeight, matR1, matDR1);

#ifdef _DEBUG
    cv::Mat matR1T; cv::transpose(matR1, matR1T);
    auto vecVecR1T = CalcUtils::matToVector<float>(matR1T);
#endif

    cv::Mat derivResidual = cv::Mat::ones(ToInt32(vecD1.size()), 1, CV_32FC1);
    int N  = 0;
    float mu = 1e-6f;
    auto eyeX = cv::Mat::eye(ToInt32(vecD1.size()), ToInt32(vecD1.size()), CV_32FC1);
    float fDrNorm = ToFloat(cv::norm(derivResidual));
    const float EPSILON = 0.01f;

    while(fDrNorm > EPSILON) { // iteration end condition
        mu = mu/3;
        auto Rsum = Rsum1; 
        auto Jr = matDR1;
        cv::Mat JrT; cv::transpose(Jr, JrT);
        auto Rt = matR1;

        cv::Mat matXX = JrT * Jr  + mu * eyeX;
        cv::Mat matYY = JrT * Rt;
        cv::solve(matXX, matYY, derivResidual, cv::DecompTypes::DECOMP_SVD);
        fDrNorm = ToFloat(cv::norm(derivResidual));
        derivResidual = -derivResidual;

#ifdef _DEBUG
        cv::Mat matDrT; cv::transpose(derivResidual, matDrT);
        auto vecVecDrT = CalcUtils::matToVector<float>(matDrT);
#endif
      
        if (++ N > 100)
            break;

        matD1 = matD1 + derivResidual;
        Rsum1 = _calcTransR1RSum(vecProcessedFramePoints, matD1, matWeight, matR1, matDR1);

        int nn = 0;
        bool flag = false;
        fDrNorm = ToFloat(cv::norm(derivResidual));

        while (Rsum1 > Rsum && fDrNorm > EPSILON) {
            if (++nn > 10) {
                flag = true;
                break;
            }

            mu = mu*2;
            cv::Mat matXX = JrT * Jr + mu * eyeX;
            cv::Mat matYY = JrT * Rt;
            cv::solve(matXX, matYY, derivResidual);
            fDrNorm = ToFloat(cv::norm(derivResidual));
            derivResidual = -derivResidual;

            matD1 = matD1 + derivResidual;
            Rsum1 = _calcTransR1RSum(vecProcessedFramePoints, matD1, matWeight, matR1, matDR1);
        }

        if (flag)
            break;
    }

    auto d = matD1;

#ifdef _DEBUG
    cv::Mat matD1T; cv::transpose(matD1, matD1T);
    auto vecVecD1T = CalcUtils::matToVector<float>(matD1T);
#endif

    a = d.at<float>(0); b =  d.at<float>(1); c = d.at<float>(2);
    xtt.release();
    ytt.release();
    dxt.release();
    dyt.release();
    int matchPointNumber = 0;
    for (int i = 0; i < vecProcessedFramePoints.size(); ++ i) {
        auto theta1i = matD1.at<float>(i * 3 + 3);
        auto Txi     = matD1.at<float>(i * 3 + 4);
        auto Tyi     = matD1.at<float>(i * 3 + 5);

        VectorOfFloat vecTargetX, vecTargetY, vecActualX, vecActualY;
        const int dataLen = ToInt32(vecProcessedFramePoints[i].size());
        vecTargetX.reserve(dataLen);
        vecTargetX.reserve(dataLen);
        vecActualX.reserve(dataLen);
        vecActualY.reserve(dataLen);
        for (const auto& mappingPoint : vecProcessedFramePoints[i]) {
            vecTargetX.push_back(mappingPoint.targetPoint.x);
            vecTargetY.push_back(mappingPoint.targetPoint.y);

            vecActualX.push_back(mappingPoint.actualPoint.x);
            vecActualY.push_back(mappingPoint.actualPoint.y);
        }

        auto minMaxIterOfX = std::minmax_element(vecTargetX.begin(), vecTargetX.end());
        auto minMaxIterOfY = std::minmax_element(vecTargetY.begin(), vecTargetY.end());

        cv::Mat xt, yt;
        CalcUtils::meshgrid<float>(
            *minMaxIterOfX.first, pstCmd->fBoardPointDist, *minMaxIterOfX.second,
            *minMaxIterOfY.first, pstCmd->fBoardPointDist, *minMaxIterOfY.second, xt, yt);
        cv::Mat dx = cv::Mat::ones(xt.size(), CV_32FC1) * NAN;
        cv::Mat dy = dx.clone();

        for (int row = 0; row < xt.rows; ++ row)
        for (int col = 0; col < xt.cols; ++ col) {
            auto index = _findMatchPoint(vecTargetX, vecTargetY, cv::Point2f(xt.at<float>(row, col), yt.at<float>(row, col)));
            if (index >= 0) {
                dx.at<float>(row, col) = vecActualX[index];
                dy.at<float>(row, col) = vecActualY[index];
                xt.at<float>(row, col) = vecTargetX[index];
                yt.at<float>(row, col) = vecTargetY[index];
                ++ matchPointNumber;
            }
        }

        auto Tx1 = - ( Txi * cos(theta1i) + Tyi * sin(theta1i));
        auto Ty1 = - (-Txi * sin(theta1i) + Tyi * cos(theta1i));

        cv::Mat xti =  xt * cos(theta1i) + yt * sin(theta1i) + Tx1;
        cv::Mat yti = -xt * sin(theta1i) + yt * cos(theta1i) + Ty1;

        cv::Mat xte = a * dx + c * dy;
        cv::Mat yte = b * dy;

        cv::Mat dxi = xti - xte;
        cv::Mat dyi = yti - yte;

        // Matlab xtt = [xtt; xte(:)];
        cv::transpose(xte, xte);
        xtt.push_back(xte.reshape(1, xte.rows * xte.cols));

        // Matlab ytt = [ytt; yte(:)];
        cv::transpose(yte, yte);
        ytt.push_back(yte.reshape(1, yte.rows * yte.cols));

        // Matlab dxt = [dxt; dxi(:)];
        cv::transpose(dxi, dxi);
        dxt.push_back(dxi.reshape(1, dxi.rows * dxi.cols));

        // Matlab dyt = [dyt; dyi(:)];
        cv::transpose(dyi, dyi);
        dyt.push_back(dyi.reshape(1, dyi.rows * dyi.cols));
    }

#ifdef _DEBUG
    cv::Mat matdxtT; cv::transpose(dxt, matdxtT);
    auto vecVecDxtT = CalcUtils::matToVector<float>(matdxtT);
    cv::Mat matdytT; cv::transpose(dyt, matdytT);
    auto vecVecDytT = CalcUtils::matToVector<float>(matdytT);
#endif

    const int minPointsRequired = pstCmd->nBezierRank * pstCmd->nBezierRank;
    if (matchPointNumber < minPointsRequired) {
        pstRpy->enStatus = VisionStatus::TABLE_MAPPING_FAIL;
        std::stringstream ss;
        ss << "The match points number " << matchPointNumber << " is less than minimum required point number " << minPointsRequired;
        WriteLog(ss.str());
        return;
    }

    const float BORDER_MARGIN = 1.f;
    double minX = 0., maxX = 0., minY = 0., maxY = 0.;
    cv::minMaxIdx(xtt, &minX, &maxX);
    cv::minMaxIdx(ytt, &minY, &maxY);

    minX += BORDER_MARGIN; maxX -= BORDER_MARGIN;
    minY += BORDER_MARGIN; maxY -= BORDER_MARGIN;

    pstRpy->fMinX = ToFloat(minX);
    pstRpy->fMaxX = ToFloat(maxX);
    pstRpy->fMinY = ToFloat(minY);
    pstRpy->fMaxY = ToFloat(maxY);
    pstRpy->a = a;
    pstRpy->b = b;
    pstRpy->c = c;
}

/*static*/ void TableMapping::_processSingleFrameData(const PR_TABLE_MAPPING_CMD::VectorOfFramePoints& vecProcessedFramePoints, const VectorOfVectorOfFloat& vecVecTransM,
        cv::Mat& xtt, cv::Mat& ytt, cv::Mat& dxt, cv::Mat& dyt,
        const PR_TABLE_MAPPING_CMD *const pstCmd, PR_TABLE_MAPPING_RPY *const pstRpy) {
    xtt.release();
    ytt.release();
    dxt.release();
    dyt.release();

    cv::Mat matXX, matYY;
    auto matK = _fitModel(vecProcessedFramePoints[0], matXX, matYY);

    cv::Mat Y1 = matXX * matK;

    cv::Mat matErr = matYY - Y1;

    assert(vecProcessedFramePoints.size() == 1);

    VectorOfFloat vecTargetX, vecTargetY, vecActualX, vecActualY, vecErrX, vecErrY;
    const int dataLen = ToInt32(vecProcessedFramePoints[0].size());
    vecTargetX.reserve(dataLen);
    vecTargetX.reserve(dataLen);
    vecActualX.reserve(dataLen);
    vecActualY.reserve(dataLen);
    int index = 0;
    for (const auto& mappingPoint : vecProcessedFramePoints[0]) {
        vecTargetX.push_back(mappingPoint.targetPoint.x);
        vecTargetY.push_back(mappingPoint.targetPoint.y);

        vecActualX.push_back(mappingPoint.actualPoint.x);
        vecActualY.push_back(mappingPoint.actualPoint.y);

        vecErrX.push_back(matErr.at<float>(index, 0));
        vecErrY.push_back(matErr.at<float>(index, 1));
        ++ index;
    }

    auto minMaxIterOfX = std::minmax_element(vecTargetX.begin(), vecTargetX.end());
    auto minMaxIterOfY = std::minmax_element(vecTargetY.begin(), vecTargetY.end());

    cv::Mat xt, yt;
    CalcUtils::meshgrid<float>(
        *minMaxIterOfX.first, pstCmd->fBoardPointDist, *minMaxIterOfX.second,
        *minMaxIterOfY.first, pstCmd->fBoardPointDist, *minMaxIterOfY.second, xt, yt);
    dxt = cv::Mat::ones(xt.size(), CV_32FC1) * NAN; 
    dyt = dxt.clone();

    int matchPointNumber = 0;
    for (int row = 0; row < xt.rows; ++row)
    for (int col = 0; col < xt.cols; ++col) {
        auto index = _findMatchPoint(vecTargetX, vecTargetY, cv::Point2f(xt.at<float>(row, col), yt.at<float>(row, col)));
        if (index >= 0) {
            dxt.at<float>(row, col) = vecErrX[index];
            dyt.at<float>(row, col) = vecErrY[index];
            xt.at<float>(row, col) = vecTargetX[index];
            yt.at<float>(row, col) = vecTargetY[index];
            ++matchPointNumber;
        }
    }

    const int minPointsRequired = pstCmd->nBezierRank * pstCmd->nBezierRank;
    if (matchPointNumber < minPointsRequired) {
        pstRpy->enStatus = VisionStatus::TABLE_MAPPING_FAIL;
        std::stringstream ss;
        ss << "The match points number " << matchPointNumber << " is less than minimum required point number " << minPointsRequired;
        WriteLog(ss.str());
        return;
    }

#ifdef _DEBUG
    cv::Mat matdxtT; cv::transpose(dxt, matdxtT);
    auto vecVecDxtT = CalcUtils::matToVector<float>(matdxtT);
    cv::Mat matdytT; cv::transpose(dyt, matdytT);
    auto vecVecDytT = CalcUtils::matToVector<float>(matdytT);
#endif

    xtt = xt; ytt = yt;

    double minX = 0., maxX = 0., minY = 0., maxY = 0.;
    cv::minMaxIdx(xtt, &minX, &maxX);
    cv::minMaxIdx(ytt, &minY, &maxY);

    pstRpy->fMinX = ToFloat(minX);
    pstRpy->fMaxX = ToFloat(maxX);
    pstRpy->fMinY = ToFloat(minY);
    pstRpy->fMaxY = ToFloat(maxY);
    pstRpy->a = 1.f;
    pstRpy->b = 1.f;
    pstRpy->c = 0.f;
}

/*static*/ VisionStatus TableMapping::run(const PR_TABLE_MAPPING_CMD *const pstCmd, PR_TABLE_MAPPING_RPY *const pstRpy) {
    VectorOfVectorOfFloat vecVecTransM;
    PR_TABLE_MAPPING_CMD::VectorOfFramePoints vecProcessedFramePoints;
    for (const auto &framePoints : pstCmd->vecFramePoints) {
        auto preHandledFramePoints = _preHandleData(framePoints);
        vecProcessedFramePoints.push_back(preHandledFramePoints);

        cv::Mat matXX, matYY;
        auto matK = _fitModel(preHandledFramePoints, matXX, matYY);
        cv::Mat matK1 = cv::Mat(matK, cv::Range(0, 2), cv::Range::all()).clone();
        cv::transpose(matK1, matK1);
        
#ifdef _DEBUG
        auto vecVecK = CalcUtils::matToVector<float>(matK);
#endif
        auto k1 = std::sqrt(matK1.at<float>(0, 0) * matK1.at<float>(0, 0) + matK1.at<float>(1, 0) * matK1.at<float>(1, 0));
        auto k2 = std::sqrt(matK1.at<float>(0, 1) * matK1.at<float>(0, 1) + matK1.at<float>(1, 1) * matK1.at<float>(1, 1));

        auto theta1 = atan( matK1.at<float>(1, 0) / matK1.at<float>(0, 0));
        auto theta2 = atan(-matK1.at<float>(0, 1) / matK1.at<float>(1, 1)) - theta1;

        auto Tx = matK.at<float>(2, 0);
        auto Ty = matK.at<float>(2, 1);

        vecVecTransM.emplace_back(VectorOfFloat{k1, k2, theta2, theta1, Tx, Ty});
    }

    cv::Mat xtt, ytt, dxt, dyt;
    if (vecProcessedFramePoints.size() > 1) 
        _processMultiFrameData(vecProcessedFramePoints, vecVecTransM, xtt, ytt, dxt, dyt, pstCmd, pstRpy);
    else
        _processSingleFrameData(vecProcessedFramePoints, vecVecTransM, xtt, ytt, dxt, dyt, pstCmd, pstRpy);

    std::vector<float> vecParamMinMaxXY{pstRpy->fMinX, pstRpy->fMaxX, pstRpy->fMinY, pstRpy->fMaxY};

    pstRpy->matXOffsetParam = _calculatePPz(xtt, ytt, dxt, vecParamMinMaxXY, pstCmd->nBezierRank, pstCmd->nBezierRank);
    pstRpy->matYOffsetParam = _calculatePPz(xtt, ytt, dyt, vecParamMinMaxXY, pstCmd->nBezierRank, pstCmd->nBezierRank);

#ifdef _DEBUG
    cv::Mat matXOffsetParamT; cv::transpose(pstRpy->matXOffsetParam, matXOffsetParamT);
    auto vecVecXOffsetParamT = CalcUtils::matToVector<float>(matXOffsetParamT);
    cv::Mat matYOffsetParamT; cv::transpose(pstRpy->matYOffsetParam, matYOffsetParamT);
    auto vecVecYOffsetParam = CalcUtils::matToVector<float>(matYOffsetParamT);
#endif

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus TableMapping::calcTableOffset(const PR_CALC_TABLE_OFFSET_CMD* const pstCmd, PR_CALC_TABLE_OFFSET_RPY* const pstRpy) {
    pstRpy->fOffsetX = 0.f;
    pstRpy->fOffsetY = 0.f;

    if (pstCmd->ptTablePos.x < pstCmd->fMinX || pstCmd->ptTablePos.x > pstCmd->fMaxX ||
        pstCmd->ptTablePos.y < pstCmd->fMinY || pstCmd->ptTablePos.y > pstCmd->fMaxY) {
        std::stringstream ss;
        ss << "CalcTableOffset: the input position " << pstCmd->ptTablePos << " is not within calibration range X [" << pstCmd->fMinX << ", "
           << pstCmd->fMaxX << "], Y [" << pstCmd->fMinY << ", " << pstCmd->fMaxY << "]";
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    cv::Mat matX = cv::Mat::zeros(1, 1, CV_32FC1);
    matX.at<float>(0) = pstCmd->ptTablePos.x;

    cv::Mat matY = cv::Mat::zeros(1, 1, CV_32FC1);
    matY.at<float>(0) = pstCmd->ptTablePos.y;

    std::vector<float> vecParamMinMaxXY{pstCmd->fMinX, pstCmd->fMaxX, pstCmd->fMinY, pstCmd->fMaxY};
    cv::Mat matXX = CalcUtils::generateBezier(matX, matY, vecParamMinMaxXY, pstCmd->nBezierRank, pstCmd->nBezierRank);

    if (matXX.cols != pstCmd->matXOffsetParam.rows || matXX.cols != pstCmd->matYOffsetParam.rows) {
        WriteLog("PR_CalcTableOffset: The input calibration result and the input bezier rank not match.");
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    cv::Mat matXOffset = matXX * pstCmd->matXOffsetParam;
    cv::Mat matYOffset = matXX * pstCmd->matYOffsetParam;

    pstRpy->fOffsetX = matXOffset.at<float>(0);
    pstRpy->fOffsetY = matYOffset.at<float>(0);

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ cv::Size2f TableMapping::_calcPointOffset(const PR_CALC_TABLE_OFFSET_CMD *const pstCmd, const cv::Point2f &ptPos) {
    cv::Mat matX = cv::Mat::zeros(1, 1, CV_32FC1);
    matX.at<float>(0) = ptPos.x;

    cv::Mat matY = cv::Mat::zeros(1, 1, CV_32FC1);
    matY.at<float>(0) = ptPos.y;

    std::vector<float> vecParamMinMaxXY{pstCmd->fMinX, pstCmd->fMaxX, pstCmd->fMinY, pstCmd->fMaxY};
    cv::Mat matXX = CalcUtils::generateBezier(matX, matY, vecParamMinMaxXY, pstCmd->nBezierRank, pstCmd->nBezierRank);

    cv::Mat matXOffset = matXX * pstCmd->matXOffsetParam;
    cv::Mat matYOffset = matXX * pstCmd->matYOffsetParam;

    float fOffsetX = matXOffset.at<float>(0);
    float fOffsetY = matYOffset.at<float>(0);
    return cv::Size2f(fOffsetX, fOffsetY);
}

/*static*/ VisionStatus TableMapping::calcTableOffsetNew(const PR_CALC_TABLE_OFFSET_CMD* const pstCmd, PR_CALC_TABLE_OFFSET_RPY* const pstRpy) {
    auto szOffset1 = _calcPointOffset(pstCmd, pstCmd->ptTablePos);

    // Calculate back to physical coordinate
    float xt2 = pstCmd->a * pstCmd->ptTablePos.x + pstCmd->c * pstCmd->ptTablePos.y + szOffset1.width;
    float yt2 = pstCmd->b * pstCmd->ptTablePos.y + szOffset1.height;

    auto szOffset3 = _calcPointOffset(pstCmd, cv::Point2f(xt2, yt2));

    float xt4 = xt2 - szOffset3.width;
    float yt4 = yt2 - szOffset3.height;

    cv::Mat matA = cv::Mat::eye(3, 3, CV_32FC1);
    matA.at<float>(0, 0) = pstCmd->a;
    matA.at<float>(0, 1) = pstCmd->c;
    matA.at<float>(1, 1) = pstCmd->b;
    cv::Mat matA1 = matA.inv();

    auto a1 = matA1.at<float>(0, 0);
    auto b1 = matA1.at<float>(1, 1);
    auto c1 = matA1.at<float>(0, 1);

    float xt5 = a1*xt4 + c1*yt4;
    float yt5 = b1*yt4;
    pstRpy->fOffsetX = xt5 - pstCmd->ptTablePos.x;
    pstRpy->fOffsetY = yt5 - pstCmd->ptTablePos.y;

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

/*static*/ VisionStatus TableMapping::calcTableOffsetNew1(const PR_CALC_TABLE_OFFSET_CMD* const pstCmd, PR_CALC_TABLE_OFFSET_RPY* const pstRpy) {
    pstRpy->fOffsetX = 0.f;
    pstRpy->fOffsetY = 0.f;

    if (pstCmd->ptTablePos.x < pstCmd->fMinX || pstCmd->ptTablePos.x > pstCmd->fMaxX ||
        pstCmd->ptTablePos.y < pstCmd->fMinY || pstCmd->ptTablePos.y > pstCmd->fMaxY) {
        std::stringstream ss;
        ss << "CalcTableOffset: the input position " << pstCmd->ptTablePos << " is not within calibration range X [" << pstCmd->fMinX << ", "
           << pstCmd->fMaxX << "], Y [" << pstCmd->fMinY << ", " << pstCmd->fMaxY << "]";
        WriteLog(ss.str());
        pstRpy->enStatus = VisionStatus::INVALID_PARAM;
        return pstRpy->enStatus;
    }

    auto szOffset1 = _calcPointOffset(pstCmd, pstCmd->ptTablePos);

    cv::Mat matA = cv::Mat::eye(3, 3, CV_32FC1);
    matA.at<float>(0, 0) = pstCmd->a;
    matA.at<float>(0, 1) = pstCmd->c;
    matA.at<float>(1, 1) = pstCmd->b;
    cv::Mat matA1 = matA.inv();

    auto a1 = matA1.at<float>(0, 0);
    auto b1 = matA1.at<float>(1, 1);
    auto c1 = matA1.at<float>(0, 1);

    // Calculate back to physical coordinate
    float xt2 = a1 * (pstCmd->ptTablePos.x - szOffset1.width) + c1 * (pstCmd->ptTablePos.y - szOffset1.height);
    float yt2 = b1 * (pstCmd->ptTablePos.y - szOffset1.height);

    pstRpy->fOffsetX = xt2 - pstCmd->ptTablePos.x;
    pstRpy->fOffsetY = yt2 - pstCmd->ptTablePos.y;

    pstRpy->enStatus = VisionStatus::OK;
    return pstRpy->enStatus;
}

}
}
