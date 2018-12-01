#include "TableMapping.h"
#include "CalcUtils.h"

namespace AOI
{
namespace Vision
{
TableMapping::TableMapping()
{
}


TableMapping::~TableMapping()
{
}

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
    const cv::Mat& matD1, cv::Mat& matR1, cv::Mat& matDR1) {
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
        std::vector<float> vecActualX, vecActualY;
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

    matR1 = Axy - Axy1;
    //cv::transpose(matR1, matR1);
    matR1 = matR1.reshape(1, matR1.total());
    float normR1 = ToFloat(cv::norm(matR1));

    /*cv::transpose(dRa, dRa);*/ dRa = dRa.reshape(1, 1);
    /*cv::transpose(dRb, dRb);*/ dRb = dRb.reshape(1, 1);
    /*cv::transpose(dRc, dRc);*/ dRc = dRc.reshape(1, 1);
    /*cv::transpose(dRTxy, dRTxy);*/ dRTxy = dRTxy.reshape(1, 3 * ToInt32(vecFramePoints.size()));

    matDR1.release();
    matDR1.push_back(dRa);
    matDR1.push_back(dRb);
    matDR1.push_back(dRc);
    matDR1.push_back(dRTxy);
    cv::transpose(matDR1, matDR1);
    matDR1 = -matDR1;

    return normR1;
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

        auto theta1 = atan(matK1.at<float>(1, 0) / matK1.at<float>(0, 0));
        auto theta2 = atan(-matK1.at<float>(0, 1) / matK1.at<float>(1, 1)) - theta1;

        auto Tx = matK.at<float>(2, 0);
        auto Ty = matK.at<float>(2, 1);

        vecVecTransM.emplace_back(VectorOfFloat{k1, k2, theta2, theta1, Tx, Ty});
    }

    cv::Mat matTransM = CalcUtils::vectorToMat(vecVecTransM);
    cv::Mat matMeanTransM;
    cv::reduce(matTransM, matMeanTransM, 0, cv::ReduceTypes::REDUCE_AVG);
    auto k1 = matMeanTransM.at<float>(0);
    auto k2 = matMeanTransM.at<float>(1);
    auto theta2 = matMeanTransM.at<float>(2);

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
    VectorOfFloat dd{0.f, 0.f, 0.f, 0.001f, -5.34f, 6.79f, -0.01f, 20.8f, 4.9f};
    for (size_t i = 0; i < dd.size(); ++ i)
        vecD1[i] += dd[i];
    cv::Mat matD1(vecD1);

    cv::Mat matR1, matDR1;
    float Rsum1 = _calcTransR1RSum(vecProcessedFramePoints, matD1, matR1, matDR1);

#ifdef _DEBUG
    cv::Mat matR1T; cv::transpose(matR1, matR1T);
    auto vecVecR1T = CalcUtils::matToVector<float>(matR1T);
#endif

    cv::Mat derivResidual = cv::Mat::ones(vecD1.size(), 1, CV_32FC1);
    int N  = 0;
    float mu = 1e-6;
    auto eyeX = cv::Mat::eye(vecD1.size(), vecD1.size(), CV_32FC1);
    float fDrNorm = cv::norm(derivResidual);
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
        fDrNorm = cv::norm(derivResidual);
        derivResidual = -derivResidual;

#ifdef _DEBUG
        cv::Mat matDrT; cv::transpose(derivResidual, matDrT);
        auto vecVecDrT = CalcUtils::matToVector<float>(matDrT);
#endif
      
        if (++ N > 100)
            break;

        matD1 = matD1 + derivResidual;
        Rsum1 = _calcTransR1RSum(vecProcessedFramePoints, matD1, matR1, matDR1);

        int nn = 0;
        bool flag = false;
        fDrNorm = cv::norm(derivResidual);        

        while (Rsum1 > Rsum && fDrNorm > EPSILON) {
            if (++nn > 10) {
                flag = true;
                break;
            }

            mu = mu*2;
            cv::Mat matXX = JrT * Jr + mu * eyeX;
            cv::Mat matYY = JrT * Rt;
            cv::solve(matXX, matYY, derivResidual);
            fDrNorm = cv::norm(derivResidual);
            derivResidual = -derivResidual;

            matD1 = matD1 + derivResidual;
            Rsum1 = _calcTransR1RSum(vecProcessedFramePoints, matD1, matR1, matDR1);
        }

        if (flag)
            break;
    }

    auto d = matD1;

#ifdef _DEBUG
    cv::Mat matD1T; cv::transpose(matD1, matD1T);
    auto vecVecD1T = CalcUtils::matToVector<float>(matD1T);
#endif

    return VisionStatus::OK;
}

}
}
