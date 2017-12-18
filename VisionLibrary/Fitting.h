#ifndef _AOI_VISION_FITTING_H_
#define _AOI_VISION_FITTING_H_

#include <vector>
#include "opencv2/core.hpp"
#include "VisionStruct.h"

namespace AOI
{
namespace Vision
{

class Fitting
{
public:
    // This function is based on Modified Least Square Methods from Paper
    // "A Few Methods for Fitting Circles to Data".
    template<typename T>
    static cv::RotatedRect fitCircle ( const T &points ) {
        cv::RotatedRect rotatedRect;
        if( points.size () < 3 )
            return rotatedRect;

        float fAverageX = 0, fAverageY = 0;
        float n = ToFloat ( points.size () );
        for( const auto &point : points ) {
            fAverageX += point.x / n;
            fAverageY += point.y / n;
        }

        //Here offset the input points with its average value, to prevent the intermediate value becomes too big, even over
        //the range of std::numenic_limits<double>::max(). Later in the end, the average value will add back.
        VectorOfPoint2f vecPointsClone;
        vecPointsClone.reserve ( points.size() );
        for( auto &point : points ) {
            cv::Point2f ptNew ( point );
            ptNew.x -= fAverageX;
            ptNew.y -= fAverageY;
            vecPointsClone.push_back ( ptNew );
        }

        double Sx = 0., Sy = 0., Sxx = 0., Sx2 = 0., Sy2 = 0., Sxy = 0., Syy = 0., Sx3 = 0., Sy3 = 0., Sxy2 = 0., Syx2 = 0.;
        for( const auto &point : vecPointsClone ) {
            Sx += point.x;
            Sy += point.y;
            Sx2 += point.x * point.x;
            Sy2 += point.y * point.y;
            Sxy += point.x * point.y;
            Sx3 += point.x * point.x * point.x;
            Sy3 += point.y * point.y * point.y;
            Sxy2 += point.x * point.y * point.y;
            Syx2 += point.y * point.x * point.x;
        }

        double A, B, C, D, E;
        A = n * Sx2 - Sx * Sx;
        B = n * Sxy - Sx * Sy;
        C = n * Sy2 - Sy * Sy;
        D = 0.5 * (n * Sxy2 - Sx * Sy2 + n * Sx3 - Sx * Sx2);
        E = 0.5 * (n * Syx2 - Sy * Sx2 + n * Sy3 - Sy * Sy2);

        auto AC_B2 = (A * C - B * B);  // The variable name is from AC - B^2
        auto am = (D * C - B * E) / AC_B2;
        auto bm = (A * E - B * D) / AC_B2;

        double rSqureSum = 0.f;
        for( const auto &point : vecPointsClone )
            rSqureSum += sqrt ( (point.x - am) * (point.x - am) + (point.y - bm) * (point.y - bm) );

        float r = static_cast<float>(rSqureSum / n);
        rotatedRect.center.x = static_cast<float>(am) + fAverageX;
        rotatedRect.center.y = static_cast<float>(bm) + fAverageY;
        rotatedRect.size = cv::Size2f ( 2.f * r, 2.f * r );
        return rotatedRect;
    }

    template<typename T>
    static T randomSelectPoints ( const T &vecPoints, int numOfPtToSelect ) {
        std::set<int> setResult;
        VectorOfPoint vecResultPoints;

        if ( ToInt32 ( vecPoints.size () ) < numOfPtToSelect)
            return vecResultPoints;

        while ( ToInt32 ( setResult.size () ) < numOfPtToSelect) {
            int nValue = std::rand () % vecPoints.size ();
            setResult.insert ( nValue );
        }
        for (auto index : setResult)
            vecResultPoints.push_back ( vecPoints[index] );
        return vecResultPoints;
    }

    template<typename T>
    static T findPointsInCircleTol ( const T &vecPoints, const cv::RotatedRect &rotatedRect, float tolerance ) {
        T vecResult;
        for (const auto &point : vecPoints) {
            auto disToCtr = sqrt ( (point.x - rotatedRect.center.x) * (point.x - rotatedRect.center.x) + (point.y - rotatedRect.center.y) * (point.y - rotatedRect.center.y) );
            auto err = disToCtr - rotatedRect.size.width / 2;
            if (fabs ( err ) < tolerance) {
                vecResult.push_back ( point );
            }
        }
        return vecResult;
    }

    /* The ransac algorithm is from paper "Random Sample Consensus: A Paradigm for Model Fitting with Apphcatlons to Image Analysis and Automated Cartography".
    *  Given a model that requires a minimum of n data points to instantiate
    * its free parameters, and a set of data points P such that the number of
    * points in P is greater than n, randomly select a subset S1 of
    * n data points from P and instantiate the model. Use the instantiated
    * model M1 to determine the subset S1* of points in P that are within
    * some error tolerance of M1. The set S1* is called the consensus set of
    * S1.
    * If g (S1*) is greater than some threshold t, which is a function of the
    * estimate of the number of gross errors in P, use S1* to compute
    * (possibly using least squares) a new model M1* and return.
    * If g (S1*) is less than t, randomly select a new subset S2 and repeat the
    * above process. If, after some predetermined number of trials, no
    * consensus set with t or more members has been found, either solve the
    * model with the largest consensus set found, or terminate in failure. */
    template<typename T>
    static cv::RotatedRect fitCircleRansac ( const T &vecPoints, float tolerance, int maxRansacTime, size_t numOfPtToFinish ) {
        cv::RotatedRect fitResult;
        if (vecPoints.size () < 3)
            return fitResult;

        int nRansacTime = 0;
        const int RANSAC_CIRCLE_POINT = 3;
        size_t nMaxConsentNum = 0;

        while (nRansacTime < maxRansacTime) {
            VectorOfPoint vecSelectedPoints = randomSelectPoints ( vecPoints, RANSAC_CIRCLE_POINT );
            cv::RotatedRect rectReult = Fitting::fitCircle ( vecSelectedPoints );
            vecSelectedPoints = findPointsInCircleTol ( vecPoints, rectReult, tolerance );

            if (vecSelectedPoints.size () >= numOfPtToFinish)
                return Fitting::fitCircle ( vecSelectedPoints );

            if (vecSelectedPoints.size () > nMaxConsentNum) {
                fitResult = Fitting::fitCircle ( vecSelectedPoints );
                nMaxConsentNum = vecSelectedPoints.size ();
            }
            ++ nRansacTime;
        }
        return fitResult;
    }

    template<typename T, >
    static std::vector<ListOfPoint::const_iterator> findPointsOverCircleTol( const ListOfPoint &listPoint, const cv::RotatedRect &rotatedRect, PR_RM_FIT_NOISE_METHOD enMethod, float tolerance )
    {
        std::vector<ListOfPoint::const_iterator> vecResult;
        for (ListOfPoint::const_iterator it = listPoint.begin (); it != listPoint.end (); ++ it) {
            cv::Point2f point = *it;
            auto disToCtr = sqrt ( (point.x - rotatedRect.center.x) * (point.x - rotatedRect.center.x) + (point.y - rotatedRect.center.y) * (point.y - rotatedRect.center.y) );
            auto err = disToCtr - rotatedRect.size.width / 2;
            if (PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR == enMethod && fabs ( err ) > tolerance)
                vecResult.push_back ( it );
            else if (PR_RM_FIT_NOISE_METHOD::POSITIVE_ERR == enMethod && err > tolerance)
                vecResult.push_back ( it );
            else if (PR_RM_FIT_NOISE_METHOD::NEGATIVE_ERR == enMethod && err < -tolerance)
                vecResult.push_back ( it );
        }
        return vecResult;
    }

    //method 1 : Exclude all the points out of positive error tolerance and inside the negative error tolerance.
    //method 2 : Exclude all the points out of positive error tolerance.
    //method 3 : Exclude all the points inside the negative error tolerance.
    template<typename T>
    static cv::RotatedRect VisionAlgorithm::fitCircleIterate ( const T &vecPoints, PR_RM_FIT_NOISE_METHOD method, float tolerance ) {
        cv::RotatedRect rotatedRect;
        if (vecPoints.size () < 3)
            return rotatedRect;

        ListOfPoint listPoint;
        for (const auto &point : vecPoints)
            listPoint.push_back ( point );

        rotatedRect = fitCircle ( listPoint );

        std::vector<ListOfPoint::const_iterator> overTolPoints = _findPointsOverCircleTol ( listPoint, rotatedRect, method, tolerance );
        int nIteratorNum = 1;
        while (!overTolPoints.empty () && nIteratorNum < 20) {
            for (const auto &it : overTolPoints)
                listPoint.erase ( it );
            rotatedRect = Fitting::fitCircle ( listPoint );
            overTolPoints = _findPointsOverCircleTol ( listPoint, rotatedRect, method, tolerance );
            ++nIteratorNum;
        }
        return rotatedRect;
    }

    static void fitLine(const VectorOfPoint &vecPoints, float &fSlope, float &fIntercept, bool reverseFit = false);
    static void fitLine(const ListOfPoint &listPoint, float &fSlope, float &fIntercept, bool reverseFit = false);
    static void fitParallelLine(const VectorOfPoint &vecPoints1,
                                const VectorOfPoint &vecPoints2,
                                float               &fSlope,
                                float               &fIntercept1,
                                float               &fIntercept2,
                                bool                 reverseFit = false);
    static void fitParallelLine(const ListOfPoint &listPoints1,
                                const ListOfPoint &listPoints2,
                                float             &fSlope,
                                float             &fIntercept1,
                                float             &fIntercept2,
                                bool               reverseFit = false);
    static void fitRect(VectorOfVectorOfPoint &vecVecPoint, 
                        float                 &fSlope1, 
                        float                 &fSlope2, 
                        std::vector<float>    &vecIntercept,
                        bool                   bLineOneReversedFit,
                        bool                   bLineTwoReversedFit);
    static void fitRect(VectorOfListOfPoint &vecListPoint, 
                        float               &fSlope1, 
                        float               &fSlope2, 
                        std::vector<float>  &vecIntercept,
                        bool                 bLineOneReversedFit,
                        bool                 bLineTwoReversedFit);
};

}
}

#endif