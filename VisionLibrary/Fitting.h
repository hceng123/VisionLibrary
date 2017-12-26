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
    static void fitParallelCircle ( const T &vecPoints1, const T &vecPoints2, cv::Point2f &ptCenter, float &fRadius1, float &fRadius2 ) {
        auto vecCombine ( vecPoints1 );
        vecCombine.insert ( vecCombine.end (), vecPoints2.begin (), vecPoints2.end () );
        auto fitTotal = fitCircle ( vecCombine );
        ptCenter = fitTotal.center;

        double rSqureSum = 0.f;
        for (const auto &point : vecPoints1)
            rSqureSum += sqrt ( (point.x - ptCenter.x) * (point.x - ptCenter.x) + (point.y - ptCenter.y) * (point.y - ptCenter.y) );
        fRadius1 = static_cast<float>( rSqureSum / vecPoints1.size () );

        rSqureSum = 0.f;
        for (const auto &point : vecPoints2)
            rSqureSum += sqrt ( (point.x - fitTotal.center.x) * (point.x - fitTotal.center.x) + (point.y - fitTotal.center.y) * (point.y - fitTotal.center.y) );
        fRadius2 = static_cast<float> ( rSqureSum / vecPoints2.size () );
    }

    template<typename T>
    static T randomSelectPoints ( const T &vecPoints, int numOfPtToSelect ) {
        std::set<int> setResult;
        T vecResultPoints;

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
            auto vecSelectedPoints = randomSelectPoints ( vecPoints, RANSAC_CIRCLE_POINT );
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

    template<typename Iter>
    static std::vector<Iter> findPointsIterOverCircleTol( Iter itBegin, Iter itEnd, const cv::RotatedRect &rotatedRect, PR_RM_FIT_NOISE_METHOD enMethod, float tolerance )
    {
        std::vector<Iter> vecResult;
        //std::vector<ListOfPoint::const_iterator> vecResult;
        for ( auto it = itBegin; it != itEnd; ++ it) {
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
    static cv::RotatedRect fitCircleIterate ( const T &vecPoints, PR_RM_FIT_NOISE_METHOD method, float tolerance ) {
        cv::RotatedRect rotatedRect;
        if (vecPoints.size () < 3)
            return rotatedRect;

        ListOfPoint listPoint;
        for (const auto &point : vecPoints)
            listPoint.push_back ( point );

        rotatedRect = fitCircle ( listPoint );

        std::vector<ListOfPoint::const_iterator> overTolPoints = findPointsIterOverCircleTol ( listPoint.cbegin(), listPoint.cend(), rotatedRect, method, tolerance );
        int nIteratorNum = 1;
        while (!overTolPoints.empty () && nIteratorNum < 20) {
            for (const auto &it : overTolPoints)
                listPoint.erase ( it );
            rotatedRect = fitCircle ( listPoint );
            overTolPoints = findPointsIterOverCircleTol ( listPoint.cbegin(), listPoint.cend(), rotatedRect, method, tolerance );
            ++ nIteratorNum;
        }
        return rotatedRect;
    }

    template<typename T>
    static cv::RotatedRect fitCircleRemoveStray ( const T &vecPoints, float fRmStrayPointRatio ) {
        cv::RotatedRect rotatedRect;
        if (vecPoints.size () < 3)
            return rotatedRect;

        rotatedRect = fitCircle ( vecPoints );

        if ( fRmStrayPointRatio > 0 ) {
            int nPointsToKeep = ToInt32 ( vecPoints.size() * ( 1.f - fRmStrayPointRatio ) );
            VectorOfFloat vecRadiusDiff;
            vecRadiusDiff.reserve ( vecPoints.size() );
            for ( const auto &point : vecPoints ) {
                float fRadiusDiff = fabs ( CalcUtils::distanceOf2Point ( point, rotatedRect.center ) - rotatedRect.size.width / 2.f );
                vecRadiusDiff.push_back ( fRadiusDiff );
            }
            auto vecIndex = CalcUtils::sort_index ( vecRadiusDiff );
            T vecKeepPoints;
            vecKeepPoints.reserve ( nPointsToKeep );
            for ( int i = 0; i < nPointsToKeep; ++ i ) {
                vecKeepPoints.push_back ( vecPoints[ vecIndex[i] ] );
            }
            rotatedRect = fitCircle ( vecKeepPoints );  //Fit again use the left over points
        }
        return rotatedRect;
    }

    template<typename T>
    static void fitParallelCircleRemoveStray ( const T &vecPoints1, const T &vecPoints2, T &vecKeepPoints1, T &vecKeepPoints2, cv::Point2f &ptCenter, float &fRadius1, float &fRadius2, float fRmStrayPointRatio ) {
        fitParallelCircle ( vecPoints1, vecPoints2, ptCenter, fRadius1, fRadius2 );
        if ( fRmStrayPointRatio > 0 ) {
            {
                int nPointsToKeep = ToInt32 ( vecPoints1.size () * (1.f - fRmStrayPointRatio) );
                VectorOfFloat vecRadiusDiff;
                vecRadiusDiff.reserve ( vecPoints1.size () );
                for( const auto &point : vecPoints1 ) {
                    float fRadiusDiff = fabs ( CalcUtils::distanceOf2Point ( point, ptCenter ) - fRadius1 );
                    vecRadiusDiff.push_back ( fRadiusDiff );
                }
                auto vecIndex = CalcUtils::sort_index ( vecRadiusDiff );
                vecKeepPoints1.reserve ( nPointsToKeep );
                for( int i = 0; i < nPointsToKeep; ++ i ) {
                    vecKeepPoints1.push_back ( vecPoints1[ vecIndex[i] ] );
                }
            }

            {
                int nPointsToKeep = ToInt32 ( vecPoints2.size () * (1.f - fRmStrayPointRatio) );
                VectorOfFloat vecRadiusDiff;
                vecRadiusDiff.reserve ( vecPoints2.size () );
                for( const auto &point : vecPoints2 ) {
                    float fRadiusDiff = fabs ( CalcUtils::distanceOf2Point ( point, ptCenter ) - fRadius2 );
                    vecRadiusDiff.push_back ( fRadiusDiff );
                }
                auto vecIndex = CalcUtils::sort_index ( vecRadiusDiff );

                vecKeepPoints2.reserve ( nPointsToKeep );
                for( int i = 0; i < nPointsToKeep; ++ i ) {
                    vecKeepPoints2.push_back ( vecPoints2[ vecIndex[i] ] );
                }
            }
            fitParallelCircle ( vecKeepPoints1, vecKeepPoints2, ptCenter, fRadius1, fRadius2 );
        }
    }

    //The equation is from http://hotmath.com/hotmath_help/topics/line-of-best-fit.html
    template<typename T>
    static void fitLine ( const T &vecPoints, float &fSlope, float &fIntercept, bool reverseFit = false ) {
        if( vecPoints.size () < 2 )
            return;

        double Sx = 0., Sy = 0., Sx2 = 0., Sy2 = 0., Sxy = 0.;
        for( const auto &point : vecPoints )   {
            Sx += point.x;
            Sy += point.y;
            Sx2 += point.x * point.x;
            Sy2 += point.y * point.y;
            Sxy += point.x * point.y;
        }
        size_t n = vecPoints.size ();
        if( reverseFit )   {
            fSlope = static_cast<float>((n * Sxy - Sx * Sy) / (n * Sy2 - Sy * Sy));
            fIntercept = static_cast<float>((Sx * Sy2 - Sy * Sxy) / (n * Sy2 - Sy * Sy));
        } else {
            fSlope = static_cast<float>((n * Sxy - Sx * Sy) / (n * Sx2 - Sx * Sx));
            fIntercept = static_cast<float>((Sy * Sx2 - Sx * Sxy) / (n * Sx2 - Sx * Sx));
        }
    }

    template<typename T>
    static void fitLineRansac(const T  &vecPoints,
                              float     fTolerance,
                            int                  nMaxRansacTime,
                            size_t               nFinishThreshold,
                            bool                 bReversedFit,
                            float               &fSlope,
                            float               &fIntercept,
                            PR_Line2f           &stLine )
    {
        int nRansacTime = 0;
        const int LINE_RANSAC_POINT = 2;
        size_t nMaxConsentNum = 0;

        while (nRansacTime < nMaxRansacTime) {
            auto vecSelectedPoints = randomSelectPoints ( vecPoints, LINE_RANSAC_POINT );
            Fitting::fitLine ( vecSelectedPoints, fSlope, fIntercept, bReversedFit );
            vecSelectedPoints = findPointInLineTol ( vecPoints, bReversedFit, fSlope, fIntercept, fTolerance );

            if ( vecSelectedPoints.size () >= nFinishThreshold ) {
                Fitting::fitLine ( vecSelectedPoints, fSlope, fIntercept, bReversedFit );
                stLine = CalcUtils::calcEndPointOfLine ( vecSelectedPoints, bReversedFit, fSlope, fIntercept );
                return;
            }

            if (vecSelectedPoints.size () > nMaxConsentNum) {
                Fitting::fitLine ( vecSelectedPoints, fSlope, fIntercept, bReversedFit );
                stLine = CalcUtils::calcEndPointOfLine ( vecSelectedPoints, bReversedFit, fSlope, fIntercept );
                nMaxConsentNum = vecSelectedPoints.size ();
            }
            ++nRansacTime;
        }
    }

    template<typename T>
    static VisionStatus fitLineRefine(const T                &vecPoints,
                                      PR_RM_FIT_NOISE_METHOD  enRmNoiseMethod,
                                      float                   fTolerance,
                                      bool                    bReversedFit,
                                      float                  &fSlope,
                                      float                  &fIntercept,
                                      PR_Line2f              &stLine )
    {
        ListOfPoint2f listPoint;
        for (const auto &point : vecPoints)
            listPoint.push_back ( point );

        int nIteratorNum = 0;
        std::vector<ListOfPoint2f::const_iterator> vecOverTolIter;
        do {
            for (const auto &it : vecOverTolIter)
                listPoint.erase ( it );

            Fitting::fitLine ( listPoint, fSlope, fIntercept, bReversedFit );
            vecOverTolIter = findPointIterOverLineTol ( listPoint.cbegin(), listPoint.cend(), bReversedFit, fSlope, fIntercept, enRmNoiseMethod, fTolerance );
        } while (!vecOverTolIter.empty () && nIteratorNum < 20);

        if ( listPoint.size () < 2 )
            return VisionStatus::TOO_MUCH_NOISE_TO_FIT;

        stLine = CalcUtils::calcEndPointOfLine ( listPoint, bReversedFit, fSlope, fIntercept );
        return VisionStatus::OK;
    }

    template<typename T>
    static T findPointInLineTol(const T &   vecPoint,
                                bool        bReversedFit,
                                const float fSlope,
                                const float fIntercept,
                                float       fTolerance )
    {
        T vecPointInTol;
        for (const auto &point : vecPoint) {
            auto disToLine = CalcUtils::ptDisToLine ( point, bReversedFit, fSlope, fIntercept );
            if (fabs ( disToLine ) <= fTolerance)
                vecPointInTol.push_back ( point );
        }
        return vecPointInTol;
    }

    template<typename T>
    static std::vector<T> findPointIterOverLineTol( T                        begin,
                                                    T                        end,
                                                    bool                     bReversedFit,
                                                    const float              fSlope,
                                                    const float              fIntercept,
                                                    PR_RM_FIT_NOISE_METHOD   method,
                                                    float                    tolerance )
    {
        std::vector<T> vecResult;
        for (auto it = begin; it != end; ++it)  {
            cv::Point2f point = *it;
            auto disToLine = CalcUtils::ptDisToLine ( point, bReversedFit, fSlope, fIntercept );
            if (PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR == method && fabs ( disToLine ) > tolerance)  {
                vecResult.push_back ( it );
            }
            else if (PR_RM_FIT_NOISE_METHOD::POSITIVE_ERR == method && disToLine > tolerance) {
                vecResult.push_back ( it );
            }
            else if (PR_RM_FIT_NOISE_METHOD::NEGATIVE_ERR == method && disToLine < -tolerance) {
                vecResult.push_back ( it );
            }
        }
        return vecResult;
    }

    template<typename T>
    static void fitParallelLine ( const T &listPoints1,
                                  const T &listPoints2,
                                  float   &fSlope,
                                  float   &fIntercept1,
                                  float   &fIntercept2,
                                  bool     reverseFit )
    {
        if( listPoints1.size () < 2 || listPoints2.size () < 2 )
            return;

        auto totalDataCount = static_cast<int>(listPoints1.size () + listPoints2.size ());
        auto dataCount1 = listPoints1.size ();
        cv::Mat B ( totalDataCount, 1, CV_32FC1 );
        cv::Mat A = cv::Mat::zeros ( totalDataCount, 3, CV_32FC1 );
        ListOfPoint::const_iterator it1 = listPoints1.begin ();
        ListOfPoint::const_iterator it2 = listPoints2.begin ();
        for( int i = 0; i < totalDataCount; ++i )  {
            if( i < ToInt32 ( listPoints1.size () ) ) {
                if( reverseFit )
                    A.at<float> ( i, 0 ) = ToFloat ( (*it1).y );
                else
                    A.at<float> ( i, 0 ) = ToFloat ( (*it1).x );

                A.at<float> ( i, 1 ) = 1.f;
                if( reverseFit )
                    B.at<float> ( i, 0 ) = ToFloat ( (*it1).x );
                else
                    B.at<float> ( i, 0 ) = ToFloat ( (*it1).y );

                ++ it1;
            }else {
                if( reverseFit )
                    A.at<float> ( i, 0 ) = ToFloat ( (*it2).y );
                else
                    A.at<float> ( i, 0 ) = ToFloat ( (*it2).x );
                A.at<float> ( i, 2 ) = 1.f;
                if( reverseFit )
                    B.at<float> ( i, 0 ) = ToFloat ( (*it2).x );
                else
                    B.at<float> ( i, 0 ) = ToFloat ( (*it2).y );

                ++ it2;
            }
        }

        cv::Mat matResultImg;
        bool bResult = cv::solve ( A, B, matResultImg, cv::DECOMP_SVD );

        fSlope = matResultImg.at<float> ( 0, 0 );
        fIntercept1 = matResultImg.at<float> ( 1, 0 );
        fIntercept2 = matResultImg.at<float> ( 2, 0 );
    }

    template<typename T>
    static void fitRect(T &vecListPoint,
                        float                 &fSlope1, 
                        float                 &fSlope2,
                        std::vector<float>    &vecIntercept,
                        bool                   bLineOneReversedFit,
                        bool                   bLineTwoReversedFit)
    {
        assert ( vecListPoint.size () == 4 );  //The input should has 4 lines
        vecIntercept.clear ();
        auto totalRows = 0;
        for( const auto &vectorPoint : vecListPoint )
            totalRows += ToInt32 ( vectorPoint.size () );
        cv::Mat A = cv::Mat::zeros ( totalRows, 5, CV_32FC1 );
        cv::Mat B = cv::Mat::zeros ( totalRows, 1, CV_32FC1 );

        int index = 0;
        for( int i = 0; i < ToInt32 ( vecListPoint.size () ); ++i )
        {
            ListOfPoint listPoint = vecListPoint[i];
            for( const auto &point : listPoint )
            {
                A.at<float> ( index, i + 1 ) = 1.f;
                if( i < 2 ) {
                    if( bLineOneReversedFit )   {
                        A.at<float> ( index, 0 ) = ToFloat ( point.y );
                        B.at<float> ( index, 0 ) = ToFloat ( point.x );
                    } else
                    {
                        A.at<float> ( index, 0 ) = ToFloat ( point.x );
                        B.at<float> ( index, 0 ) = ToFloat ( point.y );
                    }
                }else {
                    if( bLineOneReversedFit )   {
                        A.at<float> ( index, 0 ) = ToFloat ( -point.x );
                        B.at<float> ( index, 0 ) = ToFloat ( point.y );
                    } else {
                        A.at<float> ( index, 0 ) = ToFloat ( -point.y );
                        B.at<float> ( index, 0 ) = ToFloat ( point.x );
                    }
                }
                ++ index;
            }
        }

        cv::Mat matResultImg;
        bool bResult = cv::solve ( A, B, matResultImg, cv::DECOMP_QR );

        fSlope1 = matResultImg.at<float> ( 0, 0 );
        fSlope2 = -fSlope1;
        for( int i = 0; i < 4; ++i )
            vecIntercept.push_back ( matResultImg.at < float > ( i + 1, 0 ) );
    }
};

}
}

#endif