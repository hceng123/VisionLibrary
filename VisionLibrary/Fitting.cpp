#include "Fitting.h"
#include "CalcUtils.h"

namespace AOI
{
namespace Vision
{

// This function is based on Modified Least Square Methods from Paper
// "A Few Methods for Fitting Circles to Data".
/*static*/ cv::RotatedRect Fitting::fitCircle(const std::vector<cv::Point2f> &vecPoints)
{
    cv::RotatedRect rotatedRect;
    if ( vecPoints.size() < 3 )
        return rotatedRect;

    double Sx = 0., Sy = 0., Sxx = 0., Sx2 = 0., Sy2 = 0., Sxy = 0., Syy = 0., Sx3 = 0., Sy3 = 0., Sxy2 = 0., Syx2 = 0.;
    for ( const auto &point : vecPoints )   {
        Sx   += point.x;
        Sy   += point.y;
        Sx2  += point.x * point.x;
        Sy2  += point.y * point.y;
        Sxy  += point.x * point.y;
        Sx3  += point.x * point.x * point.x;
        Sy3  += point.y * point.y * point.y;
        Sxy2 += point.x * point.y * point.y;
        Syx2 += point.y * point.x * point.x;
    }

    double A, B, C, D, E;
    size_t n = vecPoints.size();
    A = n * Sx2 - Sx * Sx;
    B = n * Sxy - Sx * Sy;
    C = n * Sy2 - Sy * Sy;
    D = 0.5 * ( n * Sxy2 - Sx * Sy2 + n * Sx3 - Sx * Sx2 );
    E = 0.5 * ( n * Syx2 - Sy * Sx2 + n * Sy3 - Sy * Sy2 );

    auto AC_B2 = ( A * C - B * B);  // The variable name is from AC - B^2
    auto am = ( D * C - B * E ) / AC_B2;
    auto bm = ( A * E - B * D ) / AC_B2;

    double rSqureSum = 0.f;
    for ( const auto &point : vecPoints )
    {
        rSqureSum += sqrt ( ( point.x - am ) * ( point.x - am ) + ( point.y - bm) * ( point.y - bm) );
    }
    float r = static_cast<float>( rSqureSum / n );
    rotatedRect.center.x = static_cast<float>( am );
    rotatedRect.center.y = static_cast<float>( bm );
    rotatedRect.size = cv::Size2f( 2.f * r,  2.f * r  );

    return rotatedRect;
}

/*static*/ cv::RotatedRect Fitting::fitCircle(const ListOfPoint &listPoint)
{
    cv::RotatedRect rotatedRect;
    if ( listPoint.size() < 3 )
        return rotatedRect;

    double Sx = 0., Sy = 0., Sxx = 0., Sx2 = 0., Sy2 = 0., Sxy = 0., Syy = 0., Sx3 = 0., Sy3 = 0., Sxy2 = 0., Syx2 = 0.;
    for ( const auto &point : listPoint )   {
        Sx   += point.x;
        Sy   += point.y;
        Sx2  += point.x * point.x;
        Sy2  += point.y * point.y;
        Sxy  += point.x * point.y;
        Sx3  += point.x * point.x * point.x;
        Sy3  += point.y * point.y * point.y;
        Sxy2 += point.x * point.y * point.y;
        Syx2 += point.y * point.x * point.x;
    }

    double A, B, C, D, E;
    size_t n = listPoint.size();
    A = n * Sx2 - Sx * Sx;
    B = n * Sxy - Sx * Sy;
    C = n * Sy2 - Sy * Sy;
    D = 0.5 * ( n * Sxy2 - Sx * Sy2 + n * Sx3 - Sx * Sx2 );
    E = 0.5 * ( n * Syx2 - Sy * Sx2 + n * Sy3 - Sy * Sy2 );

    auto AC_B2 = ( A * C - B * B);  // The variable name is from AC - B^2
    auto am = ( D * C - B * E ) / AC_B2;
    auto bm = ( A * E - B * D ) / AC_B2;

    double rSqureSum = 0.f;
    for ( const auto &point : listPoint )
    {
        rSqureSum += sqrt ( ( point.x - am ) * ( point.x - am ) + ( point.y - bm) * ( point.y - bm) );
    }
    float r = static_cast<float>( rSqureSum / n );
    rotatedRect.center.x = static_cast<float>( am );
    rotatedRect.center.y = static_cast<float>( bm );
    rotatedRect.size = cv::Size2f( 2.f * r,  2.f * r  );

    return rotatedRect;
}

//The equation is from http://hotmath.com/hotmath_help/topics/line-of-best-fit.html
/*static*/ void Fitting::fitLine(const std::vector<cv::Point2f> &vecPoints, float &fSlope, float &fIntercept, bool reverseFit)
{
    if ( vecPoints.size() < 2 )
        return;

    double Sx = 0., Sy = 0., Sx2 = 0., Sy2 = 0., Sxy = 0.;
    for ( const auto &point : vecPoints )   {
        Sx   += point.x;
        Sy   += point.y;
        Sx2  += point.x * point.x;
        Sy2  += point.y * point.y;
        Sxy  += point.x * point.y;
    }
    size_t n = vecPoints.size();
    if ( reverseFit )   {
        fSlope = static_cast<float>( ( n * Sxy - Sx * Sy ) / ( n * Sy2 - Sy * Sy ) );
        fIntercept = static_cast< float >( ( Sx * Sy2 - Sy * Sxy ) / ( n * Sy2 - Sy * Sy ) );
    }else {
        fSlope = static_cast<float>( ( n * Sxy - Sx * Sy ) / ( n * Sx2 - Sx * Sx ) );
        fIntercept = static_cast< float >( ( Sy * Sx2 - Sx * Sxy ) / ( n * Sx2 - Sx * Sx ) );
    }
}

//The equation is from http://hotmath.com/hotmath_help/topics/line-of-best-fit.html
/*static*/ void Fitting::fitLine(const ListOfPoint &listPoint, float &fSlope, float &fIntercept, bool reverseFit)
{
    if ( listPoint.size() < 2 )
        return;

    double Sx = 0., Sy = 0., Sx2 = 0., Sy2 = 0., Sxy = 0.;
    for ( const auto &point : listPoint )   {
        Sx   += point.x;
        Sy   += point.y;
        Sx2  += point.x * point.x;
        Sy2  += point.y * point.y;
        Sxy  += point.x * point.y;
    }
    size_t n = listPoint.size();
    if ( reverseFit )   {
        fSlope = static_cast<float>( ( n * Sxy - Sx * Sy ) / ( n * Sy2 - Sy * Sy ) );
        fIntercept = static_cast< float >( ( Sx * Sy2 - Sy * Sxy ) / ( n * Sy2 - Sy * Sy ) );
    }else {
        fSlope = static_cast<float>( ( n * Sxy - Sx * Sy ) / ( n * Sx2 - Sx * Sx ) );
        fIntercept = static_cast< float >( ( Sy * Sx2 - Sx * Sxy ) / ( n * Sx2 - Sx * Sx ) );
    }
}

//Using least square method to fit. Solve equation A * X = B.
/*static*/ void Fitting::fitParallelLine(const std::vector<cv::Point2f> &vecPoints1,
                                         const std::vector<cv::Point2f> &vecPoints2,
                                         float                          &fSlope,
                                         float                          &fIntercept1,
                                         float                          &fIntercept2,
                                         bool                            reverseFit)
{
    if ( vecPoints1.size() < 2 || vecPoints2.size() < 2 )
        return;

    auto totalDataCount = static_cast<int>(vecPoints1.size() + vecPoints2.size());
    auto dataCount1 = vecPoints1.size();
    cv::Mat B(totalDataCount, 1, CV_32FC1);
    cv::Mat A = cv::Mat::zeros(totalDataCount, 3, CV_32FC1);
    for ( int i = 0; i < totalDataCount; ++ i )  {
        if ( i < vecPoints1.size() )    {
            if ( reverseFit )
                A.at<float>(i, 0) = vecPoints1[i].y;
            else
                A.at<float>(i, 0) = vecPoints1[i].x;

            A.at<float>(i, 1) = 1.f;
            if ( reverseFit )
                B.at<float>(i, 0) = vecPoints1[i].x;
            else
                B.at<float>(i, 0) = vecPoints1[i].y;
        }
        else
        {
            if ( reverseFit )
                A.at<float>(i, 0) = vecPoints2[i - dataCount1].y;
            else
                A.at<float>(i, 0) = vecPoints2[i - dataCount1].x;
            A.at<float>(i, 2) = 1.f;
            if ( reverseFit )
                B.at<float>(i, 0) = vecPoints2[i - dataCount1].x;
            else
                B.at<float>(i, 0) = vecPoints2[i - dataCount1].y;
        }   
    }

    cv::Mat matResult;
    bool bResult = cv::solve(A, B, matResult, cv::DECOMP_SVD);

    fSlope = matResult.at<float>(0, 0);
    fIntercept1 = matResult.at<float>(1, 0);
    fIntercept2 = matResult.at<float>(2, 0);
}

/*static*/ void Fitting::fitParallelLine(const ListOfPoint &listPoints1,
                                         const ListOfPoint &listPoints2,
                                         float             &fSlope,
                                         float             &fIntercept1,
                                         float             &fIntercept2,
                                         bool               reverseFit)
{
    if ( listPoints1.size() < 2 || listPoints2.size() < 2 )
        return;

    auto totalDataCount = static_cast<int>(listPoints1.size() + listPoints2.size());
    auto dataCount1 = listPoints1.size();
    cv::Mat B(totalDataCount, 1, CV_32FC1);
    cv::Mat A = cv::Mat::zeros(totalDataCount, 3, CV_32FC1);
    ListOfPoint::const_iterator it1 = listPoints1.begin();
    ListOfPoint::const_iterator it2 = listPoints2.begin();
    for ( int i = 0; i < totalDataCount; ++ i )  {
        if ( i < listPoints1.size() )    {
            if ( reverseFit )
                A.at<float>(i, 0) = (*it1).y;
            else
                A.at<float>(i, 0) = (*it1).x;

            A.at<float>(i, 1) = 1.f;
            if ( reverseFit )
                B.at<float>(i, 0) = (*it1).x;
            else
                B.at<float>(i, 0) = (*it1).y;

            ++ it1;
        }
        else
        {
            if ( reverseFit )
                A.at<float>(i, 0) = (*it2).y;
            else
                A.at<float>(i, 0) = (*it2).x;
            A.at<float>(i, 2) = 1.f;
            if ( reverseFit )
                B.at<float>(i, 0) = (*it2).x;
            else
                B.at<float>(i, 0) = (*it2).y;

            ++ it2;
        }   
    }

    cv::Mat matResult;
    bool bResult = cv::solve(A, B, matResult, cv::DECOMP_SVD);

    fSlope = matResult.at<float>(0, 0);
    fIntercept1 = matResult.at<float>(1, 0);
    fIntercept2 = matResult.at<float>(2, 0);
}

/*static*/ void Fitting::fitRect(VectorOfVectorOfPoint &vecVecPoint,
                                 float                 &fSlope1,
                                 float                 &fSlope2,
                                 std::vector<float>    &vecIntercept,
                                 bool                   bLineOneReversedFit,
                                 bool                   bLineTwoReversedFit)
{
    assert ( vecVecPoint.size() == 4 );  //The input should has 4 lines
    vecIntercept.clear();
    auto totalRows = 0;
    for ( const auto &vectorPoint : vecVecPoint)
        totalRows += ToInt32(vectorPoint.size());
    cv::Mat A = cv::Mat::zeros ( totalRows, 5, CV_32FC1);
    cv::Mat B = cv::Mat::zeros ( totalRows, 1, CV_32FC1);
 
    int index = 0;
    for (int i = 0; i < vecVecPoint.size(); ++ i )
    {
        for (int j = 0; j < vecVecPoint [ i ].size(); ++ j )
        {
            A.at<float>(index, i + 1 ) = 1.f;
            if ( i < 2 )   {
                if (bLineOneReversedFit)   {
                    A.at<float>(index, 0) = vecVecPoint[i][j].y;
                    B.at<float>(index, 0) = vecVecPoint[i][j].x;
                }
                else
                {
                    A.at<float>(index, 0) = vecVecPoint[i][j].x;
                    B.at<float>(index, 0) = vecVecPoint[i][j].y;
                }
            }else {
                if ( bLineOneReversedFit )   {
                    A.at<float>(index, 0) = -vecVecPoint[i][j].x;
                    B.at<float>(index, 0) =  vecVecPoint[i][j].y;
                }else {
                    A.at<float>(index, 0) = -vecVecPoint[i][j].y;
                    B.at<float>(index, 0) =  vecVecPoint[i][j].x;
                }
            }
            ++index;
        }
    }

    cv::Mat matResult;
    bool bResult = cv::solve(A, B, matResult, cv::DECOMP_QR);

    fSlope2 = fSlope1 = matResult.at<float>(0, 0);
    for ( int i = 0; i < 4; ++ i )
        vecIntercept.push_back ( matResult.at < float >( i + 1, 0 ) );
}

/*static*/ void Fitting::fitRect(VectorOfListOfPoint &vecListPoint,
                                 float               &fSlope1,
                                 float               &fSlope2,
                                 std::vector<float>  &vecIntercept,
                                 bool                 bLineOneReversedFit,
                                 bool                 bLineTwoReversedFit)
{
    assert ( vecListPoint.size() == 4 );  //The input should has 4 lines
    vecIntercept.clear();
    auto totalRows = 0;
    for ( const auto &vectorPoint : vecListPoint)
        totalRows += ToInt32(vectorPoint.size());
    cv::Mat A = cv::Mat::zeros ( totalRows, 5, CV_32FC1);
    cv::Mat B = cv::Mat::zeros ( totalRows, 1, CV_32FC1);
 
    int index = 0;
    for (int i = 0; i < vecListPoint.size(); ++ i )
    {
        ListOfPoint listPoint = vecListPoint[i];
        for (const auto &point : listPoint )
        {
            A.at<float>(index, i + 1 ) = 1.f;
            if ( i < 2 )   {
                if (bLineOneReversedFit)   {
                    A.at<float>(index, 0) = point.y;
                    B.at<float>(index, 0) = point.x;
                }
                else
                {
                    A.at<float>(index, 0) = point.x;
                    B.at<float>(index, 0) = point.y;
                }
            }else {
                if ( bLineOneReversedFit )   {
                    A.at<float>(index, 0) = -point.x;
                    B.at<float>(index, 0) =  point.y;
                }else {
                    A.at<float>(index, 0) = -point.y;
                    B.at<float>(index, 0) =  point.x;
                }
            }
            ++ index;
        }
    }

    cv::Mat matResult;
    bool bResult = cv::solve(A, B, matResult, cv::DECOMP_QR);

    fSlope1 = matResult.at<float>(0, 0);
    fSlope2 = - fSlope1;
    for ( int i = 0; i < 4; ++ i )
        vecIntercept.push_back ( matResult.at < float >( i + 1, 0 ) );
}

}
}