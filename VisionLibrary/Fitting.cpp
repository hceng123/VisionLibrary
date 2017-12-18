#include "Fitting.h"
#include "CalcUtils.h"

namespace AOI
{
namespace Vision
{

//The equation is from http://hotmath.com/hotmath_help/topics/line-of-best-fit.html
/*static*/ void Fitting::fitLine(const VectorOfPoint &vecPoints, float &fSlope, float &fIntercept, bool reverseFit)
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

    cv::Mat B(ToInt32(listPoint.size()), 1, CV_32FC1);
    cv::Mat A = cv::Mat::ones(ToInt32(listPoint.size()), 2, CV_32FC1);
    int i = 0;
    for (const auto &point : listPoint)  {
        if (reverseFit) {
            A.at<float>(i, 0) = ToFloat ( point.y );
            B.at<float>(i, 0) = ToFloat ( point.x );
        }
        else
        {
            A.at<float>(i, 0) = ToFloat ( point.x );
            B.at<float>(i, 0) = ToFloat ( point.y );
        }
        ++ i;
    }

    cv::Mat matResultImg;
    bool bResult = cv::solve(A, B, matResultImg, cv::DECOMP_SVD);

    fSlope = matResultImg.at<float>(0, 0);
    fIntercept = matResultImg.at<float>(1, 0);
}

//Using least square method to fit. Solve equation A * X = B.
/*static*/ void Fitting::fitParallelLine(const VectorOfPoint &vecPoints1,
                                         const VectorOfPoint &vecPoints2,
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
        if ( i < ToInt32 ( vecPoints1.size() ) )    {
            if ( reverseFit )
                A.at<float>(i, 0) = ToFloat ( vecPoints1[i].y );
            else
                A.at<float>(i, 0) = ToFloat ( vecPoints1[i].x );

            A.at<float>(i, 1) = 1.f;
            if ( reverseFit )
                B.at<float>(i, 0) = ToFloat ( vecPoints1[i].x );
            else
                B.at<float>(i, 0) = ToFloat ( vecPoints1[i].y );
        }
        else
        {
            if ( reverseFit )
                A.at<float>(i, 0) = ToFloat ( vecPoints2[i - dataCount1].y );
            else
                A.at<float>(i, 0) = ToFloat ( vecPoints2[i - dataCount1].x );
            A.at<float>(i, 2) = 1.f;
            if ( reverseFit )
                B.at<float>(i, 0) = ToFloat ( vecPoints2[i - dataCount1].x );
            else
                B.at<float>(i, 0) = ToFloat ( vecPoints2[i - dataCount1].y );
        }   
    }

    cv::Mat matResultImg;
    bool bResult = cv::solve(A, B, matResultImg, cv::DECOMP_SVD);

    fSlope = matResultImg.at<float>(0, 0);
    fIntercept1 = matResultImg.at<float>(1, 0);
    fIntercept2 = matResultImg.at<float>(2, 0);
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
        if ( i < ToInt32 ( listPoints1.size() ) )    {
            if ( reverseFit )
                A.at<float>(i, 0) = ToFloat ( (*it1).y );
            else
                A.at<float>(i, 0) = ToFloat ( (*it1).x );

            A.at<float>(i, 1) = 1.f;
            if ( reverseFit )
                B.at<float>(i, 0) = ToFloat ( (*it1).x );
            else
                B.at<float>(i, 0) = ToFloat ( (*it1).y );

            ++ it1;
        }
        else
        {
            if ( reverseFit )
                A.at<float>(i, 0) = ToFloat ( (*it2).y );
            else
                A.at<float>(i, 0) = ToFloat ( (*it2).x );
            A.at<float>(i, 2) = 1.f;
            if ( reverseFit )
                B.at<float>(i, 0) = ToFloat ( (*it2).x );
            else
                B.at<float>(i, 0) = ToFloat ( (*it2).y );

            ++ it2;
        }   
    }

    cv::Mat matResultImg;
    bool bResult = cv::solve(A, B, matResultImg, cv::DECOMP_SVD);

    fSlope = matResultImg.at<float>(0, 0);
    fIntercept1 = matResultImg.at<float>(1, 0);
    fIntercept2 = matResultImg.at<float>(2, 0);
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
    for (int i = 0; i < ToInt32 ( vecVecPoint.size() ); ++ i )
    {
        for (int j = 0; j < ToInt32 ( vecVecPoint [ i ].size() ); ++ j )
        {
            A.at<float>(index, i + 1 ) = 1.f;
            if ( i < 2 )   {
                if (bLineOneReversedFit)   {
                    A.at<float>(index, 0) = ToFloat ( vecVecPoint[i][j].y );
                    B.at<float>(index, 0) = ToFloat ( vecVecPoint[i][j].x );
                }
                else
                {
                    A.at<float>(index, 0) = ToFloat ( vecVecPoint[i][j].x );
                    B.at<float>(index, 0) = ToFloat ( vecVecPoint[i][j].y );
                }
            }else {
                if ( bLineOneReversedFit )   {
                    A.at<float>(index, 0) = ToFloat ( -vecVecPoint[i][j].x );
                    B.at<float>(index, 0) = ToFloat (  vecVecPoint[i][j].y );
                }else {
                    A.at<float>(index, 0) = ToFloat ( -vecVecPoint[i][j].y );
                    B.at<float>(index, 0) = ToFloat (  vecVecPoint[i][j].x );
                }
            }
            ++index;
        }
    }

    cv::Mat matResultImg;
    bool bResult = cv::solve(A, B, matResultImg, cv::DECOMP_QR);

    fSlope2 = fSlope1 = matResultImg.at<float>(0, 0);
    for ( int i = 0; i < 4; ++ i )
        vecIntercept.push_back ( matResultImg.at < float >( i + 1, 0 ) );
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
    for (int i = 0; i < ToInt32 ( vecListPoint.size() ); ++ i )
    {
        ListOfPoint listPoint = vecListPoint[i];
        for (const auto &point : listPoint )
        {
            A.at<float>(index, i + 1 ) = 1.f;
            if ( i < 2 )   {
                if (bLineOneReversedFit)   {
                    A.at<float>(index, 0) = ToFloat ( point.y );
                    B.at<float>(index, 0) = ToFloat ( point.x );
                }
                else
                {
                    A.at<float>(index, 0) = ToFloat ( point.x );
                    B.at<float>(index, 0) = ToFloat ( point.y );
                }
            }else {
                if ( bLineOneReversedFit )   {
                    A.at<float>(index, 0) = ToFloat ( -point.x );
                    B.at<float>(index, 0) = ToFloat (  point.y );
                }else {
                    A.at<float>(index, 0) = ToFloat ( -point.y );
                    B.at<float>(index, 0) = ToFloat (  point.x );
                }
            }
            ++ index;
        }
    }

    cv::Mat matResultImg;
    bool bResult = cv::solve(A, B, matResultImg, cv::DECOMP_QR);

    fSlope1 = matResultImg.at<float>(0, 0);
    fSlope2 = - fSlope1;
    for ( int i = 0; i < 4; ++ i )
        vecIntercept.push_back ( matResultImg.at < float >( i + 1, 0 ) );
}

}
}