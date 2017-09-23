#include "BaseType.h"
#include "SubFunctions.h"
#include <numeric>
#include <iostream>

namespace AOI
{
namespace Vision
{
using VectorOfFloat = std::vector<float>;
using VectorOfVectorOfFloat = std::vector<VectorOfFloat>;

template<typename Tp>
std::vector<Tp> getMultipleRow(const std::vector<Tp> &vecInput, std::vector<int> vecRow) {
    std::vector<Tp> vecResult;
    for ( auto row : vecRow )
        vecResult.push_back ( vecInput[row] );
    return vecResult;
}

template<typename Tp>
bool isVectorAbsDiffLessThan(const std::vector<Tp> &vecInput, const std::vector<int> &vecIndex, Tp fValue, Tp fTol) {
    for ( auto index : vecIndex ) {
        if ( fabs ( vecInput[index] - fValue ) > fTol )
            return false;
    }
    return true;
}

template<typename Tp>
std::vector<std::vector<Tp>> getMultipleRow(const std::vector<std::vector<Tp>> &vecVecInput, std::vector<int> vecRow) {
    std::vector<std::vector<Tp>> vecVecResult;
    for ( auto row : vecRow )
        vecVecResult.push_back ( vecVecInput[row] );
    return vecVecResult;
}

template<typename Tp>
std::vector<std::vector<Tp>> substract(const std::vector<std::vector<Tp>> &vecVecInput, const std::vector<Tp> &vecRow ) {
    assert ( vecVecInput[0].size() == vecRow.size() );
    std::vector<std::vector<Tp>> vecVecExpand;

    for ( size_t row = 0; row < vecVecInput.size(); ++ row )
        vecVecExpand.push_back ( vecRow );

    std::vector<std::vector<Tp>> vecVecResult(vecVecInput);
    for ( size_t row = 0; row < vecVecExpand.size(); ++ row )
    for ( size_t col = 0; col < vecVecExpand[row].size(); ++ col ) {
        vecVecResult[row][col] -= vecVecExpand[row][col];
    }
    return vecVecResult;
}

template<typename Tp>
std::vector<Tp> substract(const std::vector<Tp> &vecInputA, const std::vector<Tp> &vecInputB ) {
    assert ( vecInputA.size() == vecInputB.size() );
    auto vecResult ( vecInputA );
    for ( size_t i = 0; i < vecResult.size(); ++ i )
        vecResult[i] -= vecInputB[i];
    return vecResult;
}

template<typename Tp>
std::vector<Tp> add(const std::vector<Tp> &vecInputA, const std::vector<Tp> &vecInputB ) {
    assert ( vecInputA.size() == vecInputB.size() );
    auto vecResult ( vecInputA );
    for ( size_t i = 0; i < vecResult.size(); ++ i )
        vecResult[i] += vecInputB[i];
    return vecResult;
}

template<typename Tp>
std::vector<Tp> multiply( const std::vector<Tp> &vecInput, Tp factor) {
    auto vecResult ( vecInput );
    for ( auto &value : vecResult )
        value *= factor;
    return vecResult;
}

template<typename Tp>
Tp maxInVV(const std::vector<std::vector<Tp>> &vecVecInput) {
    Tp result = std::numeric_limits<Tp>::min();
    for ( size_t row = 0; row < vecVecInput.size(); ++ row )
    for ( size_t col = 0; col < vecVecInput[row].size(); ++ col ) {
        if ( result < vecVecInput[row][col] )
            result = vecVecInput[row][col];
    }
    return result;
}

template<typename Tp>
void setOneRow(std::vector<std::vector<Tp>> &vecVecInput, int row, const std::vector<Tp> &vecRow) {
    assert(vecVecInput[0].size() == vecRow.size());
    vecVecInput[row] = vecRow;
}

template<typename Tp>
void resequence(std::vector<std::vector<Tp>> &vecVecInput, const std::vector<size_t> &vecIndex) {
    auto vecVecClone(vecVecInput);
    for ( size_t i = 0; i < vecIndex.size(); ++ i )
        vecVecInput[i] = vecVecClone[vecIndex[i]];
}

template<typename Tp>
std::vector<Tp> averagOfRows(std::vector<std::vector<Tp>> &vecVecInput, const std::vector<size_t> &vecRow) {
    std::vector<Tp> vecAvg ( vecVecInput[0].size(), 0 );
    for ( auto row : vecRow ) {
        for ( size_t i = 0; i < vecAvg.size(); ++ i )
            vecAvg[i] += vecVecInput[row][i] / static_cast<Tp> ( vecRow.size() );
    }
    return vecAvg;
}

template<typename T>
std::vector<size_t> sort_index_value ( std::vector<T> &v ) {
    // initialize original index locations
    std::vector<size_t> idx ( v.size () );
    std::iota ( idx.begin (), idx.end (), 0 );

    // sort indexes based on comparing values in v
    sort ( idx.begin (), idx.end (),
        [&v]( size_t i1, size_t i2 ) {return v[i1] < v[i2]; } );

    std::vector<T> vClone(v);
    for ( size_t i = 0; i < idx.size(); ++ i )
        v[i] = vClone[idx[i]];

    return idx;
}

float matlabObjFunction(const VectorOfFloat &vecThreshold, int num_bins, const cv::Mat &omega, const cv::Mat &mu, float mu_t ) {
    std::vector<int> boundaries;
    boundaries.reserve(vecThreshold.size());
    for ( auto thresh : vecThreshold )
        boundaries.push_back ( ToInt32( std::round(thresh) ) );
    boundaries.push_back ( num_bins - 1 );
    float fDiff = mu.at<float>(boundaries[0]) / omega.at<float>(boundaries[0]) - mu_t;
    float sigma_b_squared_val = omega.at<float>(boundaries[0]) * fDiff * fDiff;
    for ( int kk = 1; kk < boundaries.size(); ++ kk ) {
        float omegaKK = omega.at<float> ( boundaries[kk] ) - omega.at<float> ( boundaries[kk - 1] );
        float muKK = (mu.at<float>( boundaries[kk] ) - mu.at<float>(boundaries[kk-1] ) ) / omegaKK;
        sigma_b_squared_val += omegaKK * pow (muKK - mu_t, 2); // Eqn. 14 in Otsu's paper
    }
    return -sigma_b_squared_val;
}

float autoThresholdObjFunction(const std::vector<float> &vecX, const VectorOfVectorOfFloat &vecVecOmega, const VectorOfVectorOfFloat &vecVecMu) {
    std::vector<float> vecNewX;
    vecNewX.push_back ( 0.f );
    for (const auto thresh : vecX)
        vecNewX.push_back ( thresh );
    vecNewX.push_back ( ToFloat ( vecVecOmega.size() - 1 ) );
    float fSigma = 0.f;
    const float MIN_P = 0.001f;
    for ( size_t i = 0; i < vecNewX.size() - 1; ++ i ) {
        int T1 = ToInt32 ( round ( vecNewX[i] ) );
        int T2 = ToInt32 ( round ( vecNewX[i + 1] ) );
        if ( vecVecOmega[T1][T2] > MIN_P ) {
            fSigma += vecVecMu[T1][T2] * vecVecMu[T1][T2] / vecVecOmega[T1][T2];
        }
    }
    return - fSigma;
}

int fminsearch (objFuntion          funfcn,
                std::vector<float> &vecX,
                int                 num_bins,
                const cv::Mat       &omega,
                const cv::Mat       &mu,
                float               mu_t )
{
    const int prnt = 0;
    auto n = ToInt32 ( vecX.size() );
    const int maxfun = 200 * ToInt32 ( vecX.size() );
    const int maxiter = 200 * ToInt32 ( vecX.size() );
    int func_evals = 0;
    int itercount = 0;
    float tolf = 1e-4f;
    float rho = 1, chi = 2, psi = 0.5, sigma = 0.5;
    std::vector<int> onesn(n, 1);
    std::vector<int> two2np1(n);
    std::iota (two2np1.begin(), two2np1.end(), 1 );
    std::vector<size_t> one2n(n);
    std::iota (one2n.begin(), one2n.end(), 0 );
    std::vector<float> fv(n + 1, 0);
    std::vector<std::vector<float>> v(n+1, std::vector<float>(n, 0.f));
    std::vector<float> xmin = vecX;
    setOneRow<float> ( v, 0, xmin );
    fv[0] = funfcn ( vecX, num_bins, omega, mu, mu_t );

    // Continue setting up the initial simplex.
    // Following improvement suggested by L.Pfeffer at Stanford
    float usual_delta = 0.05f;             // 5 percent deltas for non-zero terms
    float zero_term_delta = 0.00025f;      // Even smaller delta for zero elements of x
    for ( int i = 0; i < n; ++ i ) {
        auto y = xmin;
        if ( y[i] != 0 )
            y[i] = ( 1 + usual_delta ) * y[i];
        else
            y[i] = zero_term_delta;
        setOneRow ( v, i + 1, y );
        float f = funfcn ( y, num_bins, omega, mu, mu_t );
        fv[i+1] = f;
    }
    String how = "initial simplex";
    auto idx = sort_index_value ( fv );
    resequence(v, idx);
    ++ itercount;
    func_evals = n + 1; //Because just called funfcn n + 1 times.
    if ( prnt == 3 )
        std::cout << "itercount = " << itercount << ", func_evals = " << func_evals << ", fv = " << fv[0] << ", " << how << std::endl;
    
    while ( func_evals < maxfun && itercount < maxiter ) {
        auto vecVecTwo2Np1V = getMultipleRow (v, two2np1);
        if ( isVectorAbsDiffLessThan ( fv, two2np1, fv[0], tolf) &&
            maxInVV ( substract ( vecVecTwo2Np1V, v[0] ) ) < 1 ) {
            if ( prnt == 3 )
                std::cout << "Finish iteration with converged result." << std::endl;
            break;
        }

        // xbar = average of the n (NOT n+1) best points
        auto xbar = averagOfRows ( v, one2n );
        auto xr = substract ( multiply ( xbar, 1 + rho ), multiply ( v.back (), rho ) );  //Calculate reflection, Equal to matlab xr = (1 + rho)*xbar - rho*v(:,end);
        auto x = xr;
        auto fxr = funfcn ( x, num_bins, omega, mu, mu_t ); ++ func_evals;

        if ( fxr < fv[0] ) {
            // Calculate the expansion point
            auto xe = substract ( multiply ( xbar, 1 + rho*chi), multiply ( v.back(), rho*chi) );
            x = xe;
            auto fxe = funfcn ( x, num_bins, omega, mu, mu_t ); ++ func_evals;

            if ( fxe < fxr ) {
                v.back() = xe;
                fv.back() = fxe;
                how = "expand";
            }else {
                v.back() = xr;
                fv.back() = fxr;
                how = "reflect";
            }
        }else { //% fv(:,1) <= fxr
            if ( fxr < fv[n-1] ) {  // if fxr < fv(:,n)
                v.back() = xr;
                fv.back() = fxr;
                how = "reflect";
            }else { // % fxr >= fv(:,n)
                // Perform contraction
                if ( fxr < fv.back() ) {
                    auto xc = substract ( multiply ( xbar, 1 + psi*rho), multiply ( v.back(), psi*rho)); //xc = (1 + psi*rho)*xbar - psi*rho*v(:,end);
                    x = xc;
                    auto fxc = funfcn ( x, num_bins, omega, mu, mu_t ); ++ func_evals;
                    if ( fxc < fxr ) {
                        v.back() = xc;
                        fv.back() = fxc;
                        how = "contract outside";
                    }else
                        how = "shrink";
                }else {
                    // Perform an inside contraction
                    auto xcc = add ( multiply ( xbar, 1 - psi ), multiply ( v.back(), psi ) ); //xcc = (1-psi)*xbar + psi*v(:,end);
                    x = xcc; 
                    auto fxcc = funfcn ( x, num_bins, omega, mu, mu_t ); ++ func_evals;
                    if ( fxcc < fv.back() ) {
                        v.back() = xcc;
                        fv.back() = fxcc;
                        how = "contract inside";
                    }else {
                        // perform a shrink
                        how = "shrink";
                    }
                }

                if ( how == "shrink" ) {
                    for ( auto index : two2np1) {
                        v[index] = add ( v[0], multiply ( substract ( v[index], v[0]), sigma ) ); // Matlab, v(:,j)=v(:,1)+sigma*(v(:,j) - v(:,1));
                        fv[index] = funfcn ( v[index], num_bins, omega, mu, mu_t ); ++ func_evals;
                    }
                }
            }
        }

        auto idx = sort_index_value ( fv );
        resequence(v, idx);
        ++ itercount;

        if ( prnt == 3 )
            std::cout << "itercount = " << itercount << ", func_evals = " <<  func_evals << ", fv = " << fv[0] << ", " << how << std::endl;        
    }

    vecX = v[0];
    return 0;
}

}
}