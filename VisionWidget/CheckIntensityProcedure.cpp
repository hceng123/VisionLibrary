#include "CheckIntensityProcedure.h"
#include "messageboxdialog.h"
#include "constants.h"
#include <QMessageBox>
#include <QApplication>

using namespace AOI::Vision;

CheckIntensityProcedure::CheckIntensityProcedure(VisionView *pVisionView) : Procedure(pVisionView)
{
}

CheckIntensityProcedure::~CheckIntensityProcedure()
{
}

int CheckIntensityProcedure::run(const cv::Mat &mat)
{
    QRect rect = qApp->activeWindow()->geometry();
	std::unique_ptr<MessageBoxDialog> pMessageBox = std::make_unique<MessageBoxDialog>();
    pMessageBox->setGeometry(rect.x(), rect.y() + 50, pMessageBox->size().width(), pMessageBox->size().height());
    pMessageBox->setWindowTitle("Check Intensity Along Line");
	pMessageBox->SetMessageText1("Please select the line to check intensity");
    pMessageBox->SetMessageText2("Press and drag the left mouse buttont to input");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_LINE);
    _pVisionView->setCurrentSrchWindowIndex(0);
	int iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
        return ToInt(STATUS::NOK);
    }	
   
    _lineOfIntensityCheck = _pVisionView->getIntensityCheckLine();
    if ( VisionView::distanceOf2Point( _lineOfIntensityCheck.pt1, _lineOfIntensityCheck.pt2 ) <= 1 )   {
        pMessageBox->SetMessageText1("The input is invalid");
        pMessageBox->SetMessageText2("");
        pMessageBox->exec();
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
        return ToInt(STATUS::NOK);
    }

    checkIntensity(mat);

    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
    return 0;
}

// Interpolates pixel intensity with subpixel accuracy.
// Abount bilinear interpolation in Wikipedia:
// http://en.wikipedia.org/wiki/Bilinear_interpolation
template <class T>
float interpolate(const Mat &mat, float x, float y)
{
    // Get the nearest integer pixel coords (xi;yi).
    int xi = cvFloor(x);
    int yi = cvFloor(y);
 
    float k1 = x - xi; // Coefficients for interpolation formula.
    float k2 = y - yi;
 
    bool b1 = xi < mat.cols - 1;  // Check that pixels to the right 
    bool b2 = yi < mat.rows - 1; // and to down direction exist.
 
    float UL = mat.at<T>(Point(xi, yi));
    float UR = b1 ? mat.at<T>( Point (xi + 1, yi ) ) : 0.f;
    float LL = b2 ? mat.at<T>( Point ( xi, yi + 1) ) : 0.f;
    float LR = b1 & b2 ? mat.at <T>( Point ( xi + 1, yi + 1 ) ) : 0.f;
 
    // Interpolate pixel intensity.
    float interpolated_value = (1.0f - k1) * (1.0f - k2) * UL + k1 * (1.0f - k2) * UR +
        (1.0f - k1) * k2 * LL + k1 * k2 * LR;
 
    return interpolated_value;
}
 
//Get the intensity along an input line
template <class T>
int GetIntensityOnLine ( const Mat &mat, const Point &start, const Point &end, vector<float> &vecOutput )
{
    if ( start.x >= mat.cols || start.y >= mat.rows )
        return -1;
    if ( end.x >= mat.cols || end.y >= mat.rows )
        return -1;
 
    float fLineLen = (float)sqrt ( ( end.x - start.x ) * ( end.x - start.x ) + ( end.y - start.y ) * ( end.y - start.y ) );
    if ( fLineLen < 1 )
        return -1;
 
    float fCos = ( end.x - start.x ) / fLineLen;
    float fSin = ( end.y - start.y ) / fLineLen;
 
    float fCurrentLen = 0.f;
    while ( fCurrentLen < fLineLen )    {
        float fX = start.x + fCos * fCurrentLen;
        float fY = start.y + fSin * fCurrentLen;
        float fIntensity = interpolate<T> ( mat, fX, fY );
        vecOutput.push_back ( fIntensity );
         
        ++ fCurrentLen;
    }
 
    return 0;
}

int CheckIntensityProcedure::checkIntensity(const cv::Mat &mat)
{
    cv::Mat matGray;
    cv::cvtColor(mat, matGray, CV_BGR2GRAY);
    std::vector<float> vecOutput;
	GetIntensityOnLine<unsigned char> ( matGray, _lineOfIntensityCheck.pt1, _lineOfIntensityCheck.pt2, vecOutput );

    cv::Mat matResult = cv::Mat::ones(PR_MAX_GRAY_LEVEL, ToInt(vecOutput.size()), CV_8UC1) * PR_MAX_GRAY_LEVEL;
    for (int col = 1; col < ToInt(vecOutput.size()); ++ col)
    {
        cv::Point pt1(col - 1, PR_MAX_GRAY_LEVEL - vecOutput[col - 1]);
        cv::Point pt2(col,     PR_MAX_GRAY_LEVEL - vecOutput[col]);
        cv::line(matResult, pt1, pt2, cv::Scalar(0), 1);
    }
    cv::imshow("Gray Scale", matResult);
    cv::waitKey(0);

    return 0;
}
