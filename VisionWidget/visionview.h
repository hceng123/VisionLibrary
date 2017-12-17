#ifndef _VISIONVIEW_H_
#define _VISIONVIEW_H_

#include <QLabel>
#include <QMouseEvent>
#include <QTimer>
#include <vector>
#include <string>
#include <memory>

#include "opencv2/core/core.hpp"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include "dialogeditmask.h"
#include "VisionAPI.h"

using namespace cv;
using namespace std;
using namespace AOI::Vision;

bool static IsEuqal(float a, float b)
{
    return ( ( a - b ) < 0.00001) && ( ( b - a ) < 0.00001 );
}

class DialogEditMask;

class VisionView : public QLabel
{
    Q_OBJECT
public:

    enum class MASK_EDIT_STATE
    {
        NONE,
        ADD,
        REMOVE,
    };

    enum MASK_SHAPE
    {
        MASK_SHAPE_RECT,
        MASK_SHAPE_CIRCLE,
        MASK_SHAPE_POLYLINE,
    };

    enum class VISION_VIEW_STATE
    {
        IDLE,
        LEARNING,
        ADD_MASK,
        TEST_VISION_LIBRARY,
        MOVE,
    };

    enum class TEST_VISION_STATE
    {
        IDLE,
        SET_CIRCLE_CTR,
        SET_CIRCLE_INNER_RADIUS,
        SET_CIRCLE_OUTER_RADIUS,
        SET_MULTIPLE_WINDOW,
        SET_LINE,
        PICK_COLOR,
    };

    enum DISPLAY_SOURCE
    {
        ORIGINAL,
        INTERMEDIATE,       //Test Image Source
        RESULT,
        END,
        SIZE = END,
    };

    const float ZOOM_IN_STEP = 2.0;
    const float ZOOM_OUT_STEP = 0.5;

    //explicit VisionView(QLabel *parent = 0);
    explicit VisionView(QWidget *parent = 0, Qt::WindowFlags f=0);
    ~VisionView();
    void setState(VISION_VIEW_STATE enState);
    void setMaskEditState(MASK_EDIT_STATE enMaskEditState);
    void setMaskShape(MASK_SHAPE enMaskShape);
    void setTestVisionState(TEST_VISION_STATE enState);
    void swapImage();
    void zoomIn();
    void zoomOut();
    void startToMove();
    void restoreZoom();
    void startTimer();
    void setMat( DISPLAY_SOURCE enSource, const cv::Mat &mat);
    cv::Mat getMat( DISPLAY_SOURCE enSource = VisionView::DISPLAY_SOURCE::ORIGINAL ) const;
    cv::Mat getCurrentMat() const;
    cv::Mat getMask() const;
    void applyIntermediateResult();
    void clearMat( DISPLAY_SOURCE enSource );
    bool isDisplayResultImage() const;
    bool isDisplayGrayScale() const;
    void setCurrentSrchWindowIndex(int nIndex);
    void getFitCircleRange(cv::Point &ptCtr, float &fInnerRadius, float &fOuterRadius) const;
    void setImageDisplayMode(bool bDisplayGrayScale, bool bDisplayBinary);
    void setRGBRatio(float fRatioR, float fRatioG, float fRatioB);
    void getRGBRatio(float &fRatioR, float &fRatioG, float &fRatioB);
    void setBinaryThreshold(int nThreshold, bool bReverseThres);
    VectorOfRect getVecSrchWindow( ) const;
    cv::Rect getSelectedWindow() const;
    cv::Point getClickedPoint() const;
    PR_Line getIntensityCheckLine() const;
    static float distanceOf2Point(const cv::Point &pt1, const cv::Point &pt2);    
protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void _drawDisplay(bool bAddingMask = false);
    void _drawLearnWindow(cv::Mat &mat);
    void _drawSelectedWindow(cv::Mat &mat);
    void _drawTestVisionLibrary(cv::Mat &mat);
    void _zoomPoint(cv::Point &point, float fZoomFactor);
    void _zoomRect(cv::Rect &rect, float fZoomFactor);
    static void _zoomMat(const cv::Mat &mat, cv::Mat &matOutput, float fZoomFactor, bool bKeepSize = true);
    void _calcMoveRange();
    void _moveImage();
    void _applyMask();
    void _cutImageForDisplay(const cv::Mat &matInputImg, cv::Mat &matOutput);
private:
    cv::Rect                        _rectLrnWindow;
    cv::Rect                        _rectSrchWindow;
    bool                            _bLeftButtonDown;
    cv::Point                       _ptLeftClickStartPos;
    cv::Point                       _ptLeftClickEndPos;
    VISION_VIEW_STATE               _enState;
    TEST_VISION_STATE               _enTestVisionState;
    cv::Mat                         _matArray[DISPLAY_SOURCE::SIZE];
    DISPLAY_SOURCE                  _enDisplaySource;
	cv::Mat             			_matDisplay;
    cv::Mat                         _matMaskForDisplay;
    cv::Mat                         _matMask;
    VectorOfRect                    _vecRectMask;
    float                           _fZoomFactor;
    std::unique_ptr<QTimer>         _pTimer;
    short                           _nCameraID;
    MASK_EDIT_STATE                 _enMaskEditState;
    MASK_SHAPE                      _enMaskShape;
    vector<cv::Point>               _vecPolylinePoint;

    bool                            _bDisplayGrayScale;
    bool                            _bDisplayBinary;
    PR_RGB_RATIO                    _stRGBRatio;
    int                             _nThreshold;
    bool                            _bReverseThres;
    cv::Rect                        _rectSelectedWindow;
    cv::Point                       _ptCircleCtr;
    float                           _fInnerRangeRadius;
    float                           _fOuterRangeRadius;
    int                             _nCurrentSrchWindowIndex;
    std::vector<cv::Rect>           _vecRectSrchWindow;
    PR_Line                         _lineOfIntensityCheck;

    const cv::Scalar                _colorLrnWindow = Scalar ( 0, 255, 0 );
    const cv::Scalar                _colorMask = Scalar ( 255, 0, 0 );
    const float                     _constMaxZoomFactor = 4.f;
    const float                     _constMinZoomFactor = 0.25;
    std::unique_ptr<DialogEditMask> _pDialogEditMask;
    cv::Size                        _szDisplayCenterOffset;
    cv::Size                        _szConfirmedCenterOffset;
    cv::Size                        _szMoveRange;

signals:
    void PickColor();

protected slots:
    void showContextMenu(const QPoint& pos); // this is a slot
    void selectWindow();
    void addMask();
    void clearSelectedWindow();
    int updateMat();
};

#endif // VISIONVIEW_H
