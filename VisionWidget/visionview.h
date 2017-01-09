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

    enum MASK_EDIT_STATE
    {
        MASK_EDIT_NONE,
        MASK_EDIT_ADD,
        MASK_EDIT_REMOVE,
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
    };

    enum class TEST_VISION_STATE
    {
        UNDEFINED,
        SET_CIRCLE_CTR,
        SET_CIRCLE_INNER_RADIUS,
        SET_CIRCLE_OUTTER_RADIUS,
        SET_RECT_SRCH_WINDOW,
        SET_LINE,
    };

    //explicit VisionView(QLabel *parent = 0);
    explicit VisionView(QWidget *parent = 0, Qt::WindowFlags f=0);
    ~VisionView();
    void setMachineState(VISION_VIEW_STATE enMachineState);
    void setMaskEditState(MASK_EDIT_STATE enMaskEditState);
    void setMaskShape(MASK_SHAPE enMaskShape);
    void setTestVisionState(TEST_VISION_STATE enState);
    void swapImage();
    void zoomIn();
    void zoomOut();
    void restoreZoom();
    void startTimer();
    void setMat(const cv::Mat &mat);
    cv::Mat getMat() const;
    void setResultMat(const cv::Mat &mat);
    bool isDisplayResultImage() const;
    void setCurrentSrchWindowIndex(int nIndex);
    void getFitCircleRange(cv::Point &ptCtr, float &fInnterRadius, float &fOutterRadius) const;
    VectorOfRect getVecSrchWindow( ) const;
    PR_Line getIntensityCheckLine() const;
    static float distanceOf2Point(const cv::Point &pt1, const cv::Point &pt2);    
protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void _drawDisplay();
    void _drawLearnWindow(cv::Mat &mat);
    void _drawTestVisionLibrary(cv::Mat &mat);
    void zoomPoint(cv::Point &point, float fZoomFactor);
    void zoomRect(cv::Rect &rect, float fZoomFactor);
private:
    cv::Rect                        _rectLrnWindow;
    cv::Rect                        _rectSrchWindow;
    bool                            _bLeftButtonDown;
    cv::Point                       _ptLeftClickStartPos;
    cv::Point                       _ptLeftClickEndPos;
    VISION_VIEW_STATE               _enState;
    TEST_VISION_STATE               _enTestVisionState;
    cv::Mat                         _mat;
    cv::Mat                         _matResult;
    bool                            _bDisplayResultImage;
	cv::Mat             			_matDisplay;
    cv::Mat                         _matMask;
    VectorOfRect                    _vecRectMask;
    float                           _fZoomFactor;
    std::unique_ptr<QTimer>         _pTimer;
    short                           _nCameraID;
    MASK_EDIT_STATE                 _enMaskEditState;
    MASK_SHAPE                      _enMaskShape;
    vector<cv::Point>               _vecPolylinePoint;
    cv::Point                       _ptCircleCtr;
    float                           _fInnerRangeRadius;
    float                           _fOutterRangeRadius;
    int                             _nCurrentSrchWindowIndex;
    std::vector<cv::Rect>           _vecRectSrchWindow;
    PR_Line                         _lineOfIntensityCheck;

    const cv::Scalar                _colorLrnWindow = Scalar ( 0, 255, 0 );
    const cv::Scalar                _colorMask = Scalar ( 255, 0, 0 );
    const float                     _constMaxZoomFactor = 4.f;
    const float                     _constMinZoomFactor = 0.25;
    std::unique_ptr<DialogEditMask> _pDialogEditMask;
signals:

protected slots:
    void showContextMenu(const QPoint& pos); // this is a slot
    void addMask();
    int updateMat();
};

#endif // VISIONVIEW_H
