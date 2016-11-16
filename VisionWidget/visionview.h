#ifndef _VISIONVIEW_H_
#define _VISIONVIEW_H_

#include <QLabel>
#include <QMouseEvent>
#include <QTimer>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include "dialogeditmask.h"

using namespace cv;
using namespace std;

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

    enum VISION_VIEW_STATE
    {
        IDLE,
        LEARNING,
        ADD_MASK
    };

    //explicit VisionView(QLabel *parent = 0);
    explicit VisionView(QWidget *parent = 0, Qt::WindowFlags f=0);
    ~VisionView();
    void setMachineState(int nMachineState);
    void setMaskEditState(MASK_EDIT_STATE enMaskEditState);
    void setMaskShape(MASK_SHAPE enMaskShape);
    void zoomIn();
    void zoomOut();
    void restoreZoom();
    void startTimer();
    static float distanceOf2Point(const cv::Point &pt1, const cv::Point &pt2);
protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void _drawLearnWindow(Mat &mat);

private:
    Rect                _rectLrnWindow;
    Rect                _rectSrchWindow;
    bool                _bLeftButtonDown;
    cv::Point           _ptLeftClickStartPos;
    cv::Point           _ptLeftClickEndPos;
    int                 _nState;
    cv::Mat             _mat;
    cv::Mat             _matMask;
    vector<Rect>        _vecRectMask;
    float               _fZoomFactor;
    QTimer*             _pTimer;
    short               _nCameraID;
    MASK_EDIT_STATE     _enMaskEditState;
    MASK_SHAPE          _enMaskShape;
    vector<cv::Point>   _vecPolylinePoint;

    const cv::Scalar    _colorLrnWindow = Scalar ( 0, 255, 0 );
    const cv::Scalar    _colorMask = Scalar ( 255, 0, 0 );
    const float         _constMaxZoomFactor = 4.f;
    const float         _constMinZoomFactor = 0.25;
    DialogEditMask      *_pDialogEditMask;
signals:

protected slots:
    void showContextMenu(const QPoint& pos); // this is a slot
    void addMask();
    int updateMat();
};

#endif // VISIONVIEW_H
