#include "visionview.h"
#include "VisionHeader.h"
#include "constants.h"
//#include "webcameramanager.h"
#include <QMessageBox>
#include <QMenu>
#include <QWeakPointer>

VisionView::VisionView(QWidget *parent , Qt::WindowFlags f)
{
    this->setParent(parent, f);
    _enState = VISION_VIEW_STATE::LEARNING;

    _pTimer = std::make_unique<QTimer>(this);
    connect( _pTimer.get(), SIGNAL(timeout()), this, SLOT(updateMat()));

    setContextMenuPolicy(Qt::CustomContextMenu);
    connect( this, SIGNAL(customContextMenuRequested(const QPoint&)),
        this, SLOT(showContextMenu(const QPoint&)));

    _fZoomFactor = 1.0;
    _nCameraID = 0;
    _enMaskEditState = MASK_EDIT_NONE;
    _enMaskShape = MASK_SHAPE_RECT;
    _enMaskEditState = MASK_EDIT_ADD;
    _ptLeftClickStartPos = Point(0,0);
    _ptLeftClickEndPos = Point(0,0);
    _enTestVisionState = TEST_VISION_STATE::UNDEFINED;
}

VisionView::~VisionView()
{
}

int VisionView::updateMat()
{
    cv::Mat mat;

//    AOI::Vision::WebCameraManager::GetInstance()->GrabImage ( _nCameraID, mat );

    _mat = mat;

    Mat display ( mat.size(), mat.type() );
    mat.copyTo( display );
    if ( ! IsEuqal ( _fZoomFactor, 1.0f ) )
    {
        cv::Mat matZoomResult;
        cv::resize( display, matZoomResult, cv::Size(), _fZoomFactor, _fZoomFactor );
        if ( _fZoomFactor > 1.0f )  {
            cv::Rect rectROI( ( matZoomResult.cols - display.cols ) / 2, ( matZoomResult.rows - display.rows ) / 2, display.cols, display.rows );
            cv::Mat matROI ( matZoomResult,  rectROI );
            matROI.copyTo(display);
        }else   {
            cv::Rect rectROI( ( display.cols - matZoomResult.cols ) / 2, ( display.rows - matZoomResult.rows ) / 2, matZoomResult.cols, matZoomResult.rows );
            cv::Mat matTemp = cv::Mat::zeros( display.rows, display.cols, display.type());
            cv::Mat matROI ( matTemp, rectROI );
            matZoomResult.copyTo(matROI);
            matTemp.copyTo(display);
        }
    }
    cvtColor( display, display, CV_BGR2RGB);

    _drawLearnWindow ( display );

    QImage image = QImage((uchar*) display.data, display.cols, display.rows, display.step, QImage::Format_RGB888);

    //show Qimage using QLabel
    setPixmap(QPixmap::fromImage(image));
    return ToInt(STATUS::OK);
}

void VisionView::mousePressEvent(QMouseEvent *event)
{
    if ( Qt::LeftButton == event->button() )
    {
        _bLeftButtonDown = true;
        QPoint point = event->pos();
        _ptLeftClickStartPos.x = point.x();
        _ptLeftClickStartPos.y = point.y();

        if ( VISION_VIEW_STATE::ADD_MASK == _enState && MASK_SHAPE_POLYLINE == _enMaskShape )    {
            if ( _vecPolylinePoint.size() == 0 ) {
                _vecPolylinePoint.push_back(_ptLeftClickStartPos);
            }
        }
    }
}

void VisionView::mouseReleaseEvent(QMouseEvent *event)
{
    if ( Qt::LeftButton == event->button() )
    {
        _bLeftButtonDown = false;

        QPoint qPoint = event->pos();

        cv::Point ptCV(qPoint.x(), qPoint.y() );
        if ( VISION_VIEW_STATE::ADD_MASK == _enState && MASK_SHAPE_POLYLINE == _enMaskShape )    {
            bool bAddPolylineMask = false;
            if ( _vecPolylinePoint.size() > 2 ) {
                float fDistance = distanceOf2Point( ptCV, _vecPolylinePoint[0] );
                if ( fDistance < 10.f ) {
                    ptCV = _vecPolylinePoint[0];
                    bAddPolylineMask = true;
                }
            }
            _vecPolylinePoint.push_back(ptCV);
            if ( bAddPolylineMask ) {
                const cv::Point *pts = (const cv::Point *)Mat(_vecPolylinePoint).data;
                int npts = Mat(_vecPolylinePoint).rows;

                cv::fillPoly( _matMask, &pts, &npts, 1, Scalar ( 255, 0 , 0 ) );

                _vecPolylinePoint.clear();
                _ptLeftClickStartPos = Point(0,0);
                _ptLeftClickEndPos = Point(0,0);
            }
        }else if ( VISION_VIEW_STATE::TEST_VISION_LIBRARY == _enState && TEST_VISION_STATE::SET_CIRCLE_CTR == _enTestVisionState ) {
            _ptCircleCtr = cv::Point ( event->pos().x(), event->pos().y() );
            _drawDisplay();
        }
    }else if ( event->button() == Qt::MidButton )   {
        QMessageBox msgBox;
        msgBox.setText("Middle button single click");
        msgBox.exec();
    }
}

void VisionView::mouseMoveEvent(QMouseEvent *event)
{
    if(event->buttons() & Qt::LeftButton)   //This means left button is pressed.
    {
        //do stuff
        QPoint point = event->pos();
        _ptLeftClickEndPos.x = point.x();
        _ptLeftClickEndPos.y = point.y();

        if ( VISION_VIEW_STATE::LEARNING == _enState )  {
            _rectLrnWindow.x = _ptLeftClickStartPos.x;
            _rectLrnWindow.y = _ptLeftClickStartPos.y;
            _rectLrnWindow.width = _ptLeftClickEndPos.x - _ptLeftClickStartPos.x;
            _rectLrnWindow.height = _ptLeftClickEndPos.y - _ptLeftClickStartPos.y;
        }else if ( VISION_VIEW_STATE::ADD_MASK == _enState )    {
            if ( MASK_SHAPE_RECT == _enMaskShape ) {
                Rect rectMask;
                rectMask.x = _ptLeftClickStartPos.x;
                rectMask.y = _ptLeftClickStartPos.y;
                rectMask.width = _ptLeftClickEndPos.x - _ptLeftClickStartPos.x;
                rectMask.height = _ptLeftClickEndPos.y - _ptLeftClickStartPos.y;

                if ( MASK_EDIT_ADD == _enMaskEditState )
                    cv::rectangle ( _matMask, rectMask, Scalar(255, 255, 255), CV_FILLED, CV_AA, 0 );
                else
                    cv::rectangle ( _matMask, rectMask, Scalar(0, 0, 0 ), CV_FILLED, CV_AA, 0 );
            }else if ( MASK_SHAPE_CIRCLE == _enMaskShape )  {
                float fOffsetX = _ptLeftClickEndPos.x - _ptLeftClickStartPos.x;
                float fOffsetY = _ptLeftClickEndPos.y - _ptLeftClickStartPos.y;
                float fRadius = sqrt ( fOffsetX * fOffsetX + fOffsetY * fOffsetY );
                if ( MASK_EDIT_ADD == _enMaskEditState )
                    cv::circle( _matMask, _ptLeftClickStartPos, fRadius, Scalar(255, 255, 255), CV_FILLED);
                else
                    cv::circle( _matMask, _ptLeftClickStartPos, fRadius, Scalar(0, 0, 0), CV_FILLED);
            }else if ( MASK_SHAPE_POLYLINE == _enMaskShape )  {

            }
            _drawDisplay();
        }else if ( VISION_VIEW_STATE::TEST_VISION_LIBRARY == _enState ) {
            if ( TEST_VISION_STATE::SET_CIRCLE_INNER_RADIUS == _enTestVisionState )
                _fInnerRangeRadius = distanceOf2Point (cv::Point( point.x(), point.y()), _ptCircleCtr);
            else if ( TEST_VISION_STATE::SET_CIRCLE_OUTTER_RADIUS == _enTestVisionState )   {
                float fRadius = distanceOf2Point (cv::Point( point.x(), point.y()), _ptCircleCtr);
                if ( fRadius > _fInnerRangeRadius )
                    _fOutterRangeRadius = fRadius;
            }else if ( TEST_VISION_STATE::SET_RECT_SRCH_WINDOW == _enTestVisionState )  {
                assert ( _nCurrentSrchWindowIndex >= 0 );
                cv::Rect rect ( _ptLeftClickStartPos.x, _ptLeftClickStartPos.y, _ptLeftClickEndPos.x - _ptLeftClickStartPos.x, _ptLeftClickEndPos.y - _ptLeftClickStartPos.y);
                if ( _vecRectSrchWindow.size() <= _nCurrentSrchWindowIndex )
                    _vecRectSrchWindow.push_back ( rect );
                else
                    _vecRectSrchWindow[_nCurrentSrchWindowIndex] = rect;
            }
            _drawDisplay();
        }
    }
}

void VisionView::mouseDoubleClickEvent(QMouseEvent *event)
{
    if ( event->button() & Qt::MidButton )  {
        QMessageBox msgBox;
        msgBox.setText("Middle button double click");
        msgBox.exec();
    }
}

void VisionView::setMachineState(VISION_VIEW_STATE enMachineState)
{
    _enState = enMachineState;
}

void VisionView::_drawLearnWindow(cv::Mat &mat)
{
    cv::rectangle ( mat, _rectLrnWindow, _colorLrnWindow, 2 );

    if ( VISION_VIEW_STATE::ADD_MASK == _enState && MASK_SHAPE_POLYLINE == _enMaskShape  )    {
        if ( _vecPolylinePoint.size() > 1 ) {
            const cv::Point *pts = (const cv::Point *)Mat(_vecPolylinePoint).data;
            int npts = Mat(_vecPolylinePoint).rows;

            cv::polylines( mat, &pts, &npts, 1, false, Scalar ( 255, 0 , 0 ), 1 );
        }

        if ( _vecPolylinePoint.size() > 2 ) {
            float fDistance = distanceOf2Point( _ptLeftClickEndPos, _vecPolylinePoint[0] );
            if ( fDistance < 10.f )
                cv::circle( mat, _vecPolylinePoint[0], 10, Scalar ( 255, 0, 0 ), 1 );
        }

        if ( _ptLeftClickStartPos.x > 0 && _ptLeftClickStartPos.y > 0
             && _ptLeftClickEndPos.x > 0 && _ptLeftClickEndPos.y > 0 )
            cv::line( mat, _ptLeftClickStartPos, _ptLeftClickEndPos, Scalar ( 255, 0, 0 ), 1 );
    }

    cv::Mat copy = mat.clone();

    if ( ! _matMask.empty() )
        copy.setTo ( Scalar(255, 0, 0), _matMask );

    double alpha = 0.2;
    cv::addWeighted ( copy, alpha, mat, 1.0 - alpha, 0.0, mat );
}

void VisionView::_drawTestVisionLibrary(cv::Mat &mat)
{
    if (TEST_VISION_STATE::SET_CIRCLE_CTR == _enTestVisionState  ||
        TEST_VISION_STATE::SET_CIRCLE_INNER_RADIUS == _enTestVisionState ||
        TEST_VISION_STATE::SET_CIRCLE_OUTTER_RADIUS == _enTestVisionState )
    {
        if ( _ptCircleCtr.x > 0 && _ptCircleCtr.y > 0 )
        cv::circle ( mat, _ptCircleCtr, 2, cv::Scalar ( 0, 255, 0 ), 2 );
        if ( _fInnerRangeRadius > 2 )
            cv::circle ( mat, _ptCircleCtr, _fInnerRangeRadius, cv::Scalar ( 0, 255, 0 ), 1 );
        if ( _fOutterRangeRadius > _fInnerRangeRadius )
            cv::circle ( mat, _ptCircleCtr, _fOutterRangeRadius, cv::Scalar ( 0, 255, 0 ), 1 );
    }else if ( TEST_VISION_STATE::SET_RECT_SRCH_WINDOW == _enTestVisionState )
    {
        for ( const auto &rect : _vecRectSrchWindow )
            cv::rectangle ( mat, rect, _colorLrnWindow, 1 );
    }
}

void VisionView::_drawDisplay()
{
    if ( _mat.empty() )
        return;

    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();
    cv::Mat matZoomResult;
    if ( ! IsEuqal ( _fZoomFactor, 1.0f ) )       
        cv::resize( _mat, matZoomResult, cv::Size(), _fZoomFactor, _fZoomFactor );        
    else
        matZoomResult = _mat.clone();

    _matDisplay = cv::Mat::ones( displayHeight, displayWidth, _mat.type() ) * 255;
    _matDisplay.setTo(cv::Scalar(255,255,255));

    if (matZoomResult.cols > displayWidth && matZoomResult.rows > displayHeight)  {
        cv::Rect rectROI((matZoomResult.cols - displayWidth) / 2, (matZoomResult.rows - displayHeight) / 2, displayWidth, displayHeight);
        cv::Mat matROI(matZoomResult, rectROI);
        _matDisplay = matROI;
    }
    else if (matZoomResult.cols > displayWidth && matZoomResult.rows < displayHeight)  {
        cv::Rect rectROISrc( (matZoomResult.cols - displayWidth) / 2, 0, displayWidth, matZoomResult.rows);
        cv::Mat matSrc(matZoomResult, rectROISrc);
        cv::Rect rectROIDst(0, ( displayHeight - matZoomResult.rows ) / 2, displayWidth, matZoomResult.rows );
        cv::Mat matDst(_matDisplay, rectROIDst);
        matSrc.copyTo(matDst);
    }else if (matZoomResult.cols < displayWidth && matZoomResult.rows > displayHeight)  {
        cv::Rect rectROISrc(0, (matZoomResult.rows - displayHeight) / 2, matZoomResult.cols, displayHeight);
        cv::Mat matSrc(matZoomResult, rectROISrc);
        cv::Rect rectROIDst(( displayWidth - matZoomResult.cols ) / 2, 0, matZoomResult.cols, displayHeight );
        cv::Mat matDst(_matDisplay, rectROIDst);
        matSrc.copyTo(matDst);
    }else if (matZoomResult.cols < displayWidth && matZoomResult.rows < displayHeight) {
        cv::Rect rectROIDst((displayWidth - matZoomResult.cols) / 2, (displayHeight - matZoomResult.rows) / 2, matZoomResult.cols, matZoomResult.rows);
        cv::Mat matDst(_matDisplay, rectROIDst);
        matZoomResult.copyTo(matDst);
    }else
        _matDisplay = matZoomResult;

    cvtColor( _matDisplay, _matDisplay, CV_BGR2RGB);

    if (VISION_VIEW_STATE::LEARNING == _enState ||
        VISION_VIEW_STATE::ADD_MASK == _enState)
        _drawLearnWindow ( _matDisplay );
    else if ( VISION_VIEW_STATE::TEST_VISION_LIBRARY == _enState )
        _drawTestVisionLibrary ( _matDisplay );

    QImage image = QImage((uchar*) _matDisplay.data, _matDisplay.cols, _matDisplay.rows, _matDisplay.step, QImage::Format_RGB888);
    //show Qimage using QLabel
    setPixmap(QPixmap::fromImage(image));
}

void VisionView::showContextMenu(const QPoint& pos) // this is a slot
{
    if ( _matDisplay.empty() )
        return;

    QPoint globalPos = mapToGlobal(pos);

    QMenu contextMenu(tr("Context menu"), this);

    QAction action1("Add Mask", this);
    connect(&action1, SIGNAL(triggered()), this, SLOT(addMask()));
    contextMenu.addAction(&action1);

    QAction* selectedItem = contextMenu.exec(globalPos);
    if (selectedItem)
    {
        // something was chosen, do stuff
    }
    else
    {
        // nothing was chosen
    }
}

void VisionView::addMask()
{
    _enState = VISION_VIEW_STATE::ADD_MASK;
    if ( _matMask.empty() )
        _matMask = cv::Mat( _matDisplay.size(), CV_8UC1, cv::Scalar(0, 0, 0) );

    if ( ! _pDialogEditMask)    {
        _pDialogEditMask = std::make_unique<DialogEditMask>(this);
        _pDialogEditMask->setVisionView(this);
    }

    _pDialogEditMask->show();
    _pDialogEditMask->raise();
}

void VisionView::zoomIn()
{
    if ( _fZoomFactor < _constMaxZoomFactor )
        _fZoomFactor *= 2.f;
    _drawDisplay();
}

void VisionView::zoomOut()
{
    if ( _fZoomFactor > _constMinZoomFactor )
        _fZoomFactor /= 2.f;
    _drawDisplay();
}

void VisionView::restoreZoom()
{
    _fZoomFactor = 1.0;
}

void VisionView::setMaskEditState(MASK_EDIT_STATE enMaskEditState)
{
    _enMaskEditState = enMaskEditState;
}

void VisionView::setMaskShape(MASK_SHAPE enMaskShape)
{
    _enMaskShape = enMaskShape;
}

float VisionView::distanceOf2Point(const cv::Point &pt1, const cv::Point &pt2)
{
    float fOffsetX = pt1.x - pt2.x;
    float fOffsetY = pt1.y - pt2.y;
    float fDistance = sqrt ( fOffsetX * fOffsetX + fOffsetY * fOffsetY );
    return fDistance;
}

void VisionView::startTimer()
{
    _pTimer->start(20);
}

void VisionView::setImageFile(const std::string &filePath)
{
    _mat = cv::imread(filePath);
    _drawDisplay();
}

void VisionView::setMat(const cv::Mat &mat)
{
    _mat = mat;
    _drawDisplay();
}

void VisionView::setTestVisionState(TEST_VISION_STATE enState)
{
    _enTestVisionState = enState;
    if ( TEST_VISION_STATE::UNDEFINED == _enTestVisionState )   {
        _ptCircleCtr.x = 0;
        _ptCircleCtr.y = 0;
        _fInnerRangeRadius = 0.f;
        _fOutterRangeRadius = 0.f;
        _drawDisplay();
    }
}

void VisionView::getFitCircleRange(cv::Point &ptCtr, float &fInnterRadius, float &fOutterRadius) const
{
    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();
    
    ptCtr.x = _ptCircleCtr.x - ( displayWidth - _mat.cols ) / 2;
    ptCtr.y = _ptCircleCtr.y - ( displayHeight - _mat.rows ) / 2;
    fInnterRadius = _fInnerRangeRadius;
    fOutterRadius = _fOutterRangeRadius;
}

VectorOfRect VisionView::getVecSrchWindow() const
{
    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();
    VectorOfRect vecResult;
    for ( auto rect : _vecRectSrchWindow )
    {
        if ( ( displayWidth - _mat.cols * _fZoomFactor ) > 0 )
            rect.x      = ( rect.x - ( displayWidth - _mat.cols * _fZoomFactor ) / 2 ) / _fZoomFactor;
        if ( ( displayHeight - _mat.rows * _fZoomFactor ) > 0 )
            rect.y      = ( rect.y - ( displayWidth - _mat.rows * _fZoomFactor ) / 2 ) / _fZoomFactor;
        rect.width  /= _fZoomFactor;
        rect.height /= _fZoomFactor;
        vecResult.push_back ( rect );
    }
    return vecResult;
}

void VisionView::setCurrentSrchWindowIndex(int nIndex)
{
    _nCurrentSrchWindowIndex = nIndex;
}