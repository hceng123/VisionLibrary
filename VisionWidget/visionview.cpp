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
    _enState = VISION_VIEW_STATE::IDLE;

    _pTimer = std::make_unique<QTimer>(this);
    connect( _pTimer.get(), SIGNAL(timeout()), this, SLOT(updateMat()));

    setContextMenuPolicy(Qt::CustomContextMenu);
    connect( this, SIGNAL(customContextMenuRequested(const QPoint&)),
        this, SLOT(showContextMenu(const QPoint&)));

    _fZoomFactor = 1.0;
    _nCameraID = 0;
    _enMaskEditState = MASK_EDIT_STATE::NONE;
    _enMaskShape = MASK_SHAPE_RECT;
    _enMaskEditState = MASK_EDIT_STATE::ADD;
    _ptLeftClickStartPos = Point(0,0);
    _ptLeftClickEndPos = Point(0,0);
    _enTestVisionState = TEST_VISION_STATE::IDLE;
    _enDisplaySource = DISPLAY_SOURCE::ORIGINAL;
    _bDisplayGrayScale = false;
    _bDisplayBinary = false;
}

VisionView::~VisionView()
{
}

int VisionView::updateMat()
{
    cv::Mat mat;
    //AOI::Vision::WebCameraManager::GetInstance()->GrabImage ( _nCameraID, mat );
    _matArray[ToInt32(DISPLAY_SOURCE::ORIGINAL)] = mat;
    _enDisplaySource = DISPLAY_SOURCE::ORIGINAL;
    _drawDisplay();
    return ToInt(STATUS::OK);
}

void VisionView::mousePressEvent(QMouseEvent *event)
{
    if ( DISPLAY_SOURCE::RESULT == _enDisplaySource )
        return;

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
        }else if ( VISION_VIEW_STATE::MOVE == _enState )    {
            setCursor(Qt::ClosedHandCursor);
        }
    }
}

void VisionView::mouseReleaseEvent(QMouseEvent *event)
{
    if ( DISPLAY_SOURCE::RESULT == _enDisplaySource )
        return;

    if ( Qt::LeftButton == event->button() )
    {
        _bLeftButtonDown = false;

        QPoint qPoint = event->pos();
        cv::Point ptCV(qPoint.x(), qPoint.y() );

        if ( VISION_VIEW_STATE::ADD_MASK == _enState && MASK_SHAPE_POLYLINE == _enMaskShape )    {
            bool bPolygonFinished = false;
            if ( _vecPolylinePoint.size() > 2 ) {
                float fDistance = distanceOf2Point( ptCV, _vecPolylinePoint[0] );
                if ( fDistance < 10.f ) {
                    ptCV = _vecPolylinePoint[0];
                    bPolygonFinished = true;
                }
            }
            _vecPolylinePoint.push_back(ptCV);
            if ( bPolygonFinished ) {
                const cv::Point *pts = (const cv::Point *)Mat(_vecPolylinePoint).data;
                int npts = Mat(_vecPolylinePoint).rows;

                if ( MASK_EDIT_STATE::ADD == _enMaskEditState )
                    cv::fillPoly( _matMaskForDisplay, &pts, &npts, 1, Scalar ( 255, 255, 255 ) );
                else
                    cv::fillPoly( _matMaskForDisplay, &pts, &npts, 1, Scalar ( 0, 0, 0 ) );

                _vecPolylinePoint.clear();
                _ptLeftClickStartPos = Point(0,0);
                _ptLeftClickEndPos = Point(0,0);
            }
        }else if ( VISION_VIEW_STATE::TEST_VISION_LIBRARY == _enState && TEST_VISION_STATE::SET_CIRCLE_CTR == _enTestVisionState ) {
            _ptCircleCtr = cv::Point ( event->pos().x(), event->pos().y() );            
        }else if ( VISION_VIEW_STATE::MOVE == _enState )    {
            setCursor(Qt::OpenHandCursor);
        }
        _drawDisplay();
    }else if ( event->button() == Qt::MidButton )   {
        QMessageBox msgBox;
        msgBox.setText("Middle button single click");
        msgBox.exec();
    }
}

void VisionView::mouseMoveEvent(QMouseEvent *event)
{
    if ( DISPLAY_SOURCE::RESULT == _enDisplaySource )
        return;

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
            _drawDisplay();
        }else if ( VISION_VIEW_STATE::ADD_MASK == _enState )    {
            if ( MASK_SHAPE_RECT == _enMaskShape ) {
                Rect rectMask;
                rectMask.x = _ptLeftClickStartPos.x;
                rectMask.y = _ptLeftClickStartPos.y;
                rectMask.width = _ptLeftClickEndPos.x - _ptLeftClickStartPos.x;
                rectMask.height = _ptLeftClickEndPos.y - _ptLeftClickStartPos.y;

                if ( MASK_EDIT_STATE::ADD == _enMaskEditState )
                    cv::rectangle ( _matMaskForDisplay, rectMask, Scalar(255, 255, 255), CV_FILLED, CV_AA );
                else
                    cv::rectangle ( _matMaskForDisplay, rectMask, Scalar(0, 0, 0 ), CV_FILLED, CV_AA );
            }else if ( MASK_SHAPE_CIRCLE == _enMaskShape )  {
                float fOffsetX = _ptLeftClickEndPos.x - _ptLeftClickStartPos.x;
                float fOffsetY = _ptLeftClickEndPos.y - _ptLeftClickStartPos.y;
                float fRadius = sqrt ( fOffsetX * fOffsetX + fOffsetY * fOffsetY );
                if ( MASK_EDIT_STATE::ADD == _enMaskEditState )
                    cv::circle( _matMaskForDisplay, _ptLeftClickStartPos, fRadius, Scalar(255, 255, 255), CV_FILLED);
                else
                    cv::circle( _matMaskForDisplay, _ptLeftClickStartPos, fRadius, Scalar(0, 0, 0), CV_FILLED);
            }else if ( MASK_SHAPE_POLYLINE == _enMaskShape )  {

            }
            _drawDisplay();
        }else if ( VISION_VIEW_STATE::TEST_VISION_LIBRARY == _enState ) {
            if ( TEST_VISION_STATE::IDLE == _enTestVisionState )    {
                _rectSelectedWindow.x = _ptLeftClickStartPos.x;
                _rectSelectedWindow.y = _ptLeftClickStartPos.y;
                _rectSelectedWindow.width = _ptLeftClickEndPos.x - _ptLeftClickStartPos.x;
                _rectSelectedWindow.height = _ptLeftClickEndPos.y - _ptLeftClickStartPos.y;
                _drawDisplay();
            }
            else if ( TEST_VISION_STATE::SET_CIRCLE_INNER_RADIUS == _enTestVisionState )
                _fInnerRangeRadius = distanceOf2Point (cv::Point( point.x(), point.y()), _ptCircleCtr);
            else if ( TEST_VISION_STATE::SET_CIRCLE_OUTTER_RADIUS == _enTestVisionState )   {
                float fRadius = distanceOf2Point (cv::Point( point.x(), point.y()), _ptCircleCtr);
                if ( fRadius > _fInnerRangeRadius )
                    _fOutterRangeRadius = fRadius;
            }else if ( TEST_VISION_STATE::SET_RECT_SRCH_WINDOW == _enTestVisionState )  {
                assert ( _nCurrentSrchWindowIndex >= 0 );
                cv::Rect rect ( _ptLeftClickStartPos.x, _ptLeftClickStartPos.y, _ptLeftClickEndPos.x - _ptLeftClickStartPos.x, _ptLeftClickEndPos.y - _ptLeftClickStartPos.y);
                if ( ToInt(_vecRectSrchWindow.size()) <= _nCurrentSrchWindowIndex )
                    _vecRectSrchWindow.push_back ( rect );
                else
                    _vecRectSrchWindow[_nCurrentSrchWindowIndex] = rect;
            }
            else if ( TEST_VISION_STATE::SET_LINE == _enTestVisionState )   {
                _lineOfIntensityCheck.pt1 = _ptLeftClickStartPos;
                _lineOfIntensityCheck.pt2 = _ptLeftClickEndPos;
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

void VisionView::setState(VISION_VIEW_STATE enState)
{
    _enState = enState;
}

void VisionView::_drawLearnWindow(cv::Mat &mat)
{
    cv::rectangle ( mat, _rectLrnWindow, _colorLrnWindow, 2 );

    cv::Scalar scalarMask( 0, 0, 255 );
    if ( VISION_VIEW_STATE::ADD_MASK == _enState && MASK_SHAPE_POLYLINE == _enMaskShape  )    {
        if ( _vecPolylinePoint.size() > 1 ) {
            const cv::Point *pts = (const cv::Point *)Mat(_vecPolylinePoint).data;
            int npts = Mat(_vecPolylinePoint).rows;

            cv::polylines( mat, &pts, &npts, 1, false, scalarMask, 1 );
        }

        if ( _vecPolylinePoint.size() > 2 ) {
            float fDistance = distanceOf2Point( _ptLeftClickEndPos, _vecPolylinePoint[0] );
            if ( fDistance < 10.f )
                cv::circle( mat, _vecPolylinePoint[0], 10, scalarMask, 1 );
        }

        if ( _ptLeftClickStartPos.x > 0 && _ptLeftClickStartPos.y > 0
             && _ptLeftClickEndPos.x > 0 && _ptLeftClickEndPos.y > 0 )
            cv::line( mat, _ptLeftClickStartPos, _ptLeftClickEndPos, scalarMask, 1 );
    }

    cv::Mat copy = mat.clone();

    if ( ! _matMaskForDisplay.empty() )
        copy.setTo ( scalarMask, _matMaskForDisplay );

    double alpha = 0.4;
    cv::addWeighted ( copy, alpha, mat, 1.0 - alpha, 0.0, mat );
}

void VisionView::_drawSelectedWindow(cv::Mat &mat)
{
    if ( _rectSelectedWindow.width > 0 && _rectSelectedWindow.height > 0 )
        cv::rectangle ( mat, _rectSelectedWindow, _colorLrnWindow, 2 );

    cv::Scalar scalarMask(0, 0, 255);
    if ( VISION_VIEW_STATE::ADD_MASK == _enState && MASK_SHAPE_POLYLINE == _enMaskShape  )    {
        if ( _vecPolylinePoint.size() > 1 ) {
            const cv::Point *pts = (const cv::Point *)Mat(_vecPolylinePoint).data;
            int npts = Mat(_vecPolylinePoint).rows;

            cv::polylines( mat, &pts, &npts, 1, false, scalarMask , 1 );
        }

        if ( _vecPolylinePoint.size() > 2 ) {
            float fDistance = distanceOf2Point( _ptLeftClickEndPos, _vecPolylinePoint[0] );
            if ( fDistance < 10.f )
                cv::circle( mat, _vecPolylinePoint[0], 10, scalarMask, 1 );
        }

        if ( _ptLeftClickStartPos.x > 0 && _ptLeftClickStartPos.y > 0
             && _ptLeftClickEndPos.x > 0 && _ptLeftClickEndPos.y > 0 )
            cv::line( mat, _ptLeftClickStartPos, _ptLeftClickEndPos, scalarMask, 1 );
    }

    cv::Mat copy = mat.clone();

    if ( ! _matMaskForDisplay.empty() )
        copy.setTo ( scalarMask, _matMaskForDisplay );

    double alpha = 0.2;
    cv::addWeighted ( copy, alpha, mat, 1.0 - alpha, 0.0, mat );
}

void VisionView::_drawTestVisionLibrary(cv::Mat &mat)
{
    if ( TEST_VISION_STATE::IDLE == _enTestVisionState )
        _drawSelectedWindow (mat);
    else if (TEST_VISION_STATE::SET_CIRCLE_CTR == _enTestVisionState  ||
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
    }else if ( TEST_VISION_STATE::SET_LINE == _enTestVisionState )  {
        cv::line ( mat, _lineOfIntensityCheck.pt1, _lineOfIntensityCheck.pt2, _colorLrnWindow, 1 );
    }
}

void VisionView::_drawDisplay()
{
    cv::Mat mat = _matArray[ToInt32(_enDisplaySource)];

    if ( mat.empty() )
        return;

    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();
    cv::Mat matZoomResult;
    if ( ! IsEuqal ( _fZoomFactor, 1.0f ) )
        cv::resize( mat, matZoomResult, cv::Size(), _fZoomFactor, _fZoomFactor );
    else
        matZoomResult = mat.clone();

    _matDisplay = cv::Mat::ones( displayHeight, displayWidth, mat.type() ) * 255;
    _matDisplay.setTo(cv::Scalar(255, 255, 255));

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
    
    //cv::Rect rectChangeROI = _rectSelectedWindow;
    //if ( _rectSelectedWindow.width > 0 && _rectSelectedWindow.height > 0 )  {
    //    if ( rectChangeROI.x < displayWidth  / 2 - matZoomResult.cols / 2 ) rectChangeROI.x = displayWidth  / 2 - matZoomResult.cols / 2;
    //    if ( rectChangeROI.y < displayHeight / 2 - matZoomResult.rows / 2 ) rectChangeROI.y = displayHeight / 2 - matZoomResult.rows / 2;
    //    if ( ( rectChangeROI.x + rectChangeROI.width  ) > ( displayWidth   / 2 + matZoomResult.cols / 2 ) )  rectChangeROI.width  = ( displayWidth   / 2 + matZoomResult.cols / 2 ) - rectChangeROI.x;
    //    if ( ( rectChangeROI.y + rectChangeROI.height ) > ( displayHeight  / 2 + matZoomResult.rows / 2 ) )  rectChangeROI.height = ( displayHeight  / 2 + matZoomResult.rows / 2 ) - rectChangeROI.y;
    //}
    //if ( rectChangeROI.width > 0 && rectChangeROI.height > 0 )  {
    //    cv::Mat matSelected ( _matDisplay, _rectSelectedWindow );
    //    cv::Mat matResult = _generateDisplayImage ( matSelected );
    //    matResult.copyTo ( matSelected );
    //}
    //else
    //{
    //    cv::Mat matResult = _generateDisplayImage ( _matDisplay );
    //    _matDisplay = matResult;
    //}

    //if (VISION_VIEW_STATE::LEARNING == _enState ||
    //    VISION_VIEW_STATE::ADD_MASK == _enState)
    //    _drawLearnWindow ( _matDisplay );
    //else if ( VISION_VIEW_STATE::TEST_VISION_LIBRARY == _enState )
        _drawTestVisionLibrary ( _matDisplay );

    cvtColor( _matDisplay, _matDisplay, CV_BGR2RGB);
    QImage image = QImage((uchar*) _matDisplay.data, _matDisplay.cols, _matDisplay.rows, ToInt(_matDisplay.step), QImage::Format_RGB888);
        
    //show Qimage using QLabel
    setPixmap(QPixmap::fromImage(image));
}

void VisionView::showContextMenu(const QPoint& pos) // this is a slot
{
    if ( _matDisplay.empty() )
        return;

    QPoint globalPos = mapToGlobal(pos);

    QMenu contextMenu(tr("Context menu"), this);

    QAction actionS("Select", this);
    connect(&actionS, SIGNAL(triggered()), this, SLOT(selectWindow()));
    contextMenu.addAction(&actionS);

    QAction actionA("Add Mask", this);
    connect(&actionA, SIGNAL(triggered()), this, SLOT(addMask()));
    contextMenu.addAction(&actionA);

    QAction actionC("Clear Selection", this);
    connect(&actionC, SIGNAL(triggered()), this, SLOT(clearSelectedWindow()));
    contextMenu.addAction(&actionC);

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

void VisionView::selectWindow()
{
    _enState = VISION_VIEW_STATE::TEST_VISION_LIBRARY;
}

void VisionView::addMask()
{
    _enState = VISION_VIEW_STATE::ADD_MASK;
    if ( _matMaskForDisplay.empty() )
        _matMaskForDisplay = cv::Mat( _matDisplay.size(), CV_8UC1, cv::Scalar(0, 0, 0) );

    if ( ! _pDialogEditMask)    {
        _pDialogEditMask = std::make_unique<DialogEditMask>(this);
        _pDialogEditMask->setVisionView(this);
    }

    _pDialogEditMask->show();
    _pDialogEditMask->raise();
}

void VisionView::clearSelectedWindow()
{
    _rectSelectedWindow.width  = -1;
    _rectSelectedWindow.height = -1;
    _drawDisplay();
}

void VisionView::swapImage()
{
    if ( DISPLAY_SOURCE::RESULT == _enDisplaySource )   {
        _enDisplaySource = DISPLAY_SOURCE::ORIGINAL;
        _drawDisplay();
    }
    else if ( DISPLAY_SOURCE::ORIGINAL == _enDisplaySource && ! _matArray[ToInt32(DISPLAY_SOURCE::RESULT)].empty() ) {
        _enDisplaySource = DISPLAY_SOURCE::RESULT;
        _drawDisplay();
    }
}

void VisionView::_zoomPoint(cv::Point &point, float fZoomFactor)
{
    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();

    point.x = (point.x - displayWidth / 2)  * fZoomFactor + displayWidth / 2;
    point.y = (point.y - displayHeight / 2) * fZoomFactor + displayHeight / 2;
}

void VisionView::_zoomRect(cv::Rect &rect, float fZoomFactor)
{
    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();

    rect.x = (rect.x - displayWidth / 2)  * fZoomFactor + displayWidth / 2;
    rect.y = (rect.y - displayHeight / 2) * fZoomFactor + displayHeight / 2;
    rect.width *= fZoomFactor;
    rect.height *= fZoomFactor;
}

/*static*/ void VisionView::_zoomMat(const cv::Mat &mat, cv::Mat &matOutput, float fZoomFactor, bool bKeepSize )
{
    if ( mat.empty() || fabs ( fZoomFactor - 1 ) < 0.001 )
        return;

    cv::Mat matZoomResult;
    cv::resize ( mat, matZoomResult, cv::Size(), fZoomFactor, fZoomFactor );
    if ( ! bKeepSize )  {
        matOutput = matZoomResult;
        return;
    }

    cv::Mat matResult = cv::Mat::zeros ( mat.size(), mat.type() );
    
    if ( matResult.rows >= matZoomResult.rows && matResult.cols >= matZoomResult.cols ) {
        cv::Rect rectROIDst( ( matResult.cols - matZoomResult.cols ) / 2, ( matResult.rows - matZoomResult.rows ) / 2, matZoomResult.cols, matZoomResult.rows );
        cv::Mat matDst ( matResult, rectROIDst);
        matZoomResult.copyTo ( matDst );
    }else if  ( matResult.rows <= matZoomResult.rows && matResult.cols <= matZoomResult.cols )  {
        cv::Rect rectROISrc ( ( matZoomResult.cols - matResult.cols ) / 2, ( matZoomResult.rows - matResult.rows ) / 2, matResult.cols, matResult.rows );
        cv::Mat matSrc( matZoomResult, rectROISrc);
        matSrc.copyTo ( matResult );
    }
    matOutput = matResult;
}

void VisionView::zoomIn()
{
    if ( _fZoomFactor < _constMaxZoomFactor )   {
        _fZoomFactor *= 2.f;

        _zoomPoint(_ptCircleCtr, 2.f);
        _fInnerRangeRadius *= 2.f;
        _fOutterRangeRadius *= 2.f;

        for (auto &rect : _vecRectSrchWindow)
            _zoomRect(rect, 2.f);

        _zoomRect(_rectSelectedWindow, 2.f);

        _zoomMat(_matMaskForDisplay, _matMaskForDisplay, 2.f, true);
    }
    _drawDisplay();
}

void VisionView::zoomOut()
{
    if ( _fZoomFactor > _constMinZoomFactor )   {
        _fZoomFactor *= 0.5f;

        _zoomPoint(_ptCircleCtr, 0.5f);
        _fInnerRangeRadius *= 0.5f;
        _fOutterRangeRadius *= 0.5f;

        for (auto &rect : _vecRectSrchWindow)
            _zoomRect(rect, 0.5f);

        _zoomRect(_rectSelectedWindow, 0.5f);

         _zoomMat(_matMaskForDisplay, _matMaskForDisplay, 0.5f, true);
    }
    _drawDisplay();
}

void VisionView::startToMove()
{
    setCursor(Qt::OpenHandCursor);
    _enState = VISION_VIEW_STATE::MOVE;
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

void VisionView::clearMask()
{
    _matMaskForDisplay.release();
}

void VisionView::startTimer()
{
    _pTimer->start(20);
}


void VisionView::setMat( DISPLAY_SOURCE enSource, const cv::Mat &mat)
{
    _enDisplaySource = enSource;
    _matArray[ToInt32(_enDisplaySource)] = mat;
    _drawDisplay();
}

cv::Mat VisionView::getMat(DISPLAY_SOURCE enSource) const
{
    return _matArray[ToInt32(enSource)];
}

cv::Mat VisionView::getCurrentMat() const
{
    return _matArray[ToInt32(_enDisplaySource)];
}

cv::Mat VisionView::getMask() const
{
    if ( _matMaskForDisplay.empty() )
        return _matMaskForDisplay;

    cv::Size size( _matArray[0].size().width / _fZoomFactor, _matArray[0].size().height / _fZoomFactor );
    cv::Mat matMaskResult = cv::Mat::zeros(_matArray[0].size(), CV_8UC1) * PR_MAX_GRAY_LEVEL;

    cv::Mat  matLocalMask;
    if ( fabs ( _fZoomFactor - 1.f) < 0.001 )   {
        matLocalMask = _matMaskForDisplay.clone();
    }
    else
    {
        _zoomMat ( _matMaskForDisplay, matLocalMask, 1 / _fZoomFactor, false );
    }

    if ( matMaskResult.rows >= matLocalMask.rows && matMaskResult.cols >= matLocalMask.cols ) {
        cv::Rect rectROIDst((matMaskResult.cols - matLocalMask.cols) / 2, (matMaskResult.rows - matLocalMask.rows) / 2, matLocalMask.cols, matLocalMask.rows );
        cv::Mat matDst(matMaskResult, rectROIDst);
        matLocalMask.copyTo(matDst);
    }else if  ( matMaskResult.rows < matLocalMask.rows && matMaskResult.cols < matLocalMask.cols )  {
        cv::Rect rectROISrc(( matLocalMask.cols - matMaskResult.cols ) / 2, ( matLocalMask.rows - matMaskResult.rows) / 2, matMaskResult.cols, matMaskResult.rows );
        cv::Mat matSrc( matLocalMask, rectROISrc);
        matSrc.copyTo ( matMaskResult );
    }else if ( matMaskResult.rows > matLocalMask.rows && matMaskResult.cols < matLocalMask.cols )   {
        cv::Rect rectROISrc( (matLocalMask.cols - matMaskResult.cols) / 2, 0, matMaskResult.cols, matLocalMask.rows );
        cv::Mat matSrc(matLocalMask, rectROISrc);
        cv::Rect rectROIDst(0, ( matMaskResult.rows - matLocalMask.rows ) / 2, matMaskResult.cols, matLocalMask.rows );
        cv::Mat matDst(_matDisplay, rectROIDst);
        matSrc.copyTo(matDst);
    }else if ( matMaskResult.rows < matLocalMask.rows && matMaskResult.cols > matLocalMask.cols )   {
        cv::Rect rectROISrc( 0, (matLocalMask.rows - matMaskResult.rows) / 2, matLocalMask.cols, matMaskResult.rows );
        cv::Mat matSrc(matLocalMask, rectROISrc);
        cv::Rect rectROIDst( ( matMaskResult.cols - matLocalMask.cols ) / 2, 0,  matLocalMask.cols, matMaskResult.rows );
        cv::Mat matDst(_matDisplay, rectROIDst);
        matSrc.copyTo(matDst);
    }

    matMaskResult = cv::Scalar(255) - matMaskResult;
    return matMaskResult;
}

void VisionView::applyIntermediateResult()
{
    if ( !_matArray[ToInt32(DISPLAY_SOURCE::INTERMEDIATE)].empty() )
        _matArray[ToInt32(DISPLAY_SOURCE::ORIGINAL)] = _matArray[ToInt32(DISPLAY_SOURCE::INTERMEDIATE)];
}

void VisionView::clearMat( DISPLAY_SOURCE enSource )
{
    _matArray[ToInt32(enSource)].release();
}

bool VisionView::isDisplayResultImage() const
{
    return DISPLAY_SOURCE::RESULT == _enDisplaySource;
}

bool VisionView::isDisplayGrayScale() const
{
    return _bDisplayGrayScale;
}

void VisionView::setTestVisionState(TEST_VISION_STATE enState)
{
    _enTestVisionState = enState;
    if ( TEST_VISION_STATE::IDLE == _enTestVisionState )   {
        _ptCircleCtr.x = 0;
        _ptCircleCtr.y = 0;
        _fInnerRangeRadius = 0.f;
        _fOutterRangeRadius = 0.f;
        _vecRectSrchWindow.clear();

        _drawDisplay();
    }
}

void VisionView::setImageDisplayMode(bool bDisplayGrayScale, bool bDisplayBinary)
{
    _bDisplayGrayScale = bDisplayGrayScale;
    _bDisplayBinary    = bDisplayBinary;
    _drawDisplay();
}

void VisionView::setRGBRatio(float fRatioR, float fRatioG, float fRatioB)
{
    _stRGBRatio.fRatioR = fRatioR;
    _stRGBRatio.fRatioG = fRatioG;
    _stRGBRatio.fRatioB = fRatioB;
    _drawDisplay();
}

void VisionView::getRGBRatio(float &fRatioR, float &fRatioG, float &fRatioB)
{
    fRatioR = _stRGBRatio.fRatioR;
    fRatioG = _stRGBRatio.fRatioG;
    fRatioB = _stRGBRatio.fRatioB;
}

void VisionView::setBinaryThreshold(int nThreshold, bool bReverseThres)
{
    _nThreshold = nThreshold;
    _bReverseThres = bReverseThres;
    _drawDisplay();
}

void VisionView::getFitCircleRange(cv::Point &ptCtr, float &fInnterRadius, float &fOutterRadius) const
{
    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();    
    
    cv::Mat mat = _matArray[ToInt32(_enDisplaySource)];
    ptCtr.x = ( _ptCircleCtr.x - ( displayWidth  - mat.cols * _fZoomFactor ) / 2 ) / _fZoomFactor;
    ptCtr.y = ( _ptCircleCtr.y - ( displayHeight - mat.rows * _fZoomFactor ) / 2 ) / _fZoomFactor;
    fInnterRadius = _fInnerRangeRadius / _fZoomFactor;
    fOutterRadius = _fOutterRangeRadius/ _fZoomFactor;
}

VectorOfRect VisionView::getVecSrchWindow() const
{
    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();
    VectorOfRect vecResult;
    cv::Mat mat = _matArray[ToInt32(_enDisplaySource)];
    for ( auto rect : _vecRectSrchWindow )
    {
        rect.x = (rect.x - (displayWidth  - mat.cols * _fZoomFactor) / 2) / _fZoomFactor;
        rect.y = (rect.y - (displayHeight - mat.rows * _fZoomFactor) / 2) / _fZoomFactor;
        rect.width  /= _fZoomFactor;
        rect.height /= _fZoomFactor;
        vecResult.push_back ( rect );
    }
    return vecResult;
}

cv::Rect VisionView::getSelectedWindow() const
{
    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();
    auto rect = _rectSelectedWindow;
    cv::Mat mat = _matArray[ToInt32(_enDisplaySource)];

    if ( rect.width <= 0 || rect.height <= 0 )  {
        rect = cv::Rect ( 0, 0, mat.cols, mat.rows );
        return rect;
    }

    rect.x = (rect.x - (displayWidth  - mat.cols * _fZoomFactor) / 2) / _fZoomFactor;
    rect.y = (rect.y - (displayHeight - mat.rows * _fZoomFactor) / 2) / _fZoomFactor;
    rect.width  /= _fZoomFactor;
    rect.height /= _fZoomFactor;
    return rect;
}

PR_Line VisionView::getIntensityCheckLine() const
{
    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();
    PR_Line line;
    cv::Mat mat = _matArray[ToInt32(_enDisplaySource)];

    line.pt1.x = (_lineOfIntensityCheck.pt1.x - (displayWidth  - mat.cols * _fZoomFactor) / 2) / _fZoomFactor;
    line.pt1.y = (_lineOfIntensityCheck.pt1.y - (displayHeight - mat.rows * _fZoomFactor) / 2) / _fZoomFactor;
    line.pt2.x = (_lineOfIntensityCheck.pt2.x - (displayWidth  - mat.cols * _fZoomFactor) / 2) / _fZoomFactor;
    line.pt2.y = (_lineOfIntensityCheck.pt2.y - (displayHeight - mat.rows * _fZoomFactor) / 2) / _fZoomFactor;
    return line;
}

void VisionView::setCurrentSrchWindowIndex(int nIndex)
{
    _nCurrentSrchWindowIndex = nIndex;
}