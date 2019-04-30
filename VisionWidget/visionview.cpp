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
    _szDisplayCenterOffset = cv::Size(0, 0);
    _szConfirmedCenterOffset = cv::Size(0, 0);    
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
    //if ( DISPLAY_SOURCE::RESULT == _enDisplaySource )
    //    return;

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
        }else if ( VISION_VIEW_STATE::TEST_VISION_LIBRARY == _enState )  {
            if ( TEST_VISION_STATE::PICK_COLOR == _enTestVisionState )  {
                emit PickColor();
            }
        }
    }
}

void VisionView::mouseReleaseEvent(QMouseEvent *event)
{
    //if ( DISPLAY_SOURCE::RESULT == _enDisplaySource )
    //    return;

    if ( Qt::LeftButton == event->button() )
    {
        _bLeftButtonDown = false;

        QPoint qPoint = event->pos();
        cv::Point ptCV(qPoint.x(), qPoint.y() );

        if ( VISION_VIEW_STATE::ADD_MASK == _enState )    {
            if ( MASK_SHAPE_POLYLINE == _enMaskShape )  {
                bool bPolygonFinished = false;
                if (_vecPolylinePoint.size() > 2) {
                    float fDistance = distanceOf2Point(ptCV, _vecPolylinePoint[0]);
                    if (fDistance < 10.f) {
                        ptCV = _vecPolylinePoint[0];
                        bPolygonFinished = true;
                    }
                }
                _vecPolylinePoint.push_back(ptCV);
                if (bPolygonFinished) {
                    const cv::Point *pts = (const cv::Point *)Mat(_vecPolylinePoint).data;
                    int npts = Mat(_vecPolylinePoint).rows;

                    if (MASK_EDIT_STATE::ADD == _enMaskEditState)
                        cv::fillPoly(_matMaskForDisplay, &pts, &npts, 1, cv::Scalar(255) );
                    else
                        cv::fillPoly(_matMaskForDisplay, &pts, &npts, 1, cv::Scalar(0) );

                    _vecPolylinePoint.clear();
                    _ptLeftClickStartPos = Point(0, 0);
                    _ptLeftClickEndPos = Point(0, 0);
                }
            }
            _applyMask();
            _drawDisplay();
        }else if ( VISION_VIEW_STATE::TEST_VISION_LIBRARY == _enState && TEST_VISION_STATE::SET_CIRCLE_CTR == _enTestVisionState ) {
            _ptCircleCtr = cv::Point ( event->pos().x(), event->pos().y() );
            _ptCircleCtr.x -= _szDisplayCenterOffset.width;
            _ptCircleCtr.y -= _szDisplayCenterOffset.height;
            _drawDisplay();
        }else if ( VISION_VIEW_STATE::MOVE == _enState )    {
            setCursor(Qt::OpenHandCursor);
            _szConfirmedCenterOffset = _szDisplayCenterOffset;
        }        
    }else if ( event->button() == Qt::MidButton )   {
        QMessageBox msgBox;
        msgBox.setText("Middle button single click");
        msgBox.exec();
    }
}

void VisionView::mouseMoveEvent(QMouseEvent *event)
{
    //if ( DISPLAY_SOURCE::RESULT == _enDisplaySource )
    //    return;

    if(event->buttons() & Qt::LeftButton)   //This means left button is pressed.
    {
        //do stuff
        QPoint point = event->pos();
        _ptLeftClickEndPos.x = point.x();
        _ptLeftClickEndPos.y = point.y();

        if ( VISION_VIEW_STATE::LEARNING == _enState ) {
            _rectLrnWindow.x = _ptLeftClickStartPos.x;
            _rectLrnWindow.y = _ptLeftClickStartPos.y;
            _rectLrnWindow.width = _ptLeftClickEndPos.x - _ptLeftClickStartPos.x;
            _rectLrnWindow.height = _ptLeftClickEndPos.y - _ptLeftClickStartPos.y;
            _drawDisplay();
        }else if ( VISION_VIEW_STATE::ADD_MASK == _enState ) {
            cv::Mat matMaskZoomResult;
            _zoomMat(_matMask, matMaskZoomResult, _fZoomFactor, false);
            _cutImageForDisplay(matMaskZoomResult, _matMaskForDisplay);
            if ( MASK_SHAPE_RECT == _enMaskShape ) {
                Rect rectMask;
                rectMask.x = _ptLeftClickStartPos.x;
                rectMask.y = _ptLeftClickStartPos.y;
                rectMask.width = _ptLeftClickEndPos.x - _ptLeftClickStartPos.x;
                rectMask.height = _ptLeftClickEndPos.y - _ptLeftClickStartPos.y;

                if ( MASK_EDIT_STATE::ADD == _enMaskEditState )
                    cv::rectangle ( _matMaskForDisplay, rectMask, cv::Scalar(255), CV_FILLED, CV_AA );
                else
                    cv::rectangle ( _matMaskForDisplay, rectMask, cv::Scalar(0), CV_FILLED, CV_AA );
            }else if ( MASK_SHAPE_CIRCLE == _enMaskShape )  {
                float fOffsetX = _ptLeftClickEndPos.x - _ptLeftClickStartPos.x;
                float fOffsetY = _ptLeftClickEndPos.y - _ptLeftClickStartPos.y;
                float fRadius = sqrt ( fOffsetX * fOffsetX + fOffsetY * fOffsetY );
                if ( MASK_EDIT_STATE::ADD == _enMaskEditState )
                    cv::circle( _matMaskForDisplay, _ptLeftClickStartPos, fRadius, cv::Scalar(255), CV_FILLED);
                else
                    cv::circle( _matMaskForDisplay, _ptLeftClickStartPos, fRadius, cv::Scalar(0), CV_FILLED);
            }else if ( MASK_SHAPE_POLYLINE == _enMaskShape )  {

            }
            _drawDisplay(true);
        }else if ( VISION_VIEW_STATE::TEST_VISION_LIBRARY == _enState ) {
            if ( TEST_VISION_STATE::IDLE == _enTestVisionState )    {
                _rectSelectedWindow.x = _ptLeftClickStartPos.x;
                _rectSelectedWindow.y = _ptLeftClickStartPos.y;
                _rectSelectedWindow.width = _ptLeftClickEndPos.x - _ptLeftClickStartPos.x;
                _rectSelectedWindow.height = _ptLeftClickEndPos.y - _ptLeftClickStartPos.y;
                _rectSelectedWindow.x -= _szDisplayCenterOffset.width;
                _rectSelectedWindow.y -= _szDisplayCenterOffset.height;

                _drawDisplay ();
            }
            else if ( TEST_VISION_STATE::SET_CIRCLE_INNER_RADIUS == _enTestVisionState ) {
                cv::Point ptReal( point.x() - _szDisplayCenterOffset.width, point.y() - _szDisplayCenterOffset.height );
                _fInnerRangeRadius = distanceOf2Point ( ptReal, _ptCircleCtr );
            }else if ( TEST_VISION_STATE::SET_CIRCLE_OUTER_RADIUS == _enTestVisionState )   {
                cv::Point ptReal( point.x() - _szDisplayCenterOffset.width, point.y() - _szDisplayCenterOffset.height );
                float fRadius = distanceOf2Point ( ptReal, _ptCircleCtr);
                if ( fRadius > _fInnerRangeRadius )
                    _fOuterRangeRadius = fRadius;
            }else if ( TEST_VISION_STATE::SET_MULTIPLE_WINDOW == _enTestVisionState )  {
                assert ( _nCurrentSrchWindowIndex >= 0 );
                cv::Rect rect ( _ptLeftClickStartPos.x, _ptLeftClickStartPos.y, _ptLeftClickEndPos.x - _ptLeftClickStartPos.x, _ptLeftClickEndPos.y - _ptLeftClickStartPos.y);
                rect.x -= _szDisplayCenterOffset.width;
                rect.y -= _szDisplayCenterOffset.height;
                if ( ToInt(_vecRectSrchWindow.size()) <= _nCurrentSrchWindowIndex )
                    _vecRectSrchWindow.push_back ( rect );
                else
                    _vecRectSrchWindow[_nCurrentSrchWindowIndex] = rect;
            }
            else if ( TEST_VISION_STATE::SET_LINE == _enTestVisionState )   {
                _lineOfIntensityCheck.pt1 = _ptLeftClickStartPos;
                _lineOfIntensityCheck.pt2 = _ptLeftClickEndPos;
                _lineOfIntensityCheck.pt1.x -= _szDisplayCenterOffset.width;
                _lineOfIntensityCheck.pt1.y -= _szDisplayCenterOffset.height;
                _lineOfIntensityCheck.pt2.x -= _szDisplayCenterOffset.width;
                _lineOfIntensityCheck.pt2.y -= _szDisplayCenterOffset.height;
            }
            _drawDisplay();
        }else if ( VISION_VIEW_STATE::MOVE == _enState )    {
            _moveImage();
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
    if ( _rectSelectedWindow.width > 0 && _rectSelectedWindow.height > 0 )  {
        cv::Rect rectLocal ( _rectSelectedWindow.x + _szDisplayCenterOffset.width, _rectSelectedWindow.y + _szDisplayCenterOffset.height, _rectSelectedWindow.width, _rectSelectedWindow.height );
        cv::rectangle ( mat, rectLocal, _colorLrnWindow, 1 );
    }

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
        TEST_VISION_STATE::SET_CIRCLE_OUTER_RADIUS == _enTestVisionState )
    {
        if ( abs ( _ptCircleCtr.x ) > 0 && abs ( _ptCircleCtr.y ) > 0 ) {
            cv::Point ptDrawCtr = _ptCircleCtr;
            ptDrawCtr.x += _szDisplayCenterOffset.width;
            ptDrawCtr.y += _szDisplayCenterOffset.height;
            cv::circle ( mat, ptDrawCtr, 2, cv::Scalar ( 0, 255, 0 ), 2 );
            if( _fInnerRangeRadius > 2 )
                cv::circle ( mat, ptDrawCtr, _fInnerRangeRadius, cv::Scalar ( 0, 255, 0 ), 1 );
            if( _fOuterRangeRadius > _fInnerRangeRadius )
                cv::circle ( mat, ptDrawCtr, _fOuterRangeRadius, cv::Scalar ( 0, 255, 0 ), 1 );
        }
    }else if ( TEST_VISION_STATE::SET_MULTIPLE_WINDOW == _enTestVisionState )
    {
        for ( const auto &rect : _vecRectSrchWindow ) {
            cv::Rect rectLocal ( rect.x + _szDisplayCenterOffset.width, rect.y + _szDisplayCenterOffset.height, rect.width, rect.height );
            cv::rectangle ( mat, rectLocal, _colorLrnWindow, 1 );
        }
    }else if ( TEST_VISION_STATE::SET_LINE == _enTestVisionState )  {
        PR_Line lineDraw = _lineOfIntensityCheck;
        lineDraw.pt1.x += _szDisplayCenterOffset.width;
        lineDraw.pt1.y += _szDisplayCenterOffset.height;
        lineDraw.pt2.x += _szDisplayCenterOffset.width;
        lineDraw.pt2.y += _szDisplayCenterOffset.height;
        cv::line ( mat, lineDraw.pt1, lineDraw.pt2, _colorLrnWindow, 1 );
    }
}

void VisionView::_drawDisplay(bool bAddingMask) {
    cv::Mat mat = _matArray[ToInt32(_enDisplaySource)];
    if (mat.empty())
        return;

    cv::Mat matZoomResult, matMaskZoomResult;
    auto displayWidth  = this->size().width();
    auto displayHeight = this->size().height();

    _zoomMat(mat, matZoomResult, _fZoomFactor, false);

    _matDisplay = cv::Mat::ones(displayHeight, displayWidth, mat.type()) * 255;
    _matDisplay.setTo(cv::Scalar(255, 255, 255));

    _cutImageForDisplay(matZoomResult, _matDisplay);

    if (!bAddingMask) {
        cv::Mat matMaskZoomResult;
        _zoomMat(_matMask, matMaskZoomResult, _fZoomFactor, false);
        _cutImageForDisplay(matMaskZoomResult, _matMaskForDisplay);
    }

    _drawTestVisionLibrary(_matDisplay);

    cvtColor(_matDisplay, _matDisplay, CV_BGR2RGB);
    QImage image = QImage((uchar*)_matDisplay.data, _matDisplay.cols, _matDisplay.rows, ToInt(_matDisplay.step), QImage::Format_RGB888);

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
    unsetCursor();
}

void VisionView::addMask()
{
    _enState = VISION_VIEW_STATE::ADD_MASK;
    unsetCursor();    

    if ( ! _pDialogEditMask)    {
        _pDialogEditMask = std::make_unique<DialogEditMask>(this);
        _pDialogEditMask->setVisionView(this);
    }

    _pDialogEditMask->show();
    _pDialogEditMask->raise();
}

void VisionView::clearSelectedWindow()
{
    _enState = VISION_VIEW_STATE::TEST_VISION_LIBRARY;
    _enTestVisionState = TEST_VISION_STATE::IDLE;
    unsetCursor();

    _rectSelectedWindow.width  = -1;
    _rectSelectedWindow.height = -1;
    _vecRectSrchWindow.clear();
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

    point.x = (point.x - displayWidth  / 2) * fZoomFactor + displayWidth  / 2;
    point.y = (point.y - displayHeight / 2) * fZoomFactor + displayHeight / 2;
}

void VisionView::_zoomRect(cv::Rect &rect, float fZoomFactor)
{
    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();

    rect.x = (rect.x - displayWidth  / 2) * fZoomFactor + displayWidth  / 2;
    rect.y = (rect.y - displayHeight / 2) * fZoomFactor + displayHeight / 2;
    rect.width *= fZoomFactor;
    rect.height *= fZoomFactor;
}

/*static*/ void VisionView::_zoomMat(const cv::Mat &mat, cv::Mat &matOutput, float fZoomFactor, bool bKeepSize )
{
    if ( mat.empty() || fabs ( fZoomFactor - 1 ) < 0.001 )  {
        matOutput = mat;
        return;
    }

    cv::Mat matZoomResult;
    cv::resize ( mat, matZoomResult, cv::Size(), fZoomFactor, fZoomFactor );
    if ( ! bKeepSize )  {
        matOutput = matZoomResult;
        return;
    }

    cv::Mat matResultImg = cv::Mat::zeros ( mat.size(), mat.type() );
    
    if ( matResultImg.rows >= matZoomResult.rows && matResultImg.cols >= matZoomResult.cols ) {
        cv::Rect rectROIDst( ( matResultImg.cols - matZoomResult.cols ) / 2, ( matResultImg.rows - matZoomResult.rows ) / 2, matZoomResult.cols, matZoomResult.rows );
        cv::Mat matDst ( matResultImg, rectROIDst);
        matZoomResult.copyTo ( matDst );
    }else if  ( matResultImg.rows <= matZoomResult.rows && matResultImg.cols <= matZoomResult.cols )  {
        cv::Rect rectROISrc ( ( matZoomResult.cols - matResultImg.cols ) / 2, ( matZoomResult.rows - matResultImg.rows ) / 2, matResultImg.cols, matResultImg.rows );
        cv::Mat matSrc( matZoomResult, rectROISrc );
        matSrc.copyTo ( matResultImg );
    }
    matOutput = matResultImg;
}

void VisionView::zoomIn()
{
    if ( _fZoomFactor < _constMaxZoomFactor )   {
        _fZoomFactor *= ZOOM_IN_STEP;

        _zoomPoint(_ptCircleCtr, ZOOM_IN_STEP);
        _fInnerRangeRadius *= ZOOM_IN_STEP;
        _fOuterRangeRadius *= ZOOM_IN_STEP;

        for (auto &rect : _vecRectSrchWindow)
            _zoomRect ( rect, ZOOM_IN_STEP );

        _zoomRect ( _rectSelectedWindow, ZOOM_IN_STEP);

        _zoomMat ( _matMaskForDisplay, _matMaskForDisplay, ZOOM_IN_STEP, true );

        _szDisplayCenterOffset.width *= ZOOM_IN_STEP;
        _szDisplayCenterOffset.height *= ZOOM_IN_STEP;

        _calcMoveRange();
        _drawDisplay();
    }
    _enState = VISION_VIEW_STATE::TEST_VISION_LIBRARY;
    unsetCursor();
}

void VisionView::zoomOut()
{
    if ( _fZoomFactor > _constMinZoomFactor )   {
        _fZoomFactor *= ZOOM_OUT_STEP;

        _zoomPoint(_ptCircleCtr, 0.5f);
        _fInnerRangeRadius *= ZOOM_OUT_STEP;
        _fOuterRangeRadius *= ZOOM_OUT_STEP;

        for (auto &rect : _vecRectSrchWindow)
            _zoomRect(rect, ZOOM_OUT_STEP);

        _zoomRect(_rectSelectedWindow, ZOOM_OUT_STEP);

        _zoomMat(_matMaskForDisplay, _matMaskForDisplay, ZOOM_OUT_STEP, true);

        _szDisplayCenterOffset.width *= ZOOM_OUT_STEP;
        _szDisplayCenterOffset.height *= ZOOM_OUT_STEP;

        _calcMoveRange();
        _drawDisplay();
    }
    _enState = VISION_VIEW_STATE::TEST_VISION_LIBRARY;
    unsetCursor();
}

void VisionView::_calcMoveRange()
{
    if ( _matArray[ToInt32(DISPLAY_SOURCE::ORIGINAL)].empty() ) {
        _szMoveRange = cv::Size(0, 0);
        return;
    }

    int rows = _matArray[ToInt32(DISPLAY_SOURCE::ORIGINAL)].rows * _fZoomFactor;
    int cols = _matArray[ToInt32(DISPLAY_SOURCE::ORIGINAL)].cols * _fZoomFactor;
    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();    

    _szMoveRange.width = ( cols - displayWidth ) / 2;
    if ( _szMoveRange.width < 0 ) _szMoveRange.width = 0;

    _szMoveRange.height = ( rows - displayHeight ) / 2;
    if ( _szMoveRange.height < 0 ) _szMoveRange.height = 0;

    if ( _szDisplayCenterOffset.width > _szMoveRange.width )
        _szDisplayCenterOffset.width = _szMoveRange.width;
    else if ( _szDisplayCenterOffset.width < -_szMoveRange.width )
        _szDisplayCenterOffset.width = - _szMoveRange.width;

    if ( _szDisplayCenterOffset.height > _szMoveRange.height )
        _szDisplayCenterOffset.height = _szMoveRange.height;
    else if ( _szDisplayCenterOffset.height < -_szMoveRange.height )
        _szDisplayCenterOffset.height = - _szMoveRange.height;
}

void VisionView::_moveImage()
{
    _szDisplayCenterOffset.width = _szConfirmedCenterOffset.width + _ptLeftClickEndPos.x - _ptLeftClickStartPos.x;
    if ( _szDisplayCenterOffset.width > _szMoveRange.width )
        _szDisplayCenterOffset.width = _szMoveRange.width;
    else if ( _szDisplayCenterOffset.width < -_szMoveRange.width )
        _szDisplayCenterOffset.width = - _szMoveRange.width;

    _szDisplayCenterOffset.height = _szConfirmedCenterOffset.height + _ptLeftClickEndPos.y - _ptLeftClickStartPos.y;
    if ( _szDisplayCenterOffset.height > _szMoveRange.height )
        _szDisplayCenterOffset.height = _szMoveRange.height;
    else if ( _szDisplayCenterOffset.height < -_szMoveRange.height )
        _szDisplayCenterOffset.height = - _szMoveRange.height;

    _drawDisplay();
}

void VisionView::_applyMask()
{
    _matMask = cv::Mat::zeros(_matArray[0].size(), CV_8UC1) * PR_MAX_GRAY_LEVEL;
    cv::Mat  matLocalMask;
    if ( fabs ( _fZoomFactor - 1.f) < 0.001 )
        matLocalMask = _matMaskForDisplay.clone();
    else
        _zoomMat ( _matMaskForDisplay, matLocalMask, 1 / _fZoomFactor, false );

    if (_matMask.rows >= matLocalMask.rows && _matMask.cols >= matLocalMask.cols) {
        cv::Rect rectROIDst((_matMask.cols - matLocalMask.cols) / 2 - _szConfirmedCenterOffset.width, (_matMask.rows - matLocalMask.rows) / 2 - _szConfirmedCenterOffset.height, matLocalMask.cols, matLocalMask.rows);
        cv::Mat matDst(_matMask, rectROIDst);
        matLocalMask.copyTo(matDst);
    }
    else if (_matMask.rows < matLocalMask.rows && _matMask.cols < matLocalMask.cols)  {
        cv::Rect rectROISrc((matLocalMask.cols - _matMask.cols) / 2, (matLocalMask.rows - _matMask.rows) / 2, _matMask.cols, _matMask.rows);
        cv::Mat matSrc(matLocalMask, rectROISrc);
        matSrc.copyTo(_matMask);
    }
    else if (_matMask.rows > matLocalMask.rows && _matMask.cols < matLocalMask.cols)   {
        cv::Rect rectROISrc((matLocalMask.cols - _matMask.cols) / 2, 0, _matMask.cols, matLocalMask.rows);
        cv::Mat matSrc(matLocalMask, rectROISrc);
        cv::Rect rectROIDst(0, (_matMask.rows - matLocalMask.rows) / 2 - _szConfirmedCenterOffset.height, _matMask.cols, matLocalMask.rows);
        cv::Mat matDst(_matMask, rectROIDst);
        matSrc.copyTo(matDst);
    }
    else if (_matMask.rows < matLocalMask.rows && _matMask.cols > matLocalMask.cols)   {
        cv::Rect rectROISrc(0, (matLocalMask.rows - _matMask.rows) / 2, matLocalMask.cols, _matMask.rows);
        cv::Mat matSrc(matLocalMask, rectROISrc);
        cv::Rect rectROIDst((_matMask.cols - matLocalMask.cols) / 2 - _szConfirmedCenterOffset.width, 0, matLocalMask.cols, _matMask.rows);
        cv::Mat matDst(_matMask, rectROIDst);
        matSrc.copyTo(matDst);
    }
}

void VisionView::_cutImageForDisplay(const cv::Mat &matInputImg, cv::Mat &matOutput)
{
    if ( matInputImg.empty() )
        return;

    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();

    if (matInputImg.cols >= displayWidth && matInputImg.rows >= displayHeight)  {
        cv::Rect rectROISrc( (matInputImg.cols - displayWidth) / 2 - _szDisplayCenterOffset.width, (matInputImg.rows - displayHeight) / 2 - _szDisplayCenterOffset.height, displayWidth, displayHeight);
        cv::Mat matSrc( matInputImg, rectROISrc);
        matSrc.copyTo ( matOutput );
    }
    else if (matInputImg.cols >= displayWidth && matInputImg.rows <= displayHeight)  {
        cv::Rect rectROISrc( (matInputImg.cols - displayWidth) / 2 - _szDisplayCenterOffset.width, 0, displayWidth, matInputImg.rows);
        cv::Mat matSrc(matInputImg, rectROISrc);
        cv::Rect rectROIDst(0, ( displayHeight - matInputImg.rows ) / 2, displayWidth, matInputImg.rows );
        cv::Mat matDst(matOutput, rectROIDst);
        matSrc.copyTo(matDst);
    }else if (matInputImg.cols <= displayWidth && matInputImg.rows >= displayHeight)  {
        cv::Rect rectROISrc(0, (matInputImg.rows - displayHeight) / 2 - _szDisplayCenterOffset.height, matInputImg.cols, displayHeight);
        cv::Mat matSrc(matInputImg, rectROISrc);
        cv::Rect rectROIDst(( displayWidth - matInputImg.cols ) / 2, 0, matInputImg.cols, displayHeight );
        cv::Mat matDst(matOutput, rectROIDst);
        matSrc.copyTo(matDst);
    }else if (matInputImg.cols <= displayWidth && matInputImg.rows <= displayHeight) {
        cv::Rect rectROIDst((displayWidth - matInputImg.cols) / 2, (displayHeight - matInputImg.rows) / 2, matInputImg.cols, matInputImg.rows);
        cv::Mat matDst(matOutput, rectROIDst);
        matInputImg.copyTo(matDst);
    }
    else
    {
        QMessageBox::critical(nullptr, "Display", "This should not happen, please contact software engineer!", "Quit");
    }
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

void VisionView::startTimer()
{
    _pTimer->start(20);
}

void VisionView::setMat( DISPLAY_SOURCE enSource, const cv::Mat &mat)
{
    _enDisplaySource = enSource;
    _matArray[ToInt32(_enDisplaySource)] = mat;

    if ( DISPLAY_SOURCE::ORIGINAL == enSource ) {
        _calcMoveRange();
        _szDisplayCenterOffset = cv::Size(0, 0);
        _szConfirmedCenterOffset = cv::Size(0, 0);
        //_matMask = cv::Mat::zeros(_matArray[0].size(), CV_8UC1);
        _matMask.release();
        _matMaskForDisplay = cv::Mat::zeros( this->size().height(), this->size().width() , CV_8UC1 );
    }
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
    return cv::Scalar(255) - _matMask;
}

void VisionView::applyIntermediateResult()
{
    if (!_matArray[ToInt32(DISPLAY_SOURCE::INTERMEDIATE)].empty())
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
        _fOuterRangeRadius = 0.f;
        _vecRectSrchWindow.clear();

        _drawDisplay();
        unsetCursor();
    }
    if ( TEST_VISION_STATE::PICK_COLOR == _enTestVisionState )  {
        setCursor ( Qt::PointingHandCursor );
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

void VisionView::getFitCircleRange(cv::Point &ptCtr, float &fInnerRadius, float &fOuterRadius) const
{
    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();
    
    cv::Mat mat = _matArray[ToInt32(_enDisplaySource)];
    ptCtr.x = ( _ptCircleCtr.x - ( displayWidth  - mat.cols * _fZoomFactor ) / 2 ) / _fZoomFactor;
    ptCtr.y = ( _ptCircleCtr.y - ( displayHeight - mat.rows * _fZoomFactor ) / 2 ) / _fZoomFactor;
    fInnerRadius = _fInnerRangeRadius / _fZoomFactor;
    fOuterRadius = _fOuterRangeRadius / _fZoomFactor;
}

VectorOfRect VisionView::getVecSrchWindow() const
{
    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();
    VectorOfRect vecResult;
    cv::Mat mat = _matArray[ToInt32(_enDisplaySource)];
    for ( auto rect : _vecRectSrchWindow ) {
        rect.x = (rect.x + (mat.cols * _fZoomFactor - displayWidth  ) / 2) / _fZoomFactor;
        rect.y = (rect.y + (mat.rows * _fZoomFactor - displayHeight ) / 2) / _fZoomFactor;
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

    rect.x = (rect.x + (mat.cols * _fZoomFactor - displayWidth  ) / 2) / _fZoomFactor;
    rect.y = (rect.y + (mat.rows * _fZoomFactor - displayHeight ) / 2) / _fZoomFactor;
    rect.width  /= _fZoomFactor;
    rect.height /= _fZoomFactor;
    return rect;
}

cv::Point VisionView::getClickedPoint() const
{
    auto displayWidth = this->size().width();
    auto displayHeight = this->size().height();
    cv::Mat mat = _matArray[ToInt32(_enDisplaySource)];
    auto pt = _ptLeftClickStartPos;
    pt.x -= _szDisplayCenterOffset.width;
    pt.y -= _szDisplayCenterOffset.height;
    pt.x = (pt.x + (mat.cols * _fZoomFactor - displayWidth  ) / 2) / _fZoomFactor;
    pt.y = (pt.y + (mat.rows * _fZoomFactor - displayHeight ) / 2) / _fZoomFactor;
    return pt;
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