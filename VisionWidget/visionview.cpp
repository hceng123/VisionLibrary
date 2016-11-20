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
    _nState = LEARNING;

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
    return OK;
}

void VisionView::mousePressEvent(QMouseEvent *event)
{
    if ( Qt::LeftButton == event->button() )
    {
        _bLeftButtonDown = true;
        QPoint point = event->pos();
        _ptLeftClickStartPos.x = point.x();
        _ptLeftClickStartPos.y = point.y();

        if ( ADD_MASK == _nState && MASK_SHAPE_POLYLINE == _enMaskShape )    {
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
        if ( ADD_MASK == _nState && MASK_SHAPE_POLYLINE == _enMaskShape )    {
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
        }
    }else if ( event->button() == Qt::MidButton )   {
        QMessageBox msgBox;
        msgBox.setText("Middle button single click");
        msgBox.exec();
    }
}

void VisionView::mouseMoveEvent(QMouseEvent *event)
{
    if(event->buttons() & Qt::LeftButton)
    {
        //do stuff
        QPoint point = event->pos();
        _ptLeftClickEndPos.x = point.x();
        _ptLeftClickEndPos.y = point.y();

        if ( LEARNING == _nState )  {
            _rectLrnWindow.x = _ptLeftClickStartPos.x;
            _rectLrnWindow.y = _ptLeftClickStartPos.y;
            _rectLrnWindow.width = _ptLeftClickEndPos.x - _ptLeftClickStartPos.x;
            _rectLrnWindow.height = _ptLeftClickEndPos.y - _ptLeftClickStartPos.y;
        }else if ( ADD_MASK == _nState )    {
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

void VisionView::setMachineState(int nMachineState)
{
    _nState = nMachineState;
}

void VisionView::_drawLearnWindow(cv::Mat &mat)
{
    cv::rectangle ( mat, _rectLrnWindow, _colorLrnWindow, 2 );

    if ( ADD_MASK == _nState && MASK_SHAPE_POLYLINE == _enMaskShape  )    {
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

    _drawLearnWindow ( _matDisplay );
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
    _nState = ADD_MASK;
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