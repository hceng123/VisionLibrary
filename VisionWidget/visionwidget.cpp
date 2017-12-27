#include "visionwidget.h"
#include "FitParallelLineProcedure.h"
#include "FitRectProcedure.h"
#include "OcrProcedure.h"
#include "SrchFiducialProcedure.h"
#include "constants.h"
#include "VisionAPI.h"
#include <QFileDialog>
#include <QMessageBox>
#include "messageboxdialog.h"

using namespace AOI::Vision;

template<class T>
std::string formatMat(const cv::Mat &mat)
{
    std::string strResult;
    char chArray[100];
	for (short row = 0; row < mat.rows; ++row)
	{
		for (short col = 0; col < mat.cols; ++col)
		{
            _snprintf ( chArray, sizeof(chArray), "%.2f, ", mat.at<T>(row, col) );
			strResult += chArray;
		}
	}
    return strResult;
}

inline cv::RotatedRect rectToRotatedRect(const cv::Rect &rect) {
    cv::RotatedRect rotatedRect;
    rotatedRect.angle = 0;
    rotatedRect.center.x = rect.x + rect.width / 2;
    rotatedRect.center.y = rect.y + rect.height / 2;
    rotatedRect.size = rect.size();
    return rotatedRect;
}

VisionWidget::VisionWidget(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    setWindowIcon(QIcon(":/VisionWidget/Image/VisionWidget.png"));

    ui.visionViewToolBox->SetVisionView(ui.visionView);
    _ptrIntValidator    = std::make_unique<QIntValidator>(1,100);
    ui.lineEditFitCircleErrTol->setValidator(_ptrIntValidator.get() );
    ui.visionView->setState(VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY);

    _ptrGrayScaleWidget = std::make_unique<GrayScaleWidget>(this);
    _ptrGrayScaleWidget->setVisionView ( ui.visionView );
    ui.verticalLayout->addWidget(_ptrGrayScaleWidget.get());

    _ptrFilterWidget = std::make_unique<FilterWidget>(this);
    _ptrFilterWidget->setVisionView ( ui.visionView );
    ui.verticalLayout->addWidget(_ptrFilterWidget.get());

    _ptrThresholdWidget = std::make_unique<ThresholdWidget>(this);
    _ptrThresholdWidget->setVisionView ( ui.visionView );
    ui.verticalLayout->addWidget(_ptrThresholdWidget.get());

    _ptrColorWidget = std::make_unique<ColorWidget>(this);
    _ptrColorWidget->setVisionView ( ui.visionView );
    ui.verticalLayout->addWidget(_ptrColorWidget.get());

    _ptrFillHoleWidget = std::make_unique<FillHoleWidget>(this);
    _ptrFillHoleWidget->setVisionView( ui.visionView );
    ui.verticalLayout->addWidget(_ptrFillHoleWidget.get());

    _ptrEdgeDetectWidget = std::make_unique<EdgeDetectWidget>(this);
    _ptrEdgeDetectWidget->setVisionView ( ui.visionView );
    ui.verticalLayout->addWidget(_ptrEdgeDetectWidget.get());

    _ptrCcWidget = std::make_unique<CCWidget>(this);
    _ptrCcWidget->setVisionView ( ui.visionView );
    ui.verticalLayout->addWidget(_ptrCcWidget.get());

    PR_Init();
}

VisionWidget::~VisionWidget()
{
}

bool VisionWidget::checkDisplayImage()
{
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return false;
    }

    if ( ui.visionView->isDisplayResultImage()) {
        QMessageBox::information(this, "Vision Widget", "Please swap to the original image first!", "Quit");
        return false;
    }
    return true;
}

void VisionWidget::on_selectImageBtn_clicked()
{
    QFileDialog dialog(this);
    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
    dialog.setViewMode(QFileDialog::Detail);
    QStringList fileNames;
    if (dialog.exec())  {
        fileNames = dialog.selectedFiles();
    }else
        return;

    _sourceImagePath = fileNames[0].toStdString();
    ui.imagePathEdit->setText(fileNames[0]);

    on_checkBoxByerFormat_clicked ( ui.checkBoxByerFormat->isChecked() );
}

void VisionWidget::on_checkBoxByerFormat_clicked(bool checked)
{
    _sourceImagePath = ui.imagePathEdit->text().toStdString();
    if ( _sourceImagePath.empty() )
        return;

    cv::Mat mat;
    if ( checked )    {
        mat = cv::imread( _sourceImagePath, IMREAD_GRAYSCALE );
        cv::Mat matColor;
        cv::cvtColor (mat, matColor, CV_BayerGR2BGR );
        _matOriginal = matColor;
    }
    else
        _matOriginal = cv::imread( _sourceImagePath );

    ui.visionView->setMat( VisionView::DISPLAY_SOURCE::ORIGINAL, _matOriginal );
    ui.visionView->clearMat ( VisionView::DISPLAY_SOURCE::INTERMEDIATE );
    ui.visionView->clearMat ( VisionView::DISPLAY_SOURCE::RESULT );
}

void VisionWidget::on_fitCircleBtn_clicked()
{
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_FIT_CIRCLE_CMD stCmd;
	stCmd.matInputImg = ui.visionView->getMat();
    stCmd.matMask = ui.visionView->getMask();
    stCmd.rectROI = ui.visionView->getSelectedWindow();    
	stCmd.enRmNoiseMethod = static_cast<PR_RM_FIT_NOISE_METHOD>(ui.comboBoxFitCircleDirection->currentIndex());
	stCmd.fErrTol = ui.lineEditFitCircleErrTol->text().toFloat();
    stCmd.bPreprocessed = true;
	stCmd.bAutoThreshold = false;
	//stCmd.nThreshold = _nThreshold;
    //stCmd.enAttribute = static_cast<PR_OBJECT_ATTRIBUTE>(_nAttribute);
	stCmd.enMethod = static_cast<PR_FIT_METHOD>(ui.comboBoxFitCircleAlgorithm->currentIndex());
	stCmd.nMaxRansacTime = 20;

	PR_FIT_CIRCLE_RPY stRpy;
	VisionStatus enStatus = PR_FitCircle(&stCmd, &stRpy);
    if ( VisionStatus::OK == enStatus )
    {
        ui.visionView->setMat(VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(enStatus, &stErrStrRpy);
        QMessageBox::critical(nullptr, "Fit Line", stErrStrRpy.achErrorStr, "Quit");
    }
}

void VisionWidget::on_houghCircleBtn_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    cv::Mat matInputImg = ui.visionView->getMat();
    cv::Rect rectROI = ui.visionView->getSelectedWindow();
    cv::Mat matGray;
    cv::cvtColor ( matInputImg, matGray, CV_BGR2GRAY );
    cv::Mat matROI ( matGray, rectROI );

    std::vector<cv::Vec3f> circles;
    double dMinDist = ui.lineEditHoughCircleMinDist->text().toDouble();
    double dCannyThreshold = ui.lineEditHoughCircleCannyThreshold->text().toDouble();
    double dAccumulatorThreshold = ui.lineEditHoughCircleAccumlatorThreshold->text().toDouble();
    int nMinRadius = ui.lineEditHoughCircleMinRadius->text().toInt();
    int nMaxRadius = ui.lineEditHoughCircleMaxRadius->text().toInt();
    cv::HoughCircles ( matROI, circles, CV_HOUGH_GRADIENT, 1, dMinDist, dCannyThreshold,
        dAccumulatorThreshold, nMinRadius, nMaxRadius );
    cv::Mat matResultImage = matInputImg.clone();
    for( size_t i = 0; i < circles.size(); ++ i )
    {
         cv::Point center(cvRound(circles[i][0]) + rectROI.x, cvRound(circles[i][1]) + rectROI.y);
         int radius = cvRound(circles[i][2]);

         //Draw the center of circle.
         cv::circle( matResultImage, center, 3, Scalar(0,255,0), -1, 8, 0 );
         // draw the circle outline
         cv::circle( matResultImage, center, radius, Scalar(0,0,255), 2, 8, 0 );
    }
    ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, matResultImage );
}

void VisionWidget::on_findCircleBtn_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    QRect rect = qApp->activeWindow()->geometry();
	std::unique_ptr<MessageBoxDialog> pMessageBox = std::make_unique<MessageBoxDialog>();
    pMessageBox->setGeometry(rect.x() + 800, rect.y() + 300, pMessageBox->size().width(), pMessageBox->size().height());
    pMessageBox->setWindowTitle("Find Circle");
	pMessageBox->SetMessageText1("Please input the expected circle center");
    pMessageBox->SetMessageText2("Press and drag the left mouse buttont to input");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_CIRCLE_CTR);
	int iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
        return;
    }

    pMessageBox->SetMessageText1("Please input the min search radius");
    pMessageBox->SetMessageText2("Press and drag the left mouse buttont to input");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_CIRCLE_INNER_RADIUS);

	iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
        return;
    }

    pMessageBox->SetMessageText1("Please input the max search radius");
    pMessageBox->SetMessageText2("Press and drag the left mouse buttont to input");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_CIRCLE_OUTER_RADIUS);

	iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
        return;
    }

    cv::Point ptCtr;
    float fInnerRadius, fOuterRadius;
    ui.visionView->getFitCircleRange(ptCtr, fInnerRadius, fOuterRadius);

    PR_FIND_CIRCLE_CMD stCmd;
    PR_FIND_CIRCLE_RPY stRpy;

    stCmd.matInputImg = ui.visionView->getMat();
    stCmd.matMask = ui.visionView->getMask();
    stCmd.enInnerAttribute = static_cast<PR_OBJECT_ATTRIBUTE> ( ui.comboBoxDetectCircleAttribute->currentIndex() );
    stCmd.bFindCirclePair = static_cast<bool> ( ui.comboBoxDetectCircleFindPair->currentIndex() );
    stCmd.ptExpectedCircleCtr = ptCtr;
    stCmd.fMinSrchRadius = fInnerRadius;
    stCmd.fMaxSrchRadius = fOuterRadius;
    stCmd.fStartSrchAngle = ui.lineEditDetectCircleStartSrchAngle->text().toFloat();
    stCmd.fEndSrchAngle = ui.lineEditDetectCircleEndSrchAngle->text().toFloat();
    //stCmd.fErrTol = ui.lineEditDetectCircleErrTol->text().toFloat();
    stCmd.nCaliperCount = ui.lineEditDetectCircleCaliperCount->text().toInt();
    stCmd.fCaliperWidth = ui.lineEditDetectCircleCaliperWidth->text().toFloat();
    stCmd.fRmStrayPointRatio = ui.lineEditDetectCircleRmStrayRatio->text().toFloat();
    
    PR_FindCircle ( &stCmd, &stRpy );
    if ( stRpy.enStatus == VisionStatus::OK ) {
        ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrStrRpy);
        QMessageBox::critical(nullptr, "Find Circle", stErrStrRpy.achErrorStr, "Quit");
    }
}

void VisionWidget::on_calcRoundnessBtn_clicked()
{
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_INSP_CIRCLE_CMD stCmd;
	stCmd.matInputImg = ui.visionView->getMat();
    stCmd.matMask = ui.visionView->getMask();
    stCmd.rectROI = ui.visionView->getSelectedWindow();

	PR_INSP_CIRCLE_RPY stRpy;
	VisionStatus enStatus = PR_InspCircle(&stCmd, &stRpy);
	if (VisionStatus::OK == enStatus)	{
		ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
        ui.lineEditRoundnessResult->setText( std::to_string(stRpy.fRoundness).c_str() );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(enStatus, &stErrStrRpy);
        QMessageBox::critical(nullptr, "Fit Line", stErrStrRpy.achErrorStr, "Quit");
    }
}

void VisionWidget::on_fitLineBtn_clicked()
{
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_FIT_LINE_CMD stCmd;
	stCmd.matInputImg = ui.visionView->getMat();
    stCmd.matMask = ui.visionView->getMask();
    stCmd.enMethod = static_cast<PR_FIT_METHOD>(ui.comboBoxFitLineAlgorithm->currentIndex());
	stCmd.enRmNoiseMethod = static_cast<PR_RM_FIT_NOISE_METHOD>(ui.comboBoxFitLineDirection->currentIndex());
	stCmd.fErrTol = ui.lineEditFitLineErrTol->text().toFloat();
	stCmd.rectROI = ui.visionView->getSelectedWindow();
    stCmd.bPreprocessed = true;
    stCmd.nNumOfPtToFinishRansac = 100;

	PR_FIT_LINE_RPY stRpy;
	VisionStatus enStatus = PR_FitLine(&stCmd, &stRpy);
    if ( VisionStatus::OK == enStatus )
    {
        ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(enStatus, &stErrStrRpy);
        QMessageBox::critical(nullptr, "Fit Line", stErrStrRpy.achErrorStr, "Quit");
    }
}

void VisionWidget::on_btnFindLine_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_FIND_LINE_CMD stCmd;
    stCmd.matInputImg = ui.visionView->getMat();
    stCmd.matMask = ui.visionView->getMask();
    auto rectROI = ui.visionView->getSelectedWindow();
    stCmd.rectRotatedROI.center = cv::Point ( rectROI.x + rectROI.width / 2, rectROI.y + rectROI.height / 2 );
    stCmd.rectRotatedROI.size = rectROI.size();
    stCmd.enDetectDir = static_cast<PR_CALIPER_DIR> ( ui.comboBoxFindLineDirection->currentIndex() );
    stCmd.enAlgorithm = static_cast<PR_FIND_LINE_ALGORITHM> ( ui.comboBoxFindLineAlgorithm->currentIndex() );
    stCmd.nCaliperCount = ui.lineEditFindLineCaliperCount->text().toInt();
    stCmd.fCaliperWidth = ui.lineEditFindLineCaliperWidth->text().toFloat();
    stCmd.bFindPair = static_cast<bool> ( ui.comboBoxFindLineFindPair->currentIndex() );

    PR_FIND_LINE_RPY stRpy;
	VisionStatus enStatus = PR_FindLine(&stCmd, &stRpy);
	if (VisionStatus::OK == enStatus)	{
		ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(enStatus, &stErrStrRpy);
        QMessageBox::critical(nullptr, "Find Line", stErrStrRpy.achErrorStr, "Quit");
        ui.lineEditObjCenter->clear();
        ui.lineEditObjRotation->clear();
    }
}

void VisionWidget::on_fitParallelLineBtn_clicked()
{
    if ( ! checkDisplayImage() )
        return;

    FitParallelLineProcedure procedure(ui.visionView);
    procedure.setErrTol ( ui.lineEditFitParallelLineErrTol->text().toFloat());
    procedure.setThreshold ( ui.lineEditFitParallelLineThreshold->text().toInt());
    procedure.setAttribute ( ui.comboBoxParallelLineAttribute->currentIndex());
	int nStatus = procedure.run(_sourceImagePath);
    if ( ToInt(VisionStatus::OK) == nStatus )
    {
        ui.visionView->setMat(VisionView::DISPLAY_SOURCE::RESULT, procedure.getResultMat());
    }
}

void VisionWidget::on_fitRectBtn_clicked()
{
    if ( ! checkDisplayImage() )
        return;

    FitRectProcedure procedure(ui.visionView);
    procedure.setErrTol ( ui.lineEditFitRectErrTol->text().toFloat());
    procedure.setThreshold ( ui.lineEditFitRectThreshold->text().toInt());
    procedure.setAttribute ( ui.comboBoxRectAttribute->currentIndex());
	int nStatus = procedure.run(_sourceImagePath);
    if ( ToInt(VisionStatus::OK) == nStatus )
    {
        ui.visionView->setMat(VisionView::DISPLAY_SOURCE::RESULT, procedure.getResultMat());
    }
}

void VisionWidget::on_ocrBtn_clicked()
{
    if ( ! checkDisplayImage() )
        return;

	OcrProcedure procedure(ui.visionView);
    procedure.setTextDirection(ui.comboBoxOcrTextDirection->currentIndex());
	int nStatus = procedure.run(_sourceImagePath);
    if ( ToInt(VisionStatus::OK) == nStatus )
    {
        ui.textEditOcrResult->setText( procedure.getResult().c_str() );
    }
}

void VisionWidget::on_srchFiducialBtn_clicked()
{
    if ( ! checkDisplayImage() )
        return;

    SrchFiducialProcedure procedure(ui.visionView);
    procedure.setFiducialSize ( ui.lineEditFiducialMarkSize->text().toFloat());
    procedure.setFiducialMarge ( ui.lineEditFiducialMarkMargin->text().toFloat());
    procedure.setFicucialType ( ui.comboBoxFiducialMarkShape->currentIndex() );
    int nStatus = procedure.run(_sourceImagePath);
    if ( ToInt(VisionStatus::OK) == nStatus )
    {
        ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, procedure.getResultMat() );
    }
}

void VisionWidget::drawTmplImage(const cv::Mat &matTmpl)
{
    cv::Mat matDisplay;
    cv::Size size( ui.labelTmplView->size().height(), ui.labelTmplView->size().width() );
    cv::resize ( matTmpl, matDisplay, size );
    if ( matDisplay.channels() > 1 )
        cvtColor ( matDisplay, matDisplay, CV_BGR2RGB );
    else
        cvtColor ( matDisplay, matDisplay, CV_GRAY2RGB );
    QImage image = QImage((uchar*) matDisplay.data, matDisplay.cols, matDisplay.rows, ToInt(matDisplay.step), QImage::Format_RGB888);
    ui.labelTmplView->setPixmap(QPixmap::fromImage(image));
}

void VisionWidget::on_btnLrnTemplate_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_LRN_TEMPLATE_CMD stCmd;
    PR_LRN_TEMPLATE_RPY stRpy;
    stCmd.matInputImg = ui.visionView->getMat();
    stCmd.rectROI = ui.visionView->getSelectedWindow();
    stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF;
    if ( ui.cbMatchTmplAlgorithm->currentIndex() == 1 )
        stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_EDGE;
    else if ( ui.cbMatchTmplAlgorithm->currentIndex() == 2 )
        stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_AREA;

    if ( VisionStatus::OK == PR_LrnTmpl ( &stCmd, &stRpy) ) {
        _nTmplRecordId = stRpy.nRecordId;        
        drawTmplImage ( stRpy.matTmpl );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo( stRpy.enStatus, &stErrStrRpy);
        QMessageBox::critical(nullptr, "Match Template", stErrStrRpy.achErrorStr, "Quit");
        ui.lineEditObjCenter->clear();
        ui.lineEditObjRotation->clear();
    }    
}

void VisionWidget::on_matchTmplBtn_clicked()
{
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_MATCH_TEMPLATE_CMD stCmd;
    stCmd.matInputImg = ui.visionView->getMat();
    stCmd.nRecordId = _nTmplRecordId;
    stCmd.rectSrchWindow = ui.visionView->getSelectedWindow();
    stCmd.enMotion = static_cast<PR_OBJECT_MOTION>( ui.cbMatchTmplMotion->currentIndex() );
    stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF;
    if ( ui.cbMatchTmplAlgorithm->currentIndex() == 1 )
        stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_EDGE;
    else if ( ui.cbMatchTmplAlgorithm->currentIndex() == 2 )
        stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_AREA;

    PR_MATCH_TEMPLATE_RPY stRpy;
    VisionStatus enStatus = PR_MatchTmpl( &stCmd, &stRpy );
    if ( VisionStatus::OK == enStatus )
    {
        ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
        char chArrCenter[100];
        _snprintf( chArrCenter, sizeof ( chArrCenter ), "%.3f, %.3f", stRpy.ptObjPos.x, stRpy.ptObjPos.y);
        ui.lineEditObjCenter->setText( chArrCenter );
        std::string strRotation = std::to_string ( stRpy.fRotation );
        ui.lineEditObjRotation->setText(strRotation.c_str());
        std::string strScore = std::to_string ( stRpy.fMatchScore );
        ui.lineEditMatchScore->setText ( strScore.c_str() );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(enStatus, &stErrStrRpy);
        QMessageBox::critical(nullptr, "Match Template", stErrStrRpy.achErrorStr, "Quit");
        ui.lineEditObjCenter->clear();
        ui.lineEditObjRotation->clear();
    }
}

void VisionWidget::on_btnCalcCoverage_clicked()
{
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    cv::Mat mat = ui.visionView->getMat();
    cv::Rect rectROI = ui.visionView->getSelectedWindow();
    cv::Mat matROI(mat, rectROI);
    int nTotalArea = rectROI.width * rectROI.height;    
    cv::Mat matGray;
    if ( matROI.channels() > 1 )
        cv::cvtColor ( matROI, matGray, CV_BGR2GRAY );
    else
        matGray = matROI.clone();
    VectorOfPoint vecPoint;
    int nWhiteArea = cv::countNonZero ( matGray );
    int nBlackArea = nTotalArea - nWhiteArea;
    
    float fWhiteRatio = (float)nWhiteArea / (float)nTotalArea;
    float fBlackRatio = (float)nBlackArea / (float)nTotalArea;

    ui.lineEditWhiteArea->setText ( std::to_string ( nWhiteArea ).c_str() );
    ui.lineEditBlackArea->setText ( std::to_string ( nBlackArea ).c_str() );

    ui.lineEditWhiteRatio->setText ( std::to_string ( fWhiteRatio ).c_str() );
    ui.lineEditBlackRatio->setText ( std::to_string ( fBlackRatio ).c_str() );
}

void VisionWidget::on_btnCalibrateCamera_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_CALIBRATE_CAMERA_CMD stCmd;
    PR_CALIBRATE_CAMERA_RPY stRpy;
    stCmd.matInputImg = ui.visionView->getMat();
    stCmd.szBoardPattern.height = ui.lineEditChessboardRows->text().toInt();
    stCmd.szBoardPattern.width = ui.lineEditChessboardCols->text().toInt();
    stCmd.fPatternDist = ui.lineEditChessboardTmplDist->text().toFloat();
    stCmd.fMinTmplMatchScore = ui.lineEditTmplMatchMinScore->text().toFloat();

    PR_CalibrateCamera ( &stCmd, &stRpy );

    if ( VisionStatus::OK == stRpy.enStatus )
    {
        ui.lineEditResolutionX->setText ( std::to_string ( stRpy.dResolutionX ).c_str() );
        ui.lineEditResolutionY->setText ( std::to_string ( stRpy.dResolutionY ).c_str() );
        ui.lineEditCameraDistCoeff->setText ( formatMat<double> ( stRpy.matDistCoeffs ).c_str() );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrStrRpy);
        QMessageBox::critical(nullptr, "Calibrate Camera", stErrStrRpy.achErrorStr, "Quit");
        ui.lineEditObjCenter->clear();
        ui.lineEditObjRotation->clear();
    }
}

void VisionWidget::on_btnLrnChip_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_LRN_CHIP_CMD stCmd;
    PR_LRN_CHIP_RPY stRpy;
    stCmd.bAutoThreshold = true;
    stCmd.matInputImg = ui.visionView->getMat();
    stCmd.enInspMode = static_cast<PR_INSP_CHIP_MODE> ( ui.comboBoxChipInspMode->currentIndex() );
    stCmd.rectChip = ui.visionView->getSelectedWindow();
    PR_LrnChip ( &stCmd, &stRpy );
    if ( VisionStatus::OK == stRpy.enStatus ) {
        ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
        _nChipRecordId = stRpy.nRecordId;
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrStrRpy);
        QMessageBox::critical(nullptr, "Learn chip", stErrStrRpy.achErrorStr, "Quit");
        ui.lineEditObjCenter->clear();
        ui.lineEditObjRotation->clear();
    }
}

void VisionWidget::on_btnInspChip_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_INSP_CHIP_CMD stCmd;
    PR_INSP_CHIP_RPY stRpy;
    stCmd.matInputImg = ui.visionView->getMat();
    stCmd.enInspMode = static_cast<PR_INSP_CHIP_MODE> ( ui.comboBoxChipInspMode->currentIndex() );
    stCmd.rectSrchWindow = ui.visionView->getSelectedWindow();
    stCmd.nRecordId = _nChipRecordId;
    PR_InspChip ( &stCmd, &stRpy );
    if ( VisionStatus::OK == stRpy.enStatus ) {
        ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrStrRpy);
        QMessageBox::critical(nullptr, "Learn chip", stErrStrRpy.achErrorStr, "Quit");
        ui.lineEditObjCenter->clear();
        ui.lineEditObjRotation->clear();
    }
}

void VisionWidget::on_btnCountEdge_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_FIND_EDGE_CMD stCmd;
    PR_FIND_EDGE_RPY stRpy;
    stCmd.matInputImg = ui.visionView->getMat();
    stCmd.rectROI = ui.visionView->getSelectedWindow();
    stCmd.bAutoThreshold = ui.checkBoxCountEdgeAutoThreshold->isChecked();
    stCmd.nThreshold = ui.lineEditFitCountEdgeThreshold->text().toInt();
    stCmd.enDirection = static_cast<PR_EDGE_DIRECTION> ( ui.cbCountEdgeDirection->currentIndex() );
    stCmd.fMinLength = ui.lineEditCountEdgeMinLength->text().toFloat();
    PR_FindEdge ( &stCmd, &stRpy );
    if ( VisionStatus::OK == stRpy.enStatus ) {
        ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
        ui.lineEditEdgeCount->setText ( std::to_string( stRpy.nEdgeCount ).c_str() );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrStrRpy);
        QMessageBox::critical(nullptr, "Find Edge", stErrStrRpy.achErrorStr, "Quit");
        ui.lineEditObjCenter->clear();
        ui.lineEditObjRotation->clear();
    }
}

void VisionWidget::on_btnLrnContour_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_LRN_CONTOUR_CMD stCmd;
    PR_LRN_CONTOUR_RPY stRpy;
    stCmd.bAutoThreshold = true;
    stCmd.matInputImg = ui.visionView->getMat();
    stCmd.rectROI = ui.visionView->getSelectedWindow();
    PR_LrnContour ( &stCmd, &stRpy );
    if ( VisionStatus::OK == stRpy.enStatus ) {
        ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
        _nContourRecordId = stRpy.nRecordId;
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrStrRpy);
        QMessageBox::critical(nullptr, "Learn chip", stErrStrRpy.achErrorStr, "Quit");
        ui.lineEditObjCenter->clear();
        ui.lineEditObjRotation->clear();
    }
}

void VisionWidget::on_btnInspContour_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_INSP_CONTOUR_CMD stCmd;
    PR_INSP_CONTOUR_RPY stRpy;
    stCmd.matInputImg = ui.visionView->getMat();
    stCmd.rectROI = ui.visionView->getSelectedWindow();
    stCmd.nRecordId = _nContourRecordId;
    PR_InspContour ( &stCmd, &stRpy );
    if ( VisionStatus::OK == stRpy.enStatus ) {
        ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrStrRpy);
        if ( stErrStrRpy.enErrorLevel == PR_STATUS_ERROR_LEVEL::INSP_STATUS )
            ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
        QMessageBox::critical(nullptr, "Learn chip", stErrStrRpy.achErrorStr, "Quit");
        ui.lineEditObjCenter->clear();
        ui.lineEditObjRotation->clear();
    }
}

void VisionWidget::on_btnInspHole_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_INSP_HOLE_CMD stCmd;
    PR_INSP_HOLE_RPY stRpy;
    stCmd.matInputImg = ui.visionView->getMat();
    stCmd.rectROI = ui.visionView->getSelectedWindow();
    stCmd.enSegmentMethod = static_cast<PR_IMG_SEGMENT_METHOD>(ui.comboBoxInspHoleImgSegmentMethod->currentIndex());
    stCmd.stGrayScaleRange.nStart = ui.lineEditInspHoleImgSegmentStart->text().toInt();
    stCmd.stGrayScaleRange.nEnd = ui.lineEditInspHoleImgSegmentEnd->text().toInt();
    stCmd.enInspMode = static_cast<PR_INSP_HOLE_MODE>(ui.comboBoxHoleInspMode->currentIndex());
    PR_InspHole ( &stCmd, &stRpy );
    if ( VisionStatus::OK == stRpy.enStatus ) {
        ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrStrRpy);
        if ( stErrStrRpy.enErrorLevel == PR_STATUS_ERROR_LEVEL::INSP_STATUS )
            ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
        QMessageBox::critical(nullptr, "Learn chip", stErrStrRpy.achErrorStr, "Quit");
        ui.lineEditObjCenter->clear();
        ui.lineEditObjRotation->clear();
    }
}

void VisionWidget::on_btnAutoLocateLead_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    std::unique_ptr<MessageBoxDialog> pMessageBox = std::make_unique<MessageBoxDialog>();
    pMessageBox->setGeometry(700, 500, pMessageBox->size().width(), pMessageBox->size().height());
    pMessageBox->setWindowTitle("Auto Locate Lead");
	pMessageBox->SetMessageText1("Please input the chip window");
    pMessageBox->SetMessageText2("Press and drag the left mouse buttont to input");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_MULTIPLE_WINDOW);
    ui.visionView->setCurrentSrchWindowIndex(0);
	int iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
        return;
    }

    pMessageBox->SetMessageText1("Please input the lead search window");
    pMessageBox->SetMessageText2("Press and drag the left mouse buttont to input");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_MULTIPLE_WINDOW);
    ui.visionView->setCurrentSrchWindowIndex(1);

	iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
        return;
    }
   
    auto _vecSrchWindow = ui.visionView->getVecSrchWindow();
    if ( _vecSrchWindow.size() <= 1 )   {
        pMessageBox->SetMessageText1("The input is invalid");
        pMessageBox->SetMessageText2("");
        pMessageBox->exec();
        ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
        return;
    }

    PR_AUTO_LOCATE_LEAD_CMD stCmd;
    PR_AUTO_LOCATE_LEAD_RPY stRpy;
    stCmd.enLeadAttribute = PR_OBJECT_ATTRIBUTE::BRIGHT;
    stCmd.matInputImg = ui.visionView->getMat();
    stCmd.rectChipBody = _vecSrchWindow[0];
    stCmd.rectSrchWindow = _vecSrchWindow[1];
    
    PR_AutoLocateLead ( &stCmd, &stRpy );
    if ( VisionStatus::OK == stRpy.enStatus ) {
        ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrStrRpy);
        if ( stErrStrRpy.enErrorLevel == PR_STATUS_ERROR_LEVEL::INSP_STATUS )
            ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
        QMessageBox::critical(nullptr, "Auto Locate Lead", stErrStrRpy.achErrorStr, "Quit");
        ui.lineEditObjCenter->clear();
        ui.lineEditObjRotation->clear();
    }
}

void VisionWidget::on_btnInspLead_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    std::unique_ptr<MessageBoxDialog> pMessageBox = std::make_unique<MessageBoxDialog>();
    pMessageBox->setGeometry(700, 500, pMessageBox->size().width(), pMessageBox->size().height());
    pMessageBox->setWindowTitle("Inspect Lead");
	pMessageBox->SetMessageText1("Please input the chip window");
    pMessageBox->SetMessageText2("Press and drag the left mouse buttont to input");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_MULTIPLE_WINDOW);
    ui.visionView->setCurrentSrchWindowIndex(0);
	int iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
        return;
    }

    pMessageBox->SetMessageText1("Please input the lead search window");
    pMessageBox->SetMessageText2("Press and drag the left mouse buttont to input");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_MULTIPLE_WINDOW);
    ui.visionView->setCurrentSrchWindowIndex(1);

	iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
        return;
    }
   
    auto _vecSrchWindow = ui.visionView->getVecSrchWindow();
    if ( _vecSrchWindow.size() <= 1 )   {
        pMessageBox->SetMessageText1("The input is invalid");
        pMessageBox->SetMessageText2("");
        pMessageBox->exec();
        ui.visionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
        return;
    }

    PR_INSP_LEAD_CMD stCmd;
    PR_INSP_LEAD_RPY stRpy;
    stCmd.matInputImg = ui.visionView->getMat();
    stCmd.rectChipWindow = rectToRotatedRect ( _vecSrchWindow[0] );
    PR_INSP_LEAD_CMD::LEAD_INPUT_INFO stLeadInput;
    stLeadInput.rectSrchWindow = rectToRotatedRect ( _vecSrchWindow[1] );
    stCmd.vecLeads.push_back ( stLeadInput );
    
    PR_InspLead ( &stCmd, &stRpy );
    if ( VisionStatus::OK == stRpy.enStatus ) {
        ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrStrRpy);
        if ( stErrStrRpy.enErrorLevel == PR_STATUS_ERROR_LEVEL::INSP_STATUS )
            ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
        QMessageBox::critical(nullptr, "Inspect Lead", stErrStrRpy.achErrorStr, "Quit");
        ui.lineEditObjCenter->clear();
        ui.lineEditObjRotation->clear();
    }
}

void VisionWidget::on_btnGridAvgGrayScale_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_GRID_AVG_GRAY_SCALE_CMD stCmd;
    PR_GRID_AVG_GRAY_SCALE_RPY stRpy;
    stCmd.vecInputImgs.push_back ( ui.visionView->getMat() );
    stCmd.vecInputImgs.push_back ( ui.visionView->getMat() );
    stCmd.vecInputImgs.push_back ( ui.visionView->getMat() );
    stCmd.vecInputImgs.push_back ( ui.visionView->getMat() );

    stCmd.nGridRow = ui.lineEditGridRows->text().toInt();
    stCmd.nGridCol = ui.lineEditGridCols->text().toInt();

    PR_GridAvgGrayScale ( &stCmd, &stRpy );
    if ( VisionStatus::OK == stRpy.enStatus ) {
        ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrStrRpy);
        if ( stErrStrRpy.enErrorLevel == PR_STATUS_ERROR_LEVEL::INSP_STATUS )
            ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResultImg );
        QMessageBox::critical(nullptr, "Inspect Lead", stErrStrRpy.achErrorStr, "Quit");
    }
}

void VisionWidget::on_btnCameraMTF_clicked() {
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();
    QString strArrayTitle[] = {
        "Please input the vertical big pattern window",
        "Please input the horizontal big pattern window",
        "Please input the vertical small pattern window",
        "Please input the horizontal small pattern window",
    };

    PR_CALC_CAMERA_MTF_CMD stCmd;
    PR_CALC_CAMERA_MTF_RPY stRpy;
    for ( int i = 0; i < 4; ++ i ) {
        std::unique_ptr<MessageBoxDialog> pMessageBox = std::make_unique<MessageBoxDialog> ();
        pMessageBox->setGeometry ( 700, 500, pMessageBox->size ().width (), pMessageBox->size ().height () );
        pMessageBox->setWindowTitle ( "Camera MTF" );
        pMessageBox->SetMessageText1 ( strArrayTitle[i] );
        pMessageBox->SetMessageText2 ( "Press and drag the left mouse buttont to input" );
        pMessageBox->setWindowFlags ( Qt::WindowStaysOnTopHint );
        pMessageBox->show ();
        pMessageBox->raise ();
        pMessageBox->activateWindow ();
        ui.visionView->setTestVisionState ( VisionView::TEST_VISION_STATE::SET_MULTIPLE_WINDOW );
        ui.visionView->setCurrentSrchWindowIndex ( i );
        int iReturn = pMessageBox->exec ();
        if( iReturn != QDialog::Accepted ) {
            ui.visionView->setTestVisionState ( VisionView::TEST_VISION_STATE::IDLE );
            return;
        }
    }
    auto vecRectROI = ui.visionView->getVecSrchWindow();
    stCmd.rectVBigPatternROI = vecRectROI[0];
    stCmd.rectHBigPatternROI = vecRectROI[1];
    stCmd.rectVSmallPatternROI = vecRectROI[2];
    stCmd.rectHSmallPatternROI = vecRectROI[3];

    stCmd.matInputImg = ui.visionView->getMat();
    PR_CalcCameraMTF ( &stCmd, &stRpy );
    if ( VisionStatus::OK == stRpy.enStatus ) {
        ui.lineEditBigPatternAbsMTFV->setText( std::to_string ( stRpy.fBigPatternAbsMtfV ).c_str() );
        ui.lineEditBigPatternAbsMTFH->setText( std::to_string ( stRpy.fBigPatternAbsMtfH ).c_str() );
        ui.lineEditSmallPatternAbsMTFV->setText( std::to_string ( stRpy.fSmallPatternAbsMtfV ).c_str() );
        ui.lineEditSmallPatternAbsMTFH->setText( std::to_string ( stRpy.fSmallPatternAbsMtfH ).c_str() );
        ui.lineEditSmallPatternRelMTFV->setText( std::to_string ( stRpy.fSmallPatternRelMtfV ).c_str() );
        ui.lineEditSmallPatternRelMTFH->setText( std::to_string ( stRpy.fSmallPatternRelMtfH ).c_str() );
    }else {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo(stRpy.enStatus, &stErrStrRpy);        
        QMessageBox::critical(nullptr, "Camera MTF", stErrStrRpy.achErrorStr, "Quit");
    }
}