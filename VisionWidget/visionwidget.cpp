#include "visionwidget.h"
#include "FitCircleProcedure.h"
#include "FitLineProcedure.h"
#include "FitParallelLineProcedure.h"
#include "FitRectProcedure.h"
#include "OcrProcedure.h"
#include "SrchFiducialProcedure.h"
#include "constants.h"
#include "VisionAPI.h"
#include <QFileDialog>
#include <QMessageBox>

using namespace AOI::Vision;

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

    _ptrEdgeDetectWidget = std::make_unique<EdgeDetectWidget>(this);
    _ptrEdgeDetectWidget->setVisionView ( ui.visionView );
    ui.verticalLayout->addWidget(_ptrEdgeDetectWidget.get());

    _ptrCcWidget = std::make_unique<CCWidget>(this);
    _ptrCcWidget->setVisionView ( ui.visionView );
    ui.verticalLayout->addWidget(_ptrCcWidget.get());
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

	FitCircleProcedure procedure(ui.visionView);
    procedure.setErrTol ( ui.lineEditFitCircleErrTol->text().toFloat());
    procedure.setDirection ( ui.comboBoxFitCircleDirection->currentIndex());
    //procedure.setThreshold ( ui.lineEditFitCircleThreshold->text().toInt());
    //procedure.setAttribute ( ui.comboBoxCircleAttribute->currentIndex());
    procedure.setAlgorithm ( ui.comboBoxFitCircleAlgorithm->currentIndex());
	int nStatus = procedure.run(_sourceImagePath);
    if ( ToInt(VisionStatus::OK) == nStatus )
    {
        ui.visionView->setMat(VisionView::DISPLAY_SOURCE::RESULT, procedure.getResultMat());
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

    PR_CIRCLE_ROUNDNESS_CMD stCmd;
	stCmd.matInput = ui.visionView->getMat();
    stCmd.matMask = ui.visionView->getMask();
    stCmd.rectROI = ui.visionView->getSelectedWindow();

	PR_CIRCLE_ROUNDNESS_RPY stRpy;
	VisionStatus enStatus = PR_CircleRoundness(&stCmd, &stRpy);
	if (VisionStatus::OK == enStatus)	{
		ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResult );
        ui.lineEditRoundnessResult->setText( std::to_string(stRpy.fRoundness).c_str() );
    }
}

void VisionWidget::on_fitLineBtn_clicked()
{
    if ( ! checkDisplayImage() )
        return;

	FitLineProcedure procedure(ui.visionView);
    procedure.setErrTol ( ui.lineEditFitLineErrTol->text().toFloat());
    procedure.setThreshold ( ui.lineEditFitLineThreshold->text().toInt());
    procedure.setAttribute ( ui.comboBoxLineAttribute->currentIndex());
	int nStatus = procedure.run(_sourceImagePath);
    if ( ToInt(VisionStatus::OK) == nStatus )
    {
        ui.visionView->setMat(VisionView::DISPLAY_SOURCE::RESULT, procedure.getResultMat());
    }
}

void VisionWidget::on_detectLineBtn_clicked()
{
    if ( _sourceImagePath.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    ui.visionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    ui.visionView->applyIntermediateResult();

    PR_DETECT_LINE_CMD stCmd;
    stCmd.matInput = ui.visionView->getMat();
    stCmd.matMask = ui.visionView->getMask();
    stCmd.rectROI = ui.visionView->getSelectedWindow();
    stCmd.enDetectDir = static_cast<PR_DETECT_LINE_DIR> ( ui.comboBoxDetectLineDirection->currentIndex() );

    PR_DETECT_LINE_RPY stRpy;
	VisionStatus enStatus = PR_DetectLine(&stCmd, &stRpy);
	if (VisionStatus::OK == enStatus)	{
		ui.visionView->setMat ( VisionView::DISPLAY_SOURCE::RESULT, stRpy.matResult );
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

void VisionWidget::on_addPreProcessorBtn_clicked()
{
    //FilterWidget *pFW = new FilterWidget(this);
    //pFW->setParent(this);
    //pFW->setVisionView ( ui.visionView );
    //ui.verticalLayout->addWidget(pFW);
}