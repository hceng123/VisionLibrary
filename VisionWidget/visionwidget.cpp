#include "visionwidget.h"
#include "FitCircleProcedure.h"
#include "FitLineProcedure.h"
#include "FitParallelLineProcedure.h"
#include "FitRectProcedure.h"
#include "OcrProcedure.h"
#include "SrchFiducialProcedure.h"
#include "constants.h"
#include <QFileDialog>
#include <QMessageBox>

using namespace AOI::Vision;

VisionWidget::VisionWidget(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    setWindowIcon(QIcon(":/VisionWidget/Image/VisionWidget.png"));

    ui.visionViewToolBox->SetVisionView(ui.visionView);
    ui.lineEditFitCircleErrTol->setValidator(new QIntValidator(1,100, this) );
    ui.visionView->setMachineState(VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY);
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

    ui.imagePathEdit->setText(fileNames[0]);

    cv::Mat mat;
    int nChannel;
    if ( ui.checkBoxByerFormat->isChecked())    {
        mat = cv::imread( fileNames[0].toStdString(), IMREAD_GRAYSCALE );
        cv::Mat matColor;
        cv::cvtColor (mat, matColor, CV_BayerGR2BGR );
        _matOriginal = matColor;
    }
    else
        _matOriginal = cv::imread( fileNames[0].toStdString() );
    nChannel = mat.channels();
    ui.visionView->setMat( _matOriginal );
    _sourceImagePath = fileNames[0].toStdString();
}

void VisionWidget::on_fitCircleBtn_clicked()
{
    if ( ! checkDisplayImage() )
        return;
	FitCircleProcedure procedure(ui.visionView);
    procedure.setErrTol ( ui.lineEditFitCircleErrTol->text().toFloat());
    procedure.setThreshold ( ui.lineEditFitCircleThreshold->text().toInt());
    procedure.setAlgorithm ( ui.comboBoxFitCircleAlgorithm->currentIndex());
	int nStatus = procedure.run(_sourceImagePath);
    if ( ToInt(VisionStatus::OK) == nStatus )
    {
        ui.visionView->setResultMat(procedure.getResultMat());
    }
}

void VisionWidget::on_fitLineBtn_clicked()
{
    if ( ! checkDisplayImage() )
        return;

	FitLineProcedure procedure(ui.visionView);
    procedure.setErrTol ( ui.lineEditFitLineErrTol->text().toFloat());
    procedure.setThreshold ( ui.lineEditFitLineThreshold->text().toInt());
	int nStatus = procedure.run(_sourceImagePath);
    if ( ToInt(VisionStatus::OK) == nStatus )
    {
        ui.visionView->setResultMat(procedure.getResultMat());
    }
}

void VisionWidget::on_fitParallelLineBtn_clicked()
{
    if ( ! checkDisplayImage() )
        return;

    FitParallelLineProcedure procedure(ui.visionView);
    procedure.setErrTol ( ui.lineEditFitParallelLineErrTol->text().toFloat());
    procedure.setThreshold ( ui.lineEditFitParallelLineThreshold->text().toInt());
	int nStatus = procedure.run(_sourceImagePath);
    if ( ToInt(VisionStatus::OK) == nStatus )
    {
        ui.visionView->setResultMat(procedure.getResultMat());
    }
}

void VisionWidget::on_fitRectBtn_clicked()
{
    if ( ! checkDisplayImage() )
        return;

    FitRectProcedure procedure(ui.visionView);
    procedure.setErrTol ( ui.lineEditFitRectErrTol->text().toFloat());
    procedure.setThreshold ( ui.lineEditFitRectThreshold->text().toInt());
	int nStatus = procedure.run(_sourceImagePath);
    if ( ToInt(VisionStatus::OK) == nStatus )
    {
        ui.visionView->setResultMat(procedure.getResultMat());
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
        ui.visionView->setResultMat(procedure.getResultMat());
    }
}

void VisionWidget::on_checkBoxDisplayGrayScale_clicked(bool checked)
{
    if ( _matOriginal.empty())
        return;
    if (checked)  {
        float fBRatio = ui.lineEditBRatio->text().toFloat();
        float fGRatio = ui.lineEditGRatio->text().toFloat();
        float fRRatio = ui.lineEditRRatio->text().toFloat();
        std::vector<float> coefficients{ fBRatio, fGRatio, fRRatio };
        cv::Mat matCoefficients = cv::Mat(coefficients).reshape(1, 1);
        cv::Mat matResult;
        cv::transform(_matOriginal, matResult, matCoefficients);
        ui.visionView->setMat(matResult);
    }
    else
    {
        ui.visionView->setMat(_matOriginal);
    }
}

void VisionWidget::on_checkBoxDisplayBinary_clicked(bool checked)
{
}