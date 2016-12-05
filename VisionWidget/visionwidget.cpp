#include "visionwidget.h"
#include <QFileDialog>
#include "FitCircleProcedure.h"
#include "FitLineProcedure.h"
#include "constants.h"

using namespace AOI::Vision;

VisionWidget::VisionWidget(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    ui.visionViewToolBox->SetVisionView(ui.visionView);
    ui.lineEditFitCircleErrTol->setValidator(new QIntValidator(1,100, this) );
    ui.visionView->setMachineState(VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY);
}

VisionWidget::~VisionWidget()
{

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
    ui.visionView->setImageFile( fileNames[0].toStdString() );
    _sourceImagePath = fileNames[0].toStdString();
}

void VisionWidget::on_fitCircleBtn_clicked()
{
	FitCircleProcedure procedure(ui.visionView);
    procedure.setErrTol ( ui.lineEditFitCircleErrTol->text().toFloat());
    procedure.setThreshold ( ui.lineEditFitCircleThreshold->text().toInt());
    procedure.setAlgorithm ( ui.comboBoxFitCircleAlgorithm->currentIndex());
	int nStatus = procedure.run(_sourceImagePath);
    if ( ToInt(VisionStatus::OK) == nStatus )
    {
        ui.visionView->setMat(procedure.getResultMat());
    }
}

void VisionWidget::on_fitLineBtn_clicked()
{
	FitLineProcedure procedure(ui.visionView);
    procedure.setErrTol ( ui.lineEditFitLineErrTol->text().toFloat());
    procedure.setThreshold ( ui.lineEditFitLineThreshold->text().toInt());
    //procedure.setAlgorithm ( ui.comboBoxFitCircleAlgorithm->currentIndex());
	int nStatus = procedure.run(_sourceImagePath);
    if ( ToInt(VisionStatus::OK) == nStatus )
    {
        ui.visionView->setMat(procedure.getResultMat());
    }
}