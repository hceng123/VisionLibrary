#include "visionwidget.h"
#include <QFileDialog>
#include "FitCircleProcedure.h"

VisionWidget::VisionWidget(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    ui.visionViewToolBox->SetVisionView(ui.visionView);
    ui.lineEditFitCircleErrTol->setValidator(new QIntValidator(1,100, this) );
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
}

void VisionWidget::on_fitCircleBtn_clicked()
{
	FitCircleProcedure procedure(ui.visionView);
	procedure.run();
}