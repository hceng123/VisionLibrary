#include "dialogvisiontoolbox.h"
#include "ui_dialogvisiontoolbox.h"
#include "CheckIntensityProcedure.h"
#include <future>
#include <QFileDialog>
#include <QMessageBox>

using namespace AOI::Vision;

DialogVisionToolBox::DialogVisionToolBox(QWidget *parent, VisionView *pVisionView) :
    QDialog(parent),
    _pVisionView(pVisionView),
    ui(new Ui::DialogVisionToolBox)
{
    ui->setupUi(this);
}

DialogVisionToolBox::~DialogVisionToolBox()
{
    delete ui;
}

void DialogVisionToolBox::on_comboBoxDebugMode_currentIndexChanged(int index)
{
    PR_DEBUG_MODE enDebugMode = static_cast<PR_DEBUG_MODE>(index);
    PR_SetDebugMode ( enDebugMode );
}

void DialogVisionToolBox::on_btnDumpTimeLog_clicked()
{
    QFileDialog dialog(this);
    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setNameFilter(tr("Log Files (*.log *.txt)"));
    dialog.setViewMode(QFileDialog::Detail);
    if ( ! dialog.exec() )
        return;

    QStringList fileNames = dialog.selectedFiles();
    std::string filePath = fileNames[0].toStdString();

    std::async( std::launch::async,
        [filePath] ()
        { PR_DumpTimeLog(filePath); }
    );
}

void DialogVisionToolBox::on_btnCheckIntensity_clicked()
{
    _pVisionView->applyIntermediateResult();
    cv::Mat mat = _pVisionView->getMat(VisionView::DISPLAY_SOURCE::ORIGINAL);
    if ( mat.empty() ) {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    if ( _pVisionView->isDisplayResultImage()) {
        QMessageBox::information(this, "Vision Widget", "Please swap to the original image first!", "Quit");
        return;
    }

    CheckIntensityProcedure procedure(_pVisionView);
    procedure.run(mat);    
}

void DialogVisionToolBox::on_btnReplayLogcase_clicked()
{

    QFileDialog dialog(this);
    dialog.setFileMode(QFileDialog::ExistingFile);
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setNameFilter(tr("Log Files (*.logcase)"));
    dialog.setViewMode(QFileDialog::Detail);
    if ( ! dialog.exec() )
        return;

    QStringList fileNames = dialog.selectedFiles();
    std::string filePath = fileNames[0].toStdString();
    PR_RunLogCase(filePath);
}
