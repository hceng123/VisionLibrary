#include "dialogvisiontoolbox.h"
#include "ui_dialogvisiontoolbox.h"

using namespace AOI::Vision;

DialogVisionToolBox::DialogVisionToolBox(QWidget *parent) :
    QDialog(parent),
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