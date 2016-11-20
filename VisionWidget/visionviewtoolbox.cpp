#include "VisionViewToolBox.h"
#include "ui_visionviewtoolbox.h"

VisionViewToolBox::VisionViewToolBox(QWidget *parent) :
    QWidget(parent)
{
    ui = new Ui::VisionViewToolBox;
    ui->setupUi(this);
    _pVisionView = NULL;
}

VisionViewToolBox::~VisionViewToolBox()
{
    delete ui;
}

void VisionViewToolBox::on_zoomInButton_clicked()
{
    if ( _pVisionView == NULL )
        return;
    _pVisionView->zoomIn();
}

void VisionViewToolBox::on_zoomOutButton_clicked()
{
    _pVisionView->zoomOut();
}

void VisionViewToolBox::SetVisionView(VisionView *pVisionView)
{
    _pVisionView = pVisionView;
}

void VisionViewToolBox::on_toolBoxPushButton_clicked()
{
}
