#include "dialogeditmask.h"
#include "ui_dialogeditmask.h"

DialogEditMask::DialogEditMask(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogEditMask)
{
    ui->setupUi(this);
    _pVisionView = NULL;
}

DialogEditMask::~DialogEditMask()
{
    delete ui;
}

void DialogEditMask::on_radioButtonRect_clicked(bool checked)
{
    if ( checked && NULL != _pVisionView )
        _pVisionView->setMaskShape( VisionView::MASK_SHAPE_RECT );
}

void DialogEditMask::on_radioButtonCircle_clicked(bool checked)
{
    if ( checked && NULL != _pVisionView )
        _pVisionView->setMaskShape ( VisionView::MASK_SHAPE_CIRCLE );
}

void DialogEditMask::on_radioButtonPolyLine_clicked(bool checked)
{
    if ( checked && NULL != _pVisionView )
        _pVisionView->setMaskShape ( VisionView::MASK_SHAPE_POLYLINE );
}

void DialogEditMask::on_radioButtonAddMask_clicked(bool checked)
{
    if ( checked && NULL != _pVisionView )
        _pVisionView->setMaskEditState( VisionView::MASK_EDIT_STATE::ADD );
}

void DialogEditMask::on_radioButtonRemoveMask_clicked(bool checked)
{
    if ( checked && NULL != _pVisionView )
        _pVisionView->setMaskEditState( VisionView::MASK_EDIT_STATE::REMOVE );
}

void DialogEditMask::setVisionView(VisionView *pVisionView)
{
    _pVisionView = pVisionView;
}
