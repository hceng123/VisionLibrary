#ifndef DIALOGEDITMASK_H
#define DIALOGEDITMASK_H

#include <QDialog>
#include "visionview.h"

namespace Ui {
class DialogEditMask;
}

class VisionView;

class DialogEditMask : public QDialog
{
    Q_OBJECT

public:
    explicit DialogEditMask(QWidget *parent = 0);
    ~DialogEditMask();
    void setVisionView(VisionView *pVisionView);

private slots:
    void on_radioButtonAddMask_clicked(bool checked);

    void on_radioButtonRect_clicked(bool checked);

    void on_radioButtonCircle_clicked(bool checked);

    void on_radioButtonPolyLine_clicked(bool checked);

    void on_radioButtonRemoveMask_clicked(bool checked);

private:
    Ui::DialogEditMask *ui;
    VisionView *_pVisionView;
};

#endif // DIALOGEDITMASK_H
