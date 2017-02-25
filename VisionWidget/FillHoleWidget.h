#pragma once

#include <QWidget>
#include "ui_FillHoleWidget.h"
#include "BaseProcessWidget.h"

class FillHoleWidget : public BaseProcessWidget
{
    Q_OBJECT

public:
    FillHoleWidget(QWidget *parent = Q_NULLPTR);
    ~FillHoleWidget();
    virtual std::string myName() const override;

private slots:
    void on_btnRun_clicked();
    void on_cbAlgorithm_currentIndexChanged(int index);

private:
    Ui::FillHoleWidget ui;
};
