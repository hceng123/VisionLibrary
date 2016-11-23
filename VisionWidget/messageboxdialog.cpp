#include "messageboxdialog.h"
#include "ui_messageboxdialog.h"

MessageBoxDialog::MessageBoxDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::MessageBoxDialog)
{
    ui->setupUi(this);
}

MessageBoxDialog::~MessageBoxDialog()
{
    delete ui;
}

void MessageBoxDialog::SetMessageText1(const QString &text)
{
    ui->label_1->setText(text);
}

void MessageBoxDialog::SetMessageText2(const QString &text)
{
    ui->label_2->setText(text);
}

void MessageBoxDialog::SetMessageText3(const QString &text)
{
    ui->label_3->setText(text);
}

void MessageBoxDialog::ConnectBtn(QWidget *pWidget, void(*func)())
{
    connect(ui->buttonBox, SIGNAL(accepted()), pWidget, SLOT(func));
}

void MessageBoxDialog::closeEvent(QCloseEvent *)
{
    int i = 100;
}
