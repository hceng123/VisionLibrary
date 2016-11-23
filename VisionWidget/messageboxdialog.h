#ifndef MESSAGEBOXDIALOG_H
#define MESSAGEBOXDIALOG_H

#include <QDialog>

namespace Ui {
class MessageBoxDialog;
}

class MessageBoxDialog : public QDialog
{
    Q_OBJECT

public:
    explicit MessageBoxDialog(QWidget *parent = 0);
    ~MessageBoxDialog();
    void SetMessageText1(const QString &text);
    void SetMessageText2(const QString &text);
    void SetMessageText3(const QString &text);
    void ConnectBtn(QWidget *pWidget, void(*func)());
    void closeEvent(QCloseEvent *);
private:
    Ui::MessageBoxDialog *ui;
};

#endif // MESSAGEBOXDIALOG_H
