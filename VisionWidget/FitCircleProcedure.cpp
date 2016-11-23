#include "FitCircleProcedure.h"
#include "messageboxdialog.h"
#include <QMessageBox>

FitCircleProcedure::FitCircleProcedure(VisionView *pVisionView) : Procedure(pVisionView)
{
}

FitCircleProcedure::~FitCircleProcedure()
{
}

int FitCircleProcedure::run()
{
	MessageBoxDialog *pMessageBox = new MessageBoxDialog();
	pMessageBox->SetMessageText1("Hello world");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();	
	pMessageBox->activateWindow();
	int iReturn = pMessageBox->exec();
	if (iReturn == QDialog::Accepted)
	{
		QMessageBox messageBox;
		messageBox.setText("Accepted");
		messageBox.exec();
	}
	else   {
		QMessageBox messageBox;
		messageBox.setText("Rejected");
		messageBox.exec();
	}
	//ui->pushButton->setEnabled(true);
    return 0;
}