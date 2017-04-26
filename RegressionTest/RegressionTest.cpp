// RegressionTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "TestFunction.h"

using namespace AOI::Vision;

int _tmain(int argc, _TCHAR* argv[])
{
    /*************************
    * Test Device Inspection *
    **************************/
    TestInspDevice();
    TestInspDeviceAutoThreshold();

    /*************************
    * Test Device Inspection *
    **************************/
    TestRunLogcase();

    /*************************
    * Test Template Matching *
    **************************/
    TestTmplMatch();
    TestTmplMatch_Circle();

    /********************
    * Test Fitting Line *
    *********************/
    TestFitLine();

    /*****************************
    * Test Fitting Parallel Line *
    *****************************/
    TestFitParellelLine();

    /********************
    * Test Fitting Rect *
    *********************/
    TestFitRect();

    /**********************
    * Test Fitting Circle *
    ***********************/
    TestFitCircle();

    /**********************
    * Test Auto Threshold *
    ***********************/
    TestAutoThreshold();

    /*******************
    * Test Detect Line *
    *******************/
    TestDetectLine();

    /*******************
    * Test Calibrate Camera *
    *******************/
    TestCalibrateCamera();

	return 0;
}

