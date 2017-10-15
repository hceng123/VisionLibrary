// RegressionTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "TestFunction.h"
#include "../VisionLibrary/VisionAPI.h"

using namespace AOI::Vision;

int _tmain(int argc, _TCHAR* argv[])
{
    /*****************************************
    * Test Vision Library Internal Functions *
    *****************************************/
    _PR_InternalTest();

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
    TestSrchFiducialMark();

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

    /***************
    * Test Caliper *
    ***************/
    TestCaliper();

    /************************
    * Test Calibrate Camera *
    ************************/
    TestCalibrateCamera();

    /************************
    * Test Auto Locate Lead *
    ************************/
    TestAutoLocateLead();

    /**********************
    * Test Inspect Bridge *
    ***********************/
    TestInspBridge();

    /*************************
    * Test Inspect Chip Head *
    **************************/
    TestInspChipHead();

    /*************************
    * Test Inspect Chip Body *
    **************************/
    TestInspChipBody();

    /***************************
    * Test Inspect Chip Square *
    ****************************/
    TestInspChipSquare();

    /************************
    * Test Inspect Chip CAE *
    *************************/
    TestInspChipCAE();

    /*****************************
    * Test Inspect Chip Circular *
    ******************************/
    TestInspChipCircular();

    /*****************************
    * Test Inspect Contour *
    ******************************/
    TestInspContour();

    Test3D();

    TestCalcCameraMTF();

    PR_DumpTimeLog("./Vision/Time.log");
	return 0;
}

