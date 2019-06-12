/*****************************************************************************
 * Version.h -- $Id$
 *
 * Purpose
 *   This file includes the version of the library and the change history.
 *
 * Indentation
 *   Four characters. No tabs!
 *
 * Modifications
 *   2017-09-13 (XSG) Add PR_CalcPD function to calculate system pattern distortion.
 *   2016-10-14 (XSG) Created.
 *
 * Copyright (c) 2016-2018, Xiao Shengguang.  All rights reserved.
 ****************************************************************************/

//2019-06-11 (XSG) Add PR_MeasureDist function.
#define AOI_VISION_VERSION          "1.00.69"

//2019-05-27 (XSG) Add PR_CalcMerge4DlpHeight function.
//#define AOI_VISION_VERSION          "1.00.68"

//2019-04-28 (XSG) Add fHeightVariationCoverage to PR_INSP_3D_SOLDER_CMD so user can change the value.
//#define AOI_VISION_VERSION          "1.00.67"

//2019-04-21 (XSG) Fix pick color may choose different color problem.
//#define AOI_VISION_VERSION          "1.00.66"

//2019-03-29 (XSG) Return lead and pad position in PR_InspLeadTmpl.
//#define AOI_VISION_VERSION          "1.00.65"

//2019-03-25 (XSG) Fix inspect similarity the min similarity not correct bug.
//#define AOI_VISION_VERSION          "1.00.64"

//2019-02-23 (XSG) Develop PR_CalcRestoreIdx function.
//#define AOI_VISION_VERSION          "1.00.63"

//2019-02-18 (XSG) Develop PR_CalibDlpOffset function.
//#define AOI_VISION_VERSION          "1.00.62"

//2019-02-05 (XSG) Fix auto locate lead bugs.
//                 Fix height detect bugs.
//                 Optimize log ROI for match template and srch fiducial.
//#define AOI_VISION_VERSION          "1.00.61"

//2019-01-27 (XSG) Add PR_CombineImgNew function.
//#define AOI_VISION_VERSION          "1.00.60"

//2019-01-19 (XSG) Update table mapping algorithm.
//#define AOI_VISION_VERSION          "1.00.59"

//2019-01-16 (XSG) Update calibrate camera function.
//#define AOI_VISION_VERSION          "1.00.58"

//2019-01-12 (XSG) Update calculate table offset function.
//#define AOI_VISION_VERSION          "1.00.57"

//2019-01-10 (XSG) Add border points weight for table mapping calibration.
//#define AOI_VISION_VERSION          "1.00.56"

//2019-01-03 (XSG) Fix template match log case not correct problem.
//#define AOI_VISION_VERSION          "1.00.55"

//2018-12-27 (XSG) Fix single frame table mapping fail problem
//#define AOI_VISION_VERSION          "1.00.54"

//2018-12-25 (XSG) Add matResultMask in PR_PICK_COLOR_RPY
//#define AOI_VISION_VERSION          "1.00.53"

//2018-12-04 (XSG) Add PR_TableMapping and PR_CalcTableOffset functions.
//#define AOI_VISION_VERSION          "1.00.52"

//2018-11-16 (XSG) Add PR_InspSimilarity function.
//#define AOI_VISION_VERSION          "1.00.51"

//2018-11-09 (XSG) Fix PR_LrnOcv has dead loop problem.
//#define AOI_VISION_VERSION          "1.00.50"

//2018-10-14 (XSG) PR_OCV support accept reversed character.
//#define AOI_VISION_VERSION          "1.00.49"

//2018-09-29 (XSG) Add PR_GetOcvRecordInfo and PR_SetOcvRecordInfo API function.
//#define AOI_VISION_VERSION          "1.00.48"

//2018-09-28 (XSG) Fix PR_Insp3DSolder change original height data issue.
//#define AOI_VISION_VERSION          "1.00.47"

//2018-09-24 (XSG) Support rotated OCV inspection.
//#define AOI_VISION_VERSION          "1.00.46"

//2018-09-22 (XSG) Fix PR_Calc3DHeightDiff not correct problem, and improve the speed.
//                 Add LogCaseCalc3DHeightDiff.
//#define AOI_VISION_VERSION          "1.00.45"

//2018-09-12 (XSG) Add PR_Insp3DSolder function.
//#define AOI_VISION_VERSION          "1.00.44"

//2018-09-05 (XSG) Enhance new phase correction algorithm
//                 Support intersect algorithm to merge 3D height.
//#define AOI_VISION_VERSION          "1.00.43"

//2018-09-02 (XSG) Integrate new phase correction algorithm in PR_Calc3DHeiht.
//#define AOI_VISION_VERSION          "1.00.42"

//2018-08-22 (XSG) Add matMask in PR_CALC_3D_HEIGHT_DIFF_CMD
//#define AOI_VISION_VERSION          "1.00.41"

//2018-08-07 (XSG) Add matMask in PR_LRN_TEMPLATE_CMD.
//#define AOI_VISION_VERSION          "1.00.40"

//2018-08-06 (XSG) Add PR_GetRecordInfo function.
//                 Display the detailed result of OCV on result image.
//#define AOI_VISION_VERSION          "1.00.39"

//2018-08-01 (XSG) Add median filter for split image in learn OCV.
//#define AOI_VISION_VERSION          "1.00.38"

//2018-07-25 (XSG) Add PR_LrnOcv and PR_Ocv functions.
//                 PR_AutoLocateLead by template support horizontal lead as template.
//#define AOI_VISION_VERSION          "1.00.37"

//2018-07-19 (XSG) Enhance PR_AutoLocateLead, add fill hole utility in it to fix dark area in lead.
//#define AOI_VISION_VERSION          "1.00.36"

//2018-07-09 (XSG) Add log case for PR_AutoLocateLead.
//#define AOI_VISION_VERSION          "1.00.35"

//2018-07-03 (XSG) Enhance PR_AutoLocateLead function. Add PR_InspLeadTmpl function.
//#define AOI_VISION_VERSION          "1.00.34"

//2018-06-24 (XSG) Update PR_FindLine to follow the style of other inspection windows.
//#define AOI_VISION_VERSION          "1.00.33"

//2018-06-16 (XSG) Modify PR_InspBridge to follow the style of other inspection windows.
//#define AOI_VISION_VERSION          "1.00.32"

//2018-06-07 (XSG) For rebuild the library for PR_CalcDlpOffset test.
//#define AOI_VISION_VERSION          "1.00.31"

//2018-06-05 (XSG) Add PR_CalcDlpOffset and PR_CalcFrameValue functions.
//#define AOI_VISION_VERSION          "1.00.30"

//2018-05-25 (XSG) Return calculated height information in PR_MotorCalib3D function.
//#define AOI_VISION_VERSION          "1.00.29"

//2018-05-25 (XSG) Add PR_CalcDlpOffset and PR_CalcFrameValue functions.
//#define AOI_VISION_VERSION          "1.00.28"

//2018-05-17 (XSG) Fix PR_Calib3DBase ReverseSeq calculation not correct problem.
//#define AOI_VISION_VERSION          "1.00.27"

//2018-04-16 (XSG) Fix PR_CombineImage type not follow input image type problem.
//                 Remove not thread safe code.
//#define AOI_VISION_VERSION          "1.00.26"

//2018-03-31 (XSG) Add PR_Threshold function.
//                 Add bPreprocessedImg for PR_INSP_HOLE_CMD.
//#define AOI_VISION_VERSION          "1.00.25"

//2018-02-09 (XSG) Change the PR_STATUS_ERROR_LEVEL::NO_ERROR to PR_STATUS_ERROR_LEVEL::PR_NO_ERROR to avoid macro definition.
//                 NO_ERROR make compile fail problem.
//#define AOI_VISION_VERSION          "1.00.24"

//2018-02-06 (XSG) Add PR_SCAN_IMAGE_DIR for PR_CombineImage function.
//#define AOI_VISION_VERSION          "1.00.23"

//2018-01-25 (XSG) Add fMinMatchScore for PR_MatchTmpl and PR_SrchFiducialMark function.
//#define AOI_VISION_VERSION          "1.00.22"

//2018-01-23 (XSG) Auto calculate the ReverseSeq in PR_Calib3DBase function, the bReverseSeq changed from input to output.
//#define AOI_VISION_VERSION          "1.00.21"

//2018-01-21 (XSG) Add PR_MotorCalib3D function.
//#define AOI_VISION_VERSION          "1.00.20"

//2017-12-28 (XSG) Change PR_Caliper to PR_FindLine function.
//#define AOI_VISION_VERSION          "1.00.19"

//2017-12-20 (XSG) Add PR_FindCircle function.
//2017-12-23 (XSG) Add PR_FitLineByPoint and PR_FitCircleByPoint functions.
//#define AOI_VISION_VERSION          "1.00.18"

//2017-12-08 (XSG) Fix interval didn't return correct range size problem.
//#define AOI_VISION_VERSION          "1.00.17"

//2017-12-02 (XSG) Auto offset the phase measure range.
//#define AOI_VISION_VERSION          "1.00.16"

//2017-11-15 (XSG) Fix PR_Calc3DHeight use gamma not correct problem.
//2017-11-19 (XSG) Add PR_TwoLineAngle function to calculate angle of two lines.
//#define AOI_VISION_VERSION          "1.00.15"

//2017-11-13 (XSG) Add fRemoveLowerNoiseRatio in PR_MERGE_3D_HEIGHT_CMD to remove lower noise.
//2017-11-13 (XSG) Move median filter to the end of PR_Merge3DHeight.
//2017-11-13 (XSG) Remove nan value in the output height.
//#define AOI_VISION_VERSION          "1.00.14"

//2017-11-11 (XSG) Update the method to calculate 3D height. Change the phase range from -pi~pi to -1~1.
//2017-11-11 (XSG) Remove PR_Comb3DCalib and PR_FastCalc3DHeight function.
//2017-11-09 (XSG) Support use thinnest pattern.
//#define AOI_VISION_VERSION          "1.00.13"

//2017-11-05 (XSG) Add PR_Merge3DHeight function.
//#define AOI_VISION_VERSION          "1.00.12"

//2017-11-01 (XSG) Speed up PR_FastCalc3DHeight function by using atan2 lookup table.
//#define AOI_VISION_VERSION          "1.00.11"

//2017-10-27 (XSG) Add PR_FastCalc3DHeight function.
//#define AOI_VISION_VERSION          "1.00.10"

//2017-10-09 (XSG) Add PR_Integrate3DCalib function.
//#define AOI_VISION_VERSION          "1.00.09"

//2017-09-29 (XSG) Add PR_CalcCameraMTF function.
//2017-09-28 (XSG) Fix PR_CalibrateCamera fail problem on new optics image.
//#define AOI_VISION_VERSION          "1.00.08"

//2017-09-23 (XSG) Use Nelder-Mead algorithm to do auto threshold to speed up.
//2017-09-23 (XSG) Add PR_Calc3DBase function to avoid repeatly calculate base surface.
//#define AOI_VISION_VERSION          "1.00.07"

//2017-09-17 (XSG) Add PR_Comb3DCalib function.
//2017-09-16 (XSG) Add bSubPixelRefine in PR_MATCH_TEMPLATE_CMD. Default is false.
//#define AOI_VISION_VERSION          "1.00.06"

//2017-09-16 (XSG) Support 5 steps calibration.
//2017-09-16 (XSG) Use new method to do 3D calibrate, remove the top error points to fit surface.
//#define AOI_VISION_VERSION          "1.00.05"

//2017-09-15 (XSG) Add PR_Calc3DHeightDiff function.
//2017-09-14 (XSG) Enhance the Unwrap::_phaseUnwrapSurfaceTrk, add procedure unwrap from bottom to top.
//#define AOI_VISION_VERSION          "1.00.04"

//2017-09-14 (XSG) Add CALIB_3D_HEIGHT_SURFACE_TOO_SMALL status for calib 3D height.
//2017-09-14 (XSG) Remove 4 times harmonic wave.
//2017-09-14 (XSG) Fix Unwrap::_getIntegralTree not finish yet problem. Handle standalone pole points.
//#define AOI_VISION_VERSION          "1.00.03"

//2017-09-13 (XSG) Add PR_CalcPD function to calculate system pattern distortion.
//#define AOI_VISION_VERSION          "1.00.01"