void TestCalibCamera();
void TestCompareInputAndResult();
void TestRunRestoreImgLogCase();
void TestAutoLocateLead();
void TestAutoLocateLead_1();
void TestAutoLocateLead_2();
void TestAutoLocateLead_3();
void TestAutoLocateLeadTmpl_1();
void TestAutoLocateLeadTmpl_2();

void TestInspBridge();
void TestInspBridge_1();
void TestLrnObj();
void TestLrnObj_1();
void TestSrchObj();
void TestSrchDie();
void TestCalibCamera_1();
void TestCalibCamera_2();
void TestInspChipHead();
void TestTemplate();
void TestInspLead();
void TestCalib3dBase();
void TestCalib3dBase_1();
void TestCalc3DHeight();
void TestCalib3DHeight_01();
void TestCalib3DHeight_02();
void TestCalib3DHeight_03();
void TestIntegrate3DCalib();
void TestCalc3DHeight_With_NormalCalibParam();
void TestCalc3DHeight_With_IntegrateCalibParam();

void TestCalcMTF();
void TestCalcPD();
void TestCalcCameraMTF();

void TestFastCalc3DHeight();
void TestIntegrate3DCalibHaoYu();
void TestMerge3DHeight();
void TestFastCalc3DHeight_1();

void TestTwoLineAngle();
void TestTwoLineAngle_1();
void TestTwoLineAngle_2();
void TestTwoLineAngle_3();
void TestTwoLineAngle_4();
void TestTwoLineAngle_5();

void TestPointLineDistance_1();
void TestParallelLineDistance_1();
void TestParallelLineDistance_2();
void TestParallelLineDistance_3();

void TestLineIntersect();
void TestCrossSectionArea();
void TestFindCircle();

void TestCombineImage();
void TestCombineImage_1();
void TestCombineImage_2();
void TestCombineImage_HaoYu();

void TestMotor3DCalib();
void TestSolve();

void TestInspPolarity_01();

void TestPrThreshold();
void TestCalcFrameValue();
void TestCalcFrameValue_1();
void TestCalc4DLPHeight();
void TestQueryDlpOffset();

void TestInsp3DSolder();
void TestInsp3DSolder_1();
void TestInsp3DSolder_2();
void TestConvertToCsv();

void TestOCV_1();
void TestOCV_2();
void TestOCV_3();

void TestMatchTmpl_1();

void TestRead2DCode_1();
void TestRead2DCode_2();

template<class T>
void printfMat(const cv::Mat &mat)
{
    for (short row = 0; row < mat.rows; ++row)
    {
        for (short col = 0; col < mat.cols; ++col)
        {
            printf("%f ", mat.at<T>(row, col));
        }
        printf("\n");
    }
}

template<class T>
void printfVectorOfVector(const std::vector<std::vector<T>> &vevVecInput)
{
    for (const auto &vecInput : vevVecInput )
    {
        for (const auto value : vecInput )
        {
            printf("%.2f ", value);
        }
        printf("\n");
    }
}

template<typename _Tp>
inline std::vector<std::vector<_Tp>> matToVector(const cv::Mat &matInputImg) {
    std::vector<std::vector<_Tp>> vecVecArray;
    if ( matInputImg.isContinuous() ) {
        for ( int row = 0; row < matInputImg.rows; ++ row ) {            
            std::vector<_Tp> vecRow;
            int nRowStart = row * matInputImg.cols;
            vecRow.assign ( (_Tp *)matInputImg.datastart + nRowStart, (_Tp *)matInputImg.datastart + nRowStart + matInputImg.cols );
            vecVecArray.push_back(vecRow);
        }
    }else {
        for ( int row = 0; row < matInputImg.rows; ++ row ) {
            std::vector<_Tp> vecRow;
            vecRow.assign((_Tp*)matInputImg.ptr<uchar>(row), (_Tp*)matInputImg.ptr<uchar>(row) + matInputImg.cols);
            vecVecArray.push_back(vecRow);
        }
    }
    return vecVecArray;
}