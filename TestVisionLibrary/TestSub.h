void TestCalibCamera();
void TestCompareInputAndResult();
void TestRunRestoreImgLogCase();
void TestAutoLocateLead();
void TestAutoLocateLead_1();
void TestAutoLocateLead_2();
void TestAutoLocateLead_3();
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
void TestCalc3DHeight();
void TestCalib3DHeight();
void TestComb3DCalib();
void TestCalcMTF();
void TestCalcPD();
void TestCalcCameraMTF();
void TestIntegrate3DCalib();
void TestCalc3DHeightNew();
void TestFastCalc3DHeight();
void TestIntegrate3DCalibHaoYu();

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