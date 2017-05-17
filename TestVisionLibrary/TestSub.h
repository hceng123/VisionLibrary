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
void TestCalibCamera_1();
void TestCalibCamera_2();

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

template<typename _Tp>
inline std::vector<std::vector<_Tp>> matToVector(const cv::Mat &matInput) {
    std::vector<std::vector<_Tp>> vecVecArray;
    if ( matInput.isContinuous() ) {
        for ( int row = 0; row < matInput.rows; ++ row ) {            
            std::vector<_Tp> vecRow;
            int nRowStart = row * matInput.cols;
            vecRow.assign ( (_Tp *)matInput.datastart + nRowStart, (_Tp *)matInput.datastart + nRowStart + matInput.cols );
            vecVecArray.push_back(vecRow);
        }
    }else {
        for ( int row = 0; row < matInput.rows; ++ row ) {
            std::vector<_Tp> vecRow;
            vecRow.assign((_Tp*)matInput.ptr<uchar>(row), (_Tp*)matInput.ptr<uchar>(row) + matInput.cols);
            vecVecArray.push_back(vecRow);
        }
    }
    return vecVecArray;
}