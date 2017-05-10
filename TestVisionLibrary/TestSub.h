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