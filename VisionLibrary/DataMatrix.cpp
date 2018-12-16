#include "DataMatrix.h"

#include <zxing/LuminanceSource.h>
#include <zxing/common/Counted.h>
#include <zxing/Reader.h>
#include <zxing/aztec/AztecReader.h>
#include <zxing/common/GlobalHistogramBinarizer.h>
#include <zxing/DecodeHints.h>
#include <zxing/datamatrix/DataMatrixReader.h>
#include <zxing/MultiFormatReader.h>
#include "zxing/MatSource.h"

namespace AOI
{
namespace Vision
{

/*static*/ bool DataMatrix::read(const cv::Mat &matInput, String &strResult) {
    cv::Mat matLocal = matInput;
    zxing::Ref<zxing::LuminanceSource> source = MatSource::create(matLocal);
    int width = source->getWidth();
    int height = source->getHeight();
    
    zxing::Ref<zxing::Reader> reader;
    reader.reset(new zxing::MultiFormatReader);
    //
    zxing::Ref<zxing::Binarizer> binarizer(new zxing::GlobalHistogramBinarizer(source));
    zxing::Ref<zxing::BinaryBitmap> bitmap(new zxing::BinaryBitmap(binarizer));
    auto result = reader->decode(bitmap, zxing::DecodeHints(zxing::DecodeHints::DATA_MATRIX_HINT));
    strResult = result->getText()->getText();
    return true;
}

}
}
