#define DLLEXPORT
#include "VisionAPI.h"

#include "TimeLog.h"
#include "logcase.h"
#include "boost/filesystem.hpp"
#include "RecordManager.h"
#include "Config.h"
#include "CalcUtils.h"
#include "Log.h"

namespace AOI
{
namespace Vision
{

struct BaseVisionHandler::Impl
{
    cv::Mat mat;
};

BaseVisionHandler::BaseVisionHandler()
{
}

BaseVisionHandler::~BaseVisionHandler() = default;

FilterHandler::FilterHandler(PR_FILTER_TYPE enType):_enType(enType)
{
}

void FilterHandler::run()
{
}

}
}