#include "Config.h"

namespace AOI
{
namespace Vision
{

Config::Config()
{
}


/*static*/ Config *Config::GetInstance()
{
    static Config config;
    return &config;
}

String Config::getRecordDir() const
{
    return _strRecordDir;
}

String Config::getLogCaseDir() const
{
    return _strLogCaseDir;
}

String Config::getRecordPreFix() const {
    return _strRecordPrefix;
}

String Config::getRecordExt() const {
    return _strRecordExt;
}

String Config::getRecordLogPrefix() const {
    return _strRecordLogPrefix;
}

String Config::getRecordParamFile() const {
    return _strRecordParamFile;
}

void Config::setDebugMode(PR_DEBUG_MODE enDebugMode) {
    _enDebugMode = enDebugMode;
}

Int16 Config::getDeviceInspChannel() const
{
    return _nDeviceInspChannel;
}

Int32 Config::getRemoveCCMaxComponents()
{
    return _nRemoveCCMaxComponent;
}

String Config::getOcrCharList() const
{
    return _strOcrCharList;
}

Int32 Config::getInspBridgeMinThreshold() const {
    return _nInspBridgeMinThreshold;
}

Int32 Config::getMinPointsToCalibCamera() const {
    return _nMinPointsToCalibCamera;
}

void Config::setEnableGpu(bool bEnable) {
    _bEnableGpu = bEnable; 
}

void Config::load()
{
}

}
}