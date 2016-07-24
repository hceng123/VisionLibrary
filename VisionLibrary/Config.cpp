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

String Config::getRecordPreFix() const
{
    return _strRecordPrefix;
}

String Config::getRecordLogPrefix() const
{
    return _strRecordLogPrefix;
}

void Config::load()
{
}

}
}