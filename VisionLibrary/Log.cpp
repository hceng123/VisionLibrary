#include "Log.h"
#include <fstream>
#include "StopWatch.h"
#include "FileUtils.h"

namespace AOI
{
namespace Vision
{

Log::Log()
{
}

/*static*/ Log *Log::GetInstance()
{
    static Log log;
    return &log;
}

void Log::SetLogPath(const String &strLogPath)
{
    _strLogPath = strLogPath;
}

void Log::Write(const String &strMsg, const char *filename, int line)
{
    auto strParent = FileUtils::GetParentPathname( _strLogPath );
    if (!FileUtils::Exists(strParent))
        FileUtils::MakeDirectory(strParent);

    std::ofstream file( _strLogPath, std::ofstream::out | std::ofstream::app );
    if (!file.is_open())
        return;
    String strLocalMsg = strMsg;
    if ( strLocalMsg.length() < STANDARD_MSG_LENGTH )
        strLocalMsg.resize ( STANDARD_MSG_LENGTH, ' ');

    file << strLocalMsg << "\t" << CStopWatch::GetLocalTimeStr() << "\t" << filename << "\t" << line << std::endl;
    file.close();
}

}
}