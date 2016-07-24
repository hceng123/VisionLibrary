#include "Log.h"
#include <fstream>

namespace AOI
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
    std::ofstream file;
    file.open(_strLogPath);
    if (!file.is_open())
        return;
    file << strMsg << "\t" << filename << "\t" << line << "\n";
    file.close();
}

}