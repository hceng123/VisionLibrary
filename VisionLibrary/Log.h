#ifndef _LOG_H_
#define _LOG_H_

#include "BaseType.h"

namespace AOI
{
namespace Vision
{

#define WriteLog(strMsg)    Log::GetInstance()->Write(strMsg, __FILE__, __LINE__)

class Log
{
protected:
    Log();
    String _strLogPath = ".\\Vision\\log.log";
    const UInt32 STANDARD_MSG_LENGTH = 80;
public:
    static Log *GetInstance();
    void SetLogPath(const String &strLogPath);
    void Write(const String &strMsg, const char *filename, int line);
};

}
}

#endif