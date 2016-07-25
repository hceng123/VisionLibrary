#ifndef _CONFIG_H_
#define _CONFIG_H_

#include "BaseType.h"
#include "VisionType.h"

namespace AOI
{
namespace Vision
{

class Config
{
protected:
    Config();

    const String    _strRecordDir       =       ".\\Vision\\Record\\";
    const String    _strLogCaseDir      =       ".\\Vision\\Logcase\\";
    const String    _strRecordPrefix    =       "tmpl.";
    const String    _strRecordLogPrefix =       "tmplDir.";
    PR_DEBUG_MODE   _enDebugMode         =       PR_DEBUG_MODE::DISABLED;
public:
    static Config*          GetInstance();
    String                  getRecordDir() const;
    String                  getLogCaseDir() const;
    String                  getRecordPreFix() const;
    String                  getRecordLogPrefix() const;
    PR_DEBUG_MODE           getDebugMode() const;
    void                    setDebugMode(PR_DEBUG_MODE enDebugMode);
    void load();
};

}
}
#endif