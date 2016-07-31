#ifndef _RECORD_MANAGER_H_
#define _RECORD_MANAGER_H_

#include "Record.h"
#include "BaseType.h"
#include "VisionStatus.h"
#include <map>

namespace AOI
{
namespace Vision
{

using MapRecord = std::map<Int32, IRecord*>;
using PairRecord = std::pair<Int32, IRecord*>;
class RecordManager
{
public:
    static RecordManager*   getInstance();
    ~RecordManager();
    VisionStatus            add(IRecord *pRecord, Int32 &recordID);    
    IRecord*                get(Int32 nRecordID);
    VisionStatus            free(Int32 nRecordID);
    VisionStatus            freeAllRecord();
    VisionStatus            load();
protected:
    RecordManager() {};
    VisionStatus            _loadOneRecord(const String &strFilePath);
    IRecord*                _createRecordPtr(Int32 recordType);
    Int32                   _generateRecordID();
    String                  _getRecordFilePath(Int32 recordID);

protected:
    MapRecord   _mapRecord;
};

}
}
#endif