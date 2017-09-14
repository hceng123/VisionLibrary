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

using MapRecord = std::map<Int32, RecordPtr>;
using PairRecord = std::pair<Int32, RecordPtr>;
class RecordManager
{
public:
    static RecordManager*   getInstance();
    ~RecordManager();
    VisionStatus            add(RecordPtr pRecord, Int32 &recordID);
    RecordPtr               get(Int32 nRecordId);
    VisionStatus            free(Int32 nRecordId);
    VisionStatus            freeAllRecord();
    VisionStatus            load();
protected:
    RecordManager() {};
    VisionStatus            _loadOneRecord(const String &strFilePath);
    RecordPtr               _createRecordPtr(Int32 recordType);
    Int32                   _generateRecordID();
    String                  _getRecordFilePath(Int32 recordID);

protected:
    MapRecord   _mapRecord;
};

}
}
#endif