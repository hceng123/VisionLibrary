#ifndef _JOIN_SPLIT_DATA_STRUCT_H_
#define _JOIN_SPLIT_DATA_STRUCT_H_

#include <string>

#define MAX_FILES_JOINED        128
#define MAX_FILENAME_LEN        40
#define JANDS_ID                "JOIN_V1.0"	//
#define MAX_FILE_ID             20          // ID must be less than 20 chars
#define EZERO	                0	/* No error */

using String = std::string;


#pragma pack(push, 1)

typedef struct
{
    char    fileName[MAX_FILES_JOINED][MAX_FILENAME_LEN];   // Name of stored file
    long    fileLen[MAX_FILES_JOINED];                      // Size of stored file
    short   numFilesJoined;                                 // Num of files actually stored
} stJAndSHeader;

#pragma pack(pop)

#endif //_JOIN_SPLIT_DATA_STRUCT_H_