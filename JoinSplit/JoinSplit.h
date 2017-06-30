/* ************************************************************************
FILE    : JoinSplit.h
PURPOSE : Provide functions to put all files in a folder to a single file,
          and also can split the compressed file to a folder.
HISTORY : 20170622 XSG Created.
*/

#ifndef _JOIN_SPLIT_H_
#define _JOIN_SPLIT_H_

int joinDir (const char *sourceDir, const char *extName);
int splitFiles (const char *sourceFilePath, const char *destDir);

#endif /*_JOIN_SPLIT_H_*/




