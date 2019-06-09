/* ************************************************************************
FILE   : SPLIT.C
PURPOSE: Breaks up the files stored into a large file created by the
                JOIN.C routines.
HISTORY:
    27 Jun 00   JPT @1
    Updated to support GZIP compression.
    06 Apr 2005	--Zhan Zhuoqun @2
	- fix bug for file open
	17 Nov 2005	-- wenhong @3
	- Added version control since head file changed.
	19 Oct. 2006 -- Gu Wen @3
	-- Replace Windows fprintf with WB_fprintf.
	 06 October 2007       Guo Jianhua @4
    -- global var replace, use 'STDERR' replace 'stderr'in order to fix bug for WB_fprintf().
	27 Nov 2008 -- LBP @5
	- Added for double close protection
	07 Jun 2009 -- LBP @6
	- Fix bug on forget to close file
*/
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <string.h>
#include <dos.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <exception>

//#define _WINDOWS
#include "zlib.h"
#include "DataStruct.h"

#ifndef GZ_SUFFIX
#  define GZ_SUFFIX ".gz"
#endif
#define SUFFIX_LEN (sizeof(GZ_SUFFIX)-1)

#define BUFLEN      16384   // @1
#define MAX_NAME_LEN 1024

/* ===========================================================================
 * Display error message and exit @1
 */
static void error(const char *msg)
{
    throw std::exception ( msg );
}

/* ===========================================================================
 * Uncompress input to output then close both files. @1
 */
short gz_uncompress(gzFile in, FILE   *out)
//    gzFile in;
//    FILE   *out;
{
    char *buf;
    int len;
    int err;

    if ((buf = (char*)malloc (BUFLEN)) == NULL)
    {
        throw std::exception("Insufficient Memory for read buffer");
        return 3;
    }

    for (;;) {
        len = gzread(in, buf, BUFLEN);
        if (len < 0)
        {
            error (gzerror(in, &err));
            free (buf);
            return 2;  //@5
        }
        if (len == 0) break;

        if (len <= BUFLEN && ((int)fwrite(buf, 1, (unsigned)len, out) != len)) //@swq for kw, add len<=BUFLEN because overflow when len is longer.
        {
            error("failed fwrite");
            free (buf);
            return 2;  //@5
        }
    }
    if (fclose(out))
    {
        error("failed fclose");
        free (buf);
        return 2;  //@5
    }

    free (buf);

    if (gzclose(in) != Z_OK)
    {
        error("failed gzclose");
        return 3;
    }

    return 0;
}


/* ===========================================================================
 * Uncompress the given file and remove the original. @1
 */
static short file_uncompress(const char *file, char *outfile)
{
    FILE  *typeCheck;
    FILE  *out;
    gzFile in;
    unsigned char
        buf[2];

    if ((typeCheck = fopen (file, "rb")) != NULL)
    {
        if (fread (&buf,2,1, typeCheck) > 0)
        {
            if ((buf[0]!=0x1f) || (buf[1]!=0x8B))
            {
                rename(file, outfile);
                fclose(typeCheck);
                return 0;
            }
        }
        fclose (typeCheck);
    }
    else
        return 1;

    in = gzopen(file, "rb");
    if (in == NULL) {
        String msg = String ("Can't gzopen " ) + file;
        throw std::exception ( msg.c_str() );
        return 1;
    }
    out = fopen(outfile, "wb");
    if (out == NULL) {
        perror(outfile);
        return 1;
    }

    if(2==gz_uncompress(in, out))  //@5
	{
		fclose(out);	//@2
	}

    return 0;
}

/* ========================================================================
PURPOSE:
*/
static int readHeader (stJAndSHeader *ptrHeader, FILE *fileHdl)
{
    char fileID[MAX_FILE_ID];
    if (fread(fileID, MAX_FILE_ID, 1, fileHdl) == 1)
    {
    	if (strcmp(fileID, "JOIN_V1.0")==0) //@3
    	{
    		fread(ptrHeader, sizeof(stJAndSHeader), 1, fileHdl);
    	}	
    	else if (strcmp(fileID, "JOIN_V2.0")==0) //@3	
            fread(ptrHeader, sizeof(stJAndSHeader), 1, fileHdl);
        else //@3
        {
            String msg = String ( "Unsupported zip type!! fileID: " ) + fileID;
        	throw std::exception ( msg.c_str() );
        }	
    }    
        
    return errno;
}

#ifdef DEBUG_MODE
/* ========================================================================
PURPOSE:
*/
static printHeader (FILE *fHandle, stJAndSHeader *ptrHeader)
{
    int i;
    long
        totalSize = 0;
        
    for (i=0; i<ptrHeader->numFilesJoined; i++)
    {
        WB_fprintf( WB_STDERR, "[%3d] %13s %5ld\n", i,
            ptrHeader->fileName[i], ptrHeader->fileLen[i]);
        totalSize += ptrHeader->fileLen[i];
    }
    WB_fprintf( WB_STDERR, "Total File size = %ld\n", totalSize);
}
#endif

/* ========================================================================
PURPOSE:
*/
static int extractFiles (const char *destDir, stJAndSHeader *ptrHeader, FILE *fileHdl)
{
    char destFileName[260],//@swq for kw, from _MAX_DIR to 260, because of line 247
        *ptrFileBuf = NULL;
    FILE *destHdl = NULL;
                
    // Extract each file separately
    for (int i = 0; i < ptrHeader->numFilesJoined; ++ i )
    {
        // Get memory to store the new file contents
        if ((ptrFileBuf = (char*)malloc(ptrHeader->fileLen[i])) != NULL)
        {
            if (fread(ptrFileBuf, ptrHeader->fileLen[i], 1, fileHdl) == 1)
            {
                // Generate the new file name                
                sprintf(destFileName, "%s%s", destDir, ptrHeader->fileName[i]);

                if ((destHdl = fopen(destFileName, "wb")) != NULL)
                {
                    if ( fwrite ( ptrFileBuf, ptrHeader->fileLen[i], 1, destHdl ) != 1 )
                    {
                        if ( EZERO == errno )
                            errno = ENOSPC;

						free(ptrFileBuf);
						fclose(destHdl);

                        String msg = String("Fail Write ") + destFileName;
                        throw std::exception ( msg.c_str() );
                        break;
                    }
                    fclose (destHdl);
                }
            }
                    
            free (ptrFileBuf);
        }
        else
        {
            throw std::exception("Insufficient Memory");
            errno = ENOMEM;
            break;
        }
    }
    
    return errno;
}    

/* ========================================================================
PURPOSE :
*/
int splitFiles (const char *sourceFilePath, const char *destDir)
{
    stJAndSHeader *splitHdr = NULL;
    FILE *fileHdl = NULL;        
    char splitName[_MAX_DIR];
    short nErr = EZERO;

    sprintf ( splitName, "%s%s", sourceFilePath, ".CAT" );

    if ((nErr = file_uncompress( sourceFilePath, splitName)) != 0) // @1
    {
        String msg = String("Failed uncompress from ") + sourceFilePath + " to "+ splitName;
        throw std::exception ( msg.c_str() );
        return nErr;
    }
        
    // Clear any outstanding errors
    errno = EZERO;

    // Get memory for the split header structure
    if ((splitHdr = (stJAndSHeader*)malloc(sizeof(stJAndSHeader))) != NULL)
    {
        // Open the join file...
        if ((fileHdl = fopen (splitName, "rb")) != NULL)
        {
            // Populate the header from the join file
            if (readHeader (splitHdr, fileHdl) == 0)
            {
                String strDestDir ( destDir );
                if ( strDestDir.back() != '\\' && strDestDir.back() != '/' )
                    strDestDir.push_back ( '/' );
                // ... and split out all the files.
                extractFiles ( strDestDir.c_str(), splitHdr, fileHdl );
            }
			fclose(fileHdl);
        }
        free (splitHdr);
    }
    else
        throw std::exception ("Insufficient memory");
    
    _unlink (splitName);

    // For debugging only
    #ifdef DEBUG_MODE
    printHeader (stdout,splitHdr);
    #endif

    return errno;
}

#ifdef TEST_MODE
/* ========================================================================
PURPOSE : For functional test verification only.
*/
void main (int argc, char **argv)
{
    char
        extName[] = {".WB"},
        szPath[128],
        szDrive[128],
        szDir[128],
        szFname[128],
        szExt[128];
    int
        error = 0;

    // Handler user input
    if (argc < 2)
        WB_fprintf( WB_STDERR, "%s <BASENAME>\n", argv[0]);
    else
    {
        printf ("Splitting Files From %s%s\n", argv[1], extName);
        strcpy (szPath, argv[1]);
        _splitpath (szPath, szDrive, szDir, szFname, szExt);
        if ((error = splitFiles (szDir, szFname, extName, szDir,
            szFname, 1)) != 0)
        {
            WB_fprintf( WB_STDERR, "Error %d\n", error);
        }
    }
}
#endif
