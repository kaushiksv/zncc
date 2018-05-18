#include <iostream>
#include <stdio.h>

#ifndef SVK_UTIL_H
#define SVK_UTIL_H

unsigned char * 	readFile	(const char * const filename);
inline void *		calloc		(size_t s){
	return calloc(s, 1);
}

void clPrintErrorMacro(cl_int enumber);

#endif