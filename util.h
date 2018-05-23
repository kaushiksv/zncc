#include <iostream>
#include <stdio.h>

#ifndef SVK_UTIL_H
#define SVK_UTIL_H

#define VARLOGW(VAR, WIDTH)		printf("%*s : ", checkint(WIDTH), #VAR); std::cout<<VAR; putchar('\n');
#define VARLOG(VAR) 			VARLOGW(VAR, 32)

void				clPrintErrorMacro	(cl_int enumber);
unsigned char * 	readFile			(const char * const filename);


/* Inline functions */

inline int 			checkint	(int i){
	return i;
}

inline void *		calloc		(size_t s){
	return calloc(s, 1);
}

#endif