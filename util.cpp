#include <iostream>
#include <stdio.h>
#include <CL/cl.h>
#include <string.h>
#include "util.h"
#include "clerrmacros.h"

unsigned char * readFile(const char * const filename) {
    FILE *f = fopen(filename, "rt");
    if(!f) return NULL;
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buffer = (unsigned char *)malloc(length + 1);
    if(!buffer) return NULL;
    buffer[length] = '\0';
    fread(buffer, 1, length, f);
    fclose(f);
    return buffer;
}

void clPrintErrorMacro(cl_int enumber){
    char *s = (char *)calloc(102400);
    SVK_ALL_CL_ERRORS 
    puts(s+2);
    free(s);
}
