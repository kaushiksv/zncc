#include <iostream>
#include <stdio.h>
#include <CL/cl.h>
#include <string.h>
#include "util.h"
#include "zncc.h"
#include "lodepng.h"
#include "clerrmacros.h"

int update_status_b = 0;

unsigned char * read_file(const char * const filename) {
    FILE *f = fopen(filename, "r");
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

void cl_decode_error(cl_int enumber){
    char *s = (char *)calloc(102400);
    // Defined in clerrmacros.h
    FIND_STRINGIFY_APPEND_ALL_CL_ERRORS 
    puts(s+2);
    free(s);
}

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

void handle_lodepng_error(int error){
    if (error) {
        char s[128];
        sprintf_s(s, 128, "%d\n%s", error, lodepng_error_text(error));
        std::cout<<"Error: "<<s<<std::endl;
    }
}

void calc_elapsed_times (struct timeval *t, double *elapsedTimes, int t_n){
    for(int i=0; i<t_n; i+=2){
        if (t[i] == t[i+1]) {
            elapsedTimes[i>>1] = 0;
        } else{
            elapsedTimes[i>>1] = (t[i+1].tv_sec - t[i].tv_sec) * 1000.0;
            elapsedTimes[i>>1] += (t[i+1].tv_usec - t[i].tv_usec) / 1000.0;
        }
    }
}