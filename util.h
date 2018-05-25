#include <iostream>
#include <stdio.h>

#ifndef SVK_UTIL_H
#define SVK_UTIL_H

#define in_range(x, a, b) 		(x>=a && x<=b)

#define VARLOGW(VAR, WIDTH)		printf("%*s : ", checkint(WIDTH), #VAR); std::cout<<VAR; putchar('\n');
#define VARLOG(VAR) 			VARLOGW(VAR, 32)

#define status_update(...)		({										\
									int _ret = 0;						\
									if(::update_status_b){				\
										_ret = printf(__VA_ARGS__);		\
									}									\
									_ret;								\
								})


extern int update_status_b;


/*
  The function pfn_notify was originally written by Clifford Wolf
  and available at URL: http://svn.clifford.at/tools/trunk/examples/cldemo.c
    (Released into public domain)
*/


void 				calc_elapsed_times	(struct timeval *t, double *elapsedTimes, int t_n);
void				cl_decode_error		(cl_int enumber);
void				handle_lodepng_error(int error);
void 				pfn_notify			(const char *errinfo, const void *private_info, size_t cb, void *user_data);
unsigned char * 	read_file			(const char * const filename);

/* Inline functions */

inline void *		calloc		(size_t s){
	return calloc(s, 1);
}

inline int 			checkint	(int i){
	return i;
}

inline int			operator ==	(struct timeval const &t1, struct timeval const &t2){
    return (t1.tv_sec == t2.tv_sec  &&  t1.tv_usec == t2.tv_usec);
}



#endif