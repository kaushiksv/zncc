
#include <iostream>
#include <stdio.h>
#include <CL/cl.h>
#include "lodepng.h"
#include "util.h"

#ifndef ZNCC_H
#define ZNCC_H


#define TRUE	1
#define FALSE	0

#define LOGFILE			  "performance_log.txt"

/* Maximum-disparity, Window size, and Threshold configurable by command line arguments */
#define CPU_MAX_MAX_DISP  230
#define CPU_MAX_WIN_SIZE  100
#define GPU_MAX_MAX_DISP  64
#define GPU_MAX_WIN_SIZE  50
#define MAX_THRESHOLD     50

#define sprintf_s snprintf

/*
  The following macros CL_CHECK and CL_CHECK_ERR were originally written by Clifford Wolf
  and available at URL: http://svn.clifford.at/tools/trunk/examples/cldemo.c
    (in public domain)
*/
#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "CL_CHECK; OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     cl_decode_error(_err); \
     abort();                                                                   \
   } while (0)

#define CL_CHECK_ERR(_expr)                                                     \
   ({                                                                           \
     cl_int _err = CL_INVALID_VALUE;                                            \
     typeof(_expr) _ret = _expr;                                                \
     if (_err != CL_SUCCESS) {                                                  \
       fprintf(stderr, "CL_CHECK_ERR; OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       cl_decode_error(_err); \
       abort();                                                                 \
     }                                                                          \
     _ret;                                                                      \
   })

typedef unsigned char BYTE;

struct zncc_worker_args {
	int x_begin, x_end, y_begin, y_end;
	const BYTE *image_left, *image_right;
	int image_width, image_height;
	int window_size;
	BYTE * disparity_image;
	int disparity_width;
	int minimum_disparity;
	int maximum_disparity;
};

/*	Image manipulation functions		*/
int		shrink_and_grey			(	const BYTE * i1,
									const BYTE * i2,
									BYTE **o1,
									BYTE **o2,
									int w,
									int h,
									int shrink_factor,
									int reserve_size
								);

void	get_disparity			(	const BYTE * const image_left,
									const BYTE * const image_right,
									const int image_width,
									const int image_height,
									const int window_size,
									BYTE *disparity_image,
									const int minimum_disparity,
									const int maximum_disparity,
									const int n_threads = 3
								);


void	cross_check_inplace		(	BYTE * const buf,
									const BYTE * const buf_cross_check,
									const int w,
									const int h,
									int threshold
								);

void	occlusion_fill_inplace	(	BYTE * buf,
									const int w,
									const int h,
									int neighbourhood_size
								);

void	exec_project_cpu		(	const char *img0_arg,
									const char *img1_arg,
									const int maximum_disparity = 64,
									const int window_size = 9,
									const int threshold = 8,
									const int shrink_factor = 4,
									const int neighbourhood_size = 8,
									const int n_threads = 3,
									const int skip_depthmapping = 0
								);

void   exec_project_gpu 		(	const char * const img0_arg,
									const char * const img1_arg,
									const int maximum_disparity = 64,
									const int window_size = 9,
									const int threshold = 8,
									const int shrink_factor = 4,
									const int neighbourhood_size = 8,
									const int platform_number = 0,
									const int device_number = 0
								);

#endif
