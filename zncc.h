
#include <iostream>
#include <stdio.h>
#include <CL/cl.h>
#include "lodepng.h"
#include "util.h"

#ifndef ZNCC_H
#define ZNCC_H

#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "CL_CHECK; OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     clPrintErrorMacro(_err); \
     abort();                                                                   \
   } while (0)

#define CL_CHECK_ERR(_expr)                                                     \
   ({                                                                           \
     cl_int _err = CL_INVALID_VALUE;                                            \
     typeof(_expr) _ret = _expr;                                                \
     if (_err != CL_SUCCESS) {                                                  \
       fprintf(stderr, "CL_CHECK_ERR; OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       clPrintErrorMacro(_err); \
       abort();                                                                 \
     }                                                                          \
     _ret;                                                                      \
   })

#define sprintf_s snprintf

typedef unsigned char BYTE;
typedef struct SIZE{long cx; long cy;} *PSIZE;

struct point { int x; int y; };
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
void	get_disparity			(const BYTE * const image_left, const BYTE * const image_right, SIZE const image_size, int const window_size, BYTE *disparity_image, SIZE *disparity_size, const int minimum_disparity, const int maximum_disparity, const int n_threads = 3);
void	cross_check_inplace		(BYTE * const buf, const BYTE * const buf_cross_check, const SIZE size, int threshold);
//void	post_process_inplace	(BYTE * const buf, const BYTE * const buf_cross_check, const SIZE disparity_size, int threshold = 8);
void	occlusion_fill_inplace	(BYTE * buf, const SIZE size, int neighbourhood_size);
void	exec_project			(const int maximum_disparity = 65, const int window_size = 9, const int n_threads = 3, const int threshold = 8, const int reuse_depthmap_output_files = 0, const char *img0_fp = NULL, const char *img1_fp = NULL, const int shrink_factor = 4);

/*	I/O functions						*/
int		read_png_grey_and_shrink (const char * const filename, BYTE * * image, SIZE * size, int reserve_size = 0 );
void	handle_lodepng_error	(int error);


/*	OpenCL helpers */
int read_png_grey_and_shrink_gpu( cl_device_id device,
                  cl_context context,
                  const char * const filepath1,
                  const char * const filepath2,
                  const int maximum_disparity,
                  const int shrink_factor,
                  BYTE * * result,
                  SIZE *size  );

/* Inline helpers */
inline int point_in_image(point const pt, SIZE const size){
	return (pt.x >= 0 && pt.x < size.cx && pt.y >= 0 && pt.y < size.cy);
}

inline point operator +(point const &a, point const &b){ return point{ a.x + b.x, a.y + b.y }; }

#endif
