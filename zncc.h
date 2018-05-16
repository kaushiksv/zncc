
#include <iostream>
#include <stdio.h>
#include "lodepng.h"

#ifndef ZNCC_H
#define ZNCC_H

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

/* Inline helpers */
inline int point_in_image(point const pt, SIZE const size){
	return (pt.x >= 0 && pt.x < size.cx && pt.y >= 0 && pt.y < size.cy);
}

inline point operator +(point const &a, point const &b){ return point{ a.x + b.x, a.y + b.y }; }

#endif
