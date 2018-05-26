#define DISPARITY_DIMENSION		2 
#define MAXDISP					64

typedef __global const unsigned char	uchar_gc;
typedef __global unsigned char 			uchar_g;

/*
	  shrink_and_grey:
	Kernel to quick shrink two images pointed by buffers `i1` and `i2`,
	by `shrink_factor`. The last argument is width of unshrinked image.
*/
__kernel void shrink_and_grey(	uchar_gc	*i1,
								uchar_gc	*i2,
								uchar_g		*o1,
								uchar_g 	*o2,
								uchar		shrink_factor,
								uint 		w ) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	int ws = w/shrink_factor;
	ulong addr = ((unsigned long)(i*w + j)) * 4 * shrink_factor;
	o1[i*ws + j] = (unsigned char)(0.2126*i1[addr] + 0.7152*i1[addr + 1] + 0.0722*i1[addr + 2]);
	o2[i*ws + j] = (unsigned char)(0.2126*i2[addr] + 0.7152*i2[addr + 1] + 0.0722*i2[addr + 2]);
}


/*
	  compute_disparity:
	Kernel to compute best disparity value at (x,y) by computing ZNCC values.
	`DISPARITY_OFFSET` is used to deal with negative disparity.
	global_work_size : [ img_h x img_w x maximum_disparity ]
	local_work_size  : [   1   x   1   x maximum_disparity ]
	One work-group deals with specific pixel (x,y), for all possible d values.
	So, MAXDISP value is restricted to max supported work-group size along 2nd dim (0..2)
	However, this limitation is assumed ok for now.
	For more info about the system configuration used, see documentation.
*/

#ifndef IMAGE_WIDTH
#define IMAGE_WIDTH image_width
#endif
#ifndef IMAGE_HEIGHT
#define IMAGE_HEIGHT image_height
#endif
#ifndef WINDOW_SIZE
#define WINDOW_SIZE window_size
#endif
#ifndef DISPARITY_OFFSET
#define DISPARITY_OFFSET disparity_offset
#endif

__kernel void compute_disparity(
					__global const uchar * const image_left,
					__global const uchar * const image_right,	
					__global unsigned char *disparity_image,
					const int image_width,
					const int image_height,
					const int window_size,
					const int disparity_offset){

	const int y = get_global_id(0);
	const int x = get_global_id(1);
	const int d = get_global_id(2);
	const int half_win = ((WINDOW_SIZE) - 1) / 2;

	if(x<half_win || y<half_win || x>(IMAGE_WIDTH-half_win) || y>(IMAGE_HEIGHT-half_win)) return;

	const int xy_addr = y*IMAGE_WIDTH + x;
	const int maximum_disparity = get_local_size(DISPARITY_DIMENSION);

	// Consider removing global keyword.
	__global const uchar * const window_left = image_left + xy_addr;
	__global const uchar * const window_right = image_right + xy_addr;

	int i, j, best_disparity = 0;
	float pixel_left, pixel_right, N = 0.0, mean_r = 0.0, std_r = 0.0, best_zncc = 0.0;
	
	/*
		Variables `mean_l` and `std_l` are made `__local` to save space, and speed up.
		In local memory they occupy, 264 Bytes per work-group [(1 + 1 + 64) * 4 = 264]
	*/
	__local float mean_l, std_l, znccs[MAXDISP];
	
	// One time calculation of mean and std of left window.
	if(d == 0){
		mean_l = 0.0; std_l = 0.0;
		for(i = -half_win; i <= half_win; i++){
			for(j = -half_win; j <= half_win; j++){
				mean_l += window_left[i*IMAGE_WIDTH + j];
			}
		}
		mean_l = mean_l/(WINDOW_SIZE*WINDOW_SIZE);
		for(i = -half_win; i <= half_win; i++){
			for(j = -half_win; j <= half_win; j++){
				pixel_left = window_left[i*IMAGE_WIDTH + j] - mean_l;
				std_l += pixel_left * pixel_left;
			}
		}
		std_l = native_sqrt(std_l);
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	// Beyond this point, `__local float` variables `mean_l` and `std_l`
	// should have valid values and be accessible across all work items in this work group.

	for (i = -((WINDOW_SIZE - 1) / 2) ; i <= ((WINDOW_SIZE - 1) / 2); i++){
		for (j = -((WINDOW_SIZE - 1) / 2); j <= ((WINDOW_SIZE - 1) / 2); j++){
			mean_r += window_right[i*IMAGE_WIDTH + (j - (d + DISPARITY_OFFSET))];
		}
	}
	mean_r /= (WINDOW_SIZE*WINDOW_SIZE);
	for (i = -((WINDOW_SIZE - 1) / 2); i <= ((WINDOW_SIZE - 1) / 2); i++){
		for (j = -((WINDOW_SIZE - 1) / 2); j <= ((WINDOW_SIZE - 1) / 2); j++){
			pixel_left = (window_left[i*IMAGE_WIDTH + j] - mean_l);
			pixel_right = (window_right[i*IMAGE_WIDTH + (j - (d + DISPARITY_OFFSET))] - mean_r);
			N += (pixel_left * pixel_right);
			std_r += (pixel_right * pixel_right);
		}
	}
	std_r = native_sqrt(std_r);
	znccs[d] = (N / (std_l*std_r));

	barrier(CLK_LOCAL_MEM_FENCE);

	if(d==0){
		for(i=0; i<maximum_disparity; i++){
			if(znccs[i] > best_zncc){
				best_zncc = znccs[i];
				best_disparity = i + DISPARITY_OFFSET;
			}
		}
		// Copy disparity at (x,y) to global memory.
		disparity_image[y*IMAGE_WIDTH + x] = (unsigned char)(255*(((float)(abs(best_disparity)))/maximum_disparity));
	}
}


__kernel void cross_check_inplace(uchar_g *buf, uchar_gc *buf_cross_check, const int threshold){
	int i = get_global_id(0);
	int j = get_global_id(1);
	int h = get_global_size(0);
	int w = get_global_size(1);
	if ( abs( buf[i*w + j] - buf_cross_check[i*w + j]) > threshold )
		buf[i*w + j] = 0;
}

__kernel void occlusion_fill_inplace(uchar_g *image, const int half_win){
	const int neighbourhood_size = 8;
	int i = get_global_id(0) + half_win;
	int j = get_global_id(1) + half_win;
	int h = get_global_size(0);
	int w = get_global_size(1);
	int left, right, bottom, top, x, y, r;
	if (image[i*w + j] == 0)
	{
		for(r=0; r<neighbourhood_size; r++)
		{
			left = j-r; right = j+r;
			top = i-r; bottom = i+r;
			if(left<0) left = 0;
			if(right>=w) right = w-1;
			if(top<0) top = 0;
			if(bottom>=h) bottom = h-1;
			for(x=left; x<=right; x++)
			{
				for(y=top; y<=bottom; y++)
				{
					if (image[y*w + x])
					{
						image[i*w + j] = image[y*w + x];
						return;
					}
				}
			}
		}
	}
}

