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
	`disparity_offset` is used to deal with negative disparity.
	global_work_size : [ img_h x img_w x maximum_disparity ]
	local_work_size  : [   1   x   1   x maximum_disparity ]
	One work-group deals with specific pixel (x,y), for all possible d values.
	So, MAXDISP value is restricted to max supported work-group size along 2nd dim (0..2)
	However, this limitation is assumed ok for now.
	For more info about the system configuration used, see documentation.
*/
__kernel void compute_disparity(
					__global const uchar * const image_left,
					__global const uchar * const image_right,	
					__global unsigned char *disparity_image,
					const int image_width,
					const int image_height,
					int window_size,
					const int disparity_offset){

	const int y = get_global_id(0);
	const int x = get_global_id(1);
	const int d = get_global_id(2);
	const int half_win = ((window_size) - 1) / 2;

	if(x<half_win || y<half_win || x>(image_width-half_win) || y>(image_height-half_win)) return;

	const int xy_addr = y*image_width + x;
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
		for(i = y - half_win; i < y + half_win; i++){
			for(j = x - half_win; j < x + half_win; j++){
				mean_l += window_left[i*image_width + j];
			}
		}
		mean_l = mean_l/(window_size*window_size);
		for(i = y - half_win; i < y + half_win; i++){
			for(j = x - half_win; j < x + half_win; j++){
				pixel_left = window_left[i*image_width + j] - mean_l;
				std_l += pixel_left * pixel_left;
			}
		}
		std_l = native_sqrt(std_l);
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	// Beyond this point, `__local float` variables `mean_l` and `std_l`
	// should have valid values and be accessible across all work items in this work group.

	for (i = -((window_size - 1) / 2) ; i <= ((window_size - 1) / 2); i++){
		for (j = -((window_size - 1) / 2); j <= ((window_size - 1) / 2); j++){
			mean_r += window_right[i*image_width + (j - (d + disparity_offset))];
		}
	}
	mean_r /= (window_size*window_size);
	for (i = -((window_size - 1) / 2); i <= ((window_size - 1) / 2); i++){
		for (j = -((window_size - 1) / 2); j <= ((window_size - 1) / 2); j++){
			pixel_left = (window_left[i*image_width + j] - mean_l);
			pixel_right = (window_right[i*image_width + (j - (d + disparity_offset))] - mean_r);
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
				best_disparity = i + disparity_offset;
			}
		}
		// Copy disparity at (x,y) to global memory.
		disparity_image[y*image_width + x] = (unsigned char)(255*(((float)(abs(best_disparity)))/maximum_disparity));
	}
}
