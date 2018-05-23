// __kernel void getMeanAndStd(__global const uchar *left_image, __global float *mean, __global float *std, int w, int h, int window_size){
// 	int i, j, offset;
// 	float local_mean = 0.0, local_std = 0.0, p;
// 	const int x = get_global_id(0);
// 	const int y = get_global_id(1);
// 	const int d = get_global_id(2)
// 	const int half_win = (window_size-1)/2;
// 	const int xy_addr = y*w + x;
	
// 	if(x<half_win || y<half_win || x>(w-half_win) || y>(h-half_win)) return;
	
// 	for(i = y - half_win; i < y + half_win; i++){
// 		offset = xy_addr + i*w; // To save few additions
// 		for(j = x - half_win; j < x + half_win; j++){
// 			local_mean += left_image[offset + j];
// 		}
// 	}
// 	local_mean = local_mean/(window_size*window_size);
	
// 	for(i = y - half_win; i < y + half_win; i++){
// 		offset = xy_addr + i*w; // To save few additions
// 		for(j = x - half_win; j < x + half_win; j++){
// 			p = left_image[offset + j] - local_mean;
// 			local_std += p*p;
// 		}
// 	}
// 	local_std = sqrt(local_std);
// 	mean[xy_addr] = local_mean;
// 	std[xy_addr] = local_std;
// }

// 0... get_work_dim()-1


typedef __global const unsigned char	uchar_gc;
typedef __global unsigned char 			uchar_g;

__kernel void shrink_and_grey(	__global const unsigned char *i1,
								__global const unsigned char *i2,
								__global unsigned char *o1,
								__global unsigned char *o2,
								unsigned char shrink_factor, // MAKE CHAR
								unsigned int w ){
	//unsigned char shrink_factor = ((unsigned char)(shrink_factor_i));
	int i = get_global_id(0); // 0...503
	int j = get_global_id(1); // 0...734
	int ws = w/shrink_factor;
	unsigned long addr = ((unsigned long)(i*w + j)) * 4 * shrink_factor;
	o1[i*ws + j] = (unsigned char)(0.2126*i1[addr] + 0.7152*i1[addr + 1] + 0.0722*i1[addr + 2]);
	o2[i*ws + j] = (unsigned char)(0.2126*i2[addr] + 0.7152*i2[addr + 1] + 0.0722*i2[addr + 2]);
	//float x = get_global_size(1);// get_global_size(1);
	//float y = get_global_size(0);
	//float ii = ((float)i);
	//float jj = ((float)j);
	//o1[i*ws + j] = 255*sqrt(jj*jj + ii*ii)/sqrt(x*x + y*y);
	//o1[i*ws + j] = 255* sqrt(x*x + y*y); //(((float)(j))/i);
	//(unsigned char)(0.2126*i1[addr] + 0.7152*i1[addr + 1] + 0.0722*i1[addr + 2]);
	//ws = (w/shrink_factor);
	//addr = (i*w + j)*Bpp;
	// 64*(((float)(addr))/5927040); //
	 //(unsigned char)(0.2126*i1[addr] + 0.7152*i1[addr + 1] + 0.0722*i1[addr + 2]);
	//o2[i*ws + j] = (unsigned char)(0.2126*i2[addr] + 0.7152*i2[addr + 1] + 0.0722*i2[addr + 2]);
}


#define DISPARITY_DIMENSION 2 
#define MAXDISP 64
/*
	Kernel to compute ZNCCs at (x,y) and return best disparity value.
	`disparity_offset` is used to deal with negative disparity when calculating
*/

__kernel void compute_disparity(
					__global const uchar * const image_left,
					__global const uchar * const image_right,	
					__global unsigned char *disparity_image,
					const int image_width,
					const int image_height,
					int window_size,
					const int disparity_offset){
					//__global float * znccs_g ){


	const int y = get_global_id(0);
	const int x = get_global_id(1);
	const int d = get_global_id(2);
	const int half_win = ((window_size) - 1) / 2;

	if(x<half_win || y<half_win || x>(image_width-half_win) || y>(image_height-half_win)) return;

	const int xy_addr = y*image_width + x;
	const int maximum_disparity = get_local_size(DISPARITY_DIMENSION);
	__global const uchar * const window_left = image_left + xy_addr;
	__global const uchar * const window_right = image_right + xy_addr;

	int i, j, best_disparity = 0;
	float pixel_left, pixel_right, N = 0.0, mean_r = 0.0, std_r = 0.0, best_zncc = 0.0;
	
	// Shared in work-group. 64*4 + 2*4 = 264 Bytes/work-group in local memory.
	//__global float znccs_g[1024*1024][MAXDISP]; //, mean_l_g[1024*1024], std_l_g[1024*1024];
	__local float mean_l, std_l, znccs[MAXDISP];
//#define znccs (znccs_g+y*image_width + x)
//define mean_l mean_l_g[y*image_width + x]
//define std_l std_l_r[y*image_width + x]
	
	// One time calculation of mean/std of left window.
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
		std_l = sqrt(std_l);
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	// Beyond this point, `__local float` variables `mean_l` and `std_l`
	// should be accessible across all work items in this work group.
	// Work-group size is set to 1x1xMAXDISP by host
	// (One work-group for specific x and y, and, all d values)
	// So, MAXDISP value is restricted to max supported work-group size along 2nd dim (0..2)
	// This limitation is ok for now. This value is 64 for GeForce GTX 1060 3GB.

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
	std_r = sqrt(std_r);
	znccs[d] = (N / (std_l*std_r));

	barrier(CLK_LOCAL_MEM_FENCE);

	if(d==0){
		for(i=0; i<maximum_disparity; i++){
			if(znccs[i] > best_zncc){
				best_zncc = znccs[i];
				best_disparity = i;
			}
		}

		// Copy to global memory.
		disparity_image[y*image_width + x] = (unsigned char)(255*(((float)(abs(best_disparity)))/maximum_disparity));
	}
}


// Methods

// __kernel getLeftMeanAndStds:
// 	Left Window Averages for window at each point in image.

// __kernel calcZncc:
// 	Use Mean and Std to find zncc(x, y, d)
// 	(1) Find right window mean
// 	(2) Find N and std
// 	(3) Store Zncc

// __kernel getDisparity:
// 	2D kernel to find minimum max zncc, and get disparity value.

// Host:
// 	getLeftmeanAndStds(right)
// 	calcZncc(right, left)
// 	getDisparity(rightZncc)

// __kernel crossCheck(__global uchar *a, __global uchar *b, const uchar threshold){
// 	int i = get_global_id(0)
// 	if(abs_diff(a[i], b[i])>threshold)
// 		a[i] = 0;
// }
// __kernel occlusionFill
//		x, y . Increase distance gradually
// 	At each iteration (execution of kernel):
