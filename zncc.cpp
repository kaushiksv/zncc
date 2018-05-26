#include <iostream>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <sys/time.h>
#include <CL/cl.h>
#include "zncc.h"
#include "util.h"

/*
	Single function to scale down 2 images by 'shrink_factor' and greyscale them from RGBA format.
*/
void shrink_and_grey(const BYTE * i1, const BYTE * i2, BYTE *o1, BYTE *o2, int w, int h, int shrink_factor){
	int small_w, small_h, i, j, allocation_size;
	small_w = w / shrink_factor;
	small_h = h / shrink_factor;
	for (i = 0; i < small_h; i++){
		for (j = 0; j < small_w; j++){
			o1[i*small_w + j] = BYTE(	0.2126*i1[4 * shrink_factor * (i*w + j)    ] + \
										0.7152*i1[4 * shrink_factor * (i*w + j) + 1] + \
										0.0722*i1[4 * shrink_factor * (i*w + j) + 2] );
			o2[i*small_w + j] = BYTE(	0.2126*i2[4 * shrink_factor * (i*w + j)    ] + \
										0.7152*i2[4 * shrink_factor * (i*w + j) + 1] + \
										0.0722*i2[4 * shrink_factor * (i*w + j) + 2] );
		}
	}
}


// Helper function to store all worker agruments in a structure variable.
void pack_zncc_worker_args(zncc_worker_args * const s, int const x_begin, int const x_end, 
		int const y_begin, int const y_end,	const BYTE * const image_left,	const BYTE * const image_right,
		int const image_width, int const image_height, int const window_size, int const minimum_disparity, int const maximum_disparity,
		BYTE * const disparity_image){
	s->x_begin = x_begin;
	s->x_end = x_end;
	s->y_begin = y_begin;
	s->y_end = y_end;
	s->image_left = image_left;
	s->image_right = image_right;
	s->image_width = image_width;
	s->image_height = image_height;
	s->window_size = window_size;
	s->minimum_disparity = minimum_disparity;
	s->maximum_disparity = maximum_disparity;
	s->disparity_image = disparity_image;
}

// Helper function to retrieve all worker agruments from a structure variable.
void unpack_zncc_worker_args( const zncc_worker_args * const s, int  &x_begin, int &x_end,
		int &y_begin, int &y_end, const BYTE * * image_left, const BYTE * * image_right,
		int *image_width, int *image_height, int *window_size, int *minimum_disparity, int *maximum_disparity,
		BYTE * *disparity_image){
	x_begin = s->x_begin;
	x_end = s->x_end;
	y_begin = s->y_begin;
	y_end = s->y_end;
	*image_left = s->image_left;
	*image_right = s->image_right;
	*image_width = s->image_width;
	*image_height = s->image_height;
	*window_size = s->window_size;
	*minimum_disparity = s->minimum_disparity;
	*maximum_disparity = s->maximum_disparity;
	*disparity_image = s->disparity_image;
}


int zncc_worker(zncc_worker_args *s){
	const BYTE *image_left, *image_right;
	const BYTE *window_left, *window_right;
	BYTE *disparity_image;
	int x_begin, x_end, y_begin, y_end;
	int image_width, image_height, window_size, minimum_disparity, maximum_disparity, disparity_range;
	int x, y, i, j, d, gap;
	float zncc;
	float mean_l = 0, std_l = 0, mean_r = 0, std_r = 0, N = 0;
	float new_zncc = 0.0;
	long long pixel_left, pixel_right;
	
	// Retrieve values
	unpack_zncc_worker_args(s, x_begin, x_end, y_begin, y_end, &image_left, &image_right, &image_width, &image_height, &window_size, &minimum_disparity, &maximum_disparity, &disparity_image);
	gap = (window_size - 1) / 2;
	disparity_range = (maximum_disparity-minimum_disparity)	;

	for (y = y_begin; y < y_end; y++){
		for (x = x_begin; x < x_end; x++){

			/* 
				Middle of window (same as x, y)
				From here, we index the buffer (with negative values as well).
				This is valid in C, and makes the code cleaner to understand.

				https://stackoverflow.com/questions/3473675/are-negative-array-indexes-allowed-in-c
					C99 §6.5.2.1/2:
					The deﬁnition of the subscript operator [] is that
					E1[E2] is identical to (*((E1)+(E2))).
			*/
			window_left = image_left + (y*image_width + x);
			window_right = image_right + (y*image_width + x);

			// Set disparity value to zero initially
			disparity_image[y*image_width + x] = 0;



			/*	Mean and std of left image don't change with disparity value.
				Removing out of loop improves speed. */

			// Find the mean of left window
			mean_l = 0; std_l = 0;
			for (i = -((window_size - 1) / 2); i <= ((window_size - 1) / 2); i++){
				for (j = -((window_size - 1) / 2); j <= ((window_size - 1) / 2); j++){
					mean_l += window_left[i*image_width + j];
				}
			}
			mean_l /= (window_size*window_size);
			
			// Find std and left window
			for (j = -((window_size - 1) / 2); j <= ((window_size - 1) / 2); j++){
				for (i = -((window_size - 1) / 2); i <= ((window_size - 1) / 2); i++){
					pixel_left = window_left[i*image_width + j] - mean_l;
					std_l += (pixel_left * pixel_left);
				}
			}
			std_l = sqrt(std_l);
			zncc = 0;


			for (d = minimum_disparity; d < maximum_disparity; d++){
				// Right window mean
				mean_r = 0;
				for (i = -((window_size - 1) / 2) ; i <= ((window_size - 1) / 2); i++){
					for (j = -((window_size - 1) / 2); j <= ((window_size - 1) / 2); j++){
						mean_r += window_right[i*image_width + (j-d)];
					}
				}
				mean_r /= (window_size*window_size);

				// Right std and Numerator in single loop.
				N = 0;
				std_r = 0;
				for (i = -((window_size - 1) / 2); i <= ((window_size - 1) / 2); i++){
					for (j = -((window_size - 1) / 2); j <= ((window_size - 1) / 2); j++){
						pixel_left = (window_left[i*image_width + j] - mean_l);
						pixel_right = (window_right[i*image_width + (j-d)] - mean_r);
						N += (pixel_left * pixel_right);
						std_r += (pixel_right * pixel_right);
					}
				}
				std_r = sqrt(std_r);
				new_zncc = (N / (std_l*std_r));

				// If the current value is best, pick it,
				// and also store disparity after normalization
				if (new_zncc > zncc)
				{
					zncc = new_zncc;
					disparity_image[y*image_width + x] = BYTE(255*abs(float(d))/disparity_range);
				}
			}
		}
	}
	return 0;
}


void get_disparity(const BYTE * const image_left, const BYTE * const image_right, const int image_width, const int image_height, int const window_size, BYTE * disparity_image, const int minimum_disparity, const int maximum_disparity, const int n_threads) {
	
	const int gap = (window_size - 1) / 2;
	int xstart, xend, ystart, yend, t;
	
	pthread_t *threads = new pthread_t[n_threads];
	zncc_worker_args *s = new zncc_worker_args[n_threads];

	// Omit the `gap' pixels (half the window) along the borders
	xstart = ystart = gap;
	xend = image_width - gap;
	yend = image_height - gap;

	// Spawn `n_threads' threads. This could be 1 as well.
	for (t = 0; t < n_threads; t++){
		/*
			Pack the arguments needed for worker into a structure variable.

			Split the payload along task along vertical. Each row is
			contiguous block of memory and improves performance over
			splitting the payload task along the horizontal (cache etc.)
		*/
		pack_zncc_worker_args(s+t, xstart, xend, ystart+(t*(yend-ystart)/n_threads), ystart+((t+1)*(yend-ystart)/n_threads), image_left, image_right, image_width, image_height, window_size, minimum_disparity, maximum_disparity, disparity_image);
		pthread_create(threads+t, NULL, (void *(*)(void *))(zncc_worker), (void *)(s+t));
	}

	// Wait for the threads to complete
	for (t = 0; t < n_threads; t++) {
		pthread_join(threads[t], NULL);
	}

	// Clear the buffers.
	delete[] s;
	delete[] threads;

}

void cross_check_inplace(BYTE * const buf, const BYTE * const buf_cross_check, const int w, const int h, int threshold){
	int i, j;
	for (i = 0; i < h; i++){
		for (j = 0; j < w; j++){
			if ( abs( buf[i*w + j] - buf_cross_check[i*w + j]) > threshold )
				buf[i*w + j] = 0;
		}
	}
}


void occlusion_fill_inplace(BYTE * image, const int w, const int h, int neighbourhood_size){
	int i, j, abort;
	register int left, right, bottom, top, x, y, r;
	for (i = 0; i < h; i++)
	{
		for (j = 0; j < w; j++)
		{
			if (image[i*w + j] == 0)
			{
				abort = 0;
				for(r=0; r<neighbourhood_size && !abort; r++)
				{
					left = j-r; right = j+r;
					top = i-r; bottom = i+r;
					if(left<0) left = 0;
					if(right>=w) right = w-1;
					if(top<0) top = 0;
					if(bottom>=h) bottom = h-1;
					for(x=left; x<=right && !abort; x++)
					{
						for(y=top; y<=bottom && !abort; y++)
						{
							if (image[y*w + x])
							{
								image[i*w + j] = image[y*w + x];
								abort = 1;
							}
						}
					}
				}
			}
		}
	}
}


void exec_project_cpu(	const char *img0_arg,
						const char *img1_arg,
						const int maximum_disparity,
						const int window_size,
						const int threshold,
						const int shrink_factor,
						const int neighbourhood_size,
						const int n_threads,
						const int skip_depthmapping ) {

	// Vectors used by lodepng::decode
	std::vector<BYTE>	im0, im1;

	// Pointers to default filepath.
	const char 			*im0_filepath = "im0.png";
	const char 			*im1_filepath = "im1.png";

	// Initialize variables
    unsigned int 		im0_h = 0;
    unsigned int 		im0_w = 0;
    unsigned int 		im1_h = 0;
    unsigned int 		im1_w = 0;

    // Buffers pointer for scaled down greyscale images
	BYTE *				image_left;
	BYTE *				image_right;

	// Preliminiary maps
	BYTE *				disparity_image_0;
	BYTE *				disparity_image_1;
    int 				error;

    // Represents the dimensions of scaled down image, and size in bytes.
    int 				small_w;
    int 				small_h;
	int 				allocation_size;

	// Time measurement
	struct timeval 		t[10];
    double 				elapsedTimes[5];
	memset(t, 0, sizeof(t));

    // Arrays to contain filepaths when writing intermediate images to disk.
	char 				d0_filepath[128]; // Preliminary dmap 1
	char 				d1_filepath[128]; // Preliminary dmap 2
	char 				cc_filepath[128]; // Cross-checked image
	char 				of_filepath[128]; // Occlusion filled image

	// Buffer for sprintf
	char 				s[256];


	/* Regarding intermediate images flag and paths */
	char *	env = getenv("INTIMG");
	int 	output_intermediate_images = (strcmp(env?env:"", "1") == 0);
	sprintf_s(d0_filepath, 128, "outputs/MD%02d_T%02d_W%02d_d0.png", maximum_disparity, threshold, window_size);
	sprintf_s(d1_filepath, 128, "outputs/MD%02d_T%02d_W%02d_d1.png", maximum_disparity, threshold, window_size);
	sprintf_s(cc_filepath, 128, "outputs/MD%02d_T%02d_W%02d_cc.png", maximum_disparity, threshold, window_size);
	if(output_intermediate_images)
		sprintf_s(of_filepath, 128, "outputs/MD%02d_T%02d_W%02d_of.png", maximum_disparity, threshold, window_size);
	else
		sprintf_s(of_filepath, 128, "outputs/depthmap.png");

	
	if (skip_depthmapping) {
		/*	Set default input filepaths when reusing preliminary dmaps
			to perform cross checking and occlusion fill.

			Feature DISABLED. See main.cpp ("&& FALSE")

			Was used for exploring different params.
		*/

		im0_filepath = d0_filepath; // Default to previously made prelim dmaps.
		im1_filepath = d1_filepath;
	} 

	status_update("Reading png files...\n");

	// Use custom filepath if mentioned in command line argument.
	im0_filepath = img0_arg?img0_arg:im0_filepath;
	im1_filepath = img1_arg?img1_arg:im1_filepath;

	/*		Read Input PNG Images 		*/
	// C++ standard requires short circuit evaluation. So this will work.
    if ((error=lodepng::decode(im0, im0_w, im0_h, im0_filepath)) || (error=lodepng::decode(im1, im1_w, im1_h, im1_filepath))){
		std::cout << "Unable to read at least 1 input image: " << im0_filepath << ", "<< im1_filepath << std::endl;
		handle_lodepng_error(error);
		return;
	}
	if(im0_w!=im1_w || im0_h!=im1_h){
		printf("Image sizes mismatch.\n");
		return;
	}

	// Scale down image. Default is 4.
	small_w = im0_w/shrink_factor;
	small_h = im0_h/shrink_factor;


	/*		Allocate memory for small images (including small leading/trailing reserves of size maxdisp each).
			The small leading and trailing buffer space helps avoid if statements when finding ZNCC values

			window_right[.... j-d]
			                    ^   This would make go out of bounds either at beginning or end
			                        (depends if 'd' is +ve or -ve )
	*/
	allocation_size = small_w * small_h + 2*maximum_disparity;
	image_left = ((BYTE *)(calloc(allocation_size))) + maximum_disparity;
	image_right = ((BYTE *)(calloc(allocation_size))) + maximum_disparity;
	if(!image_left || !image_right){
		printf("Memory allocation failed. Aborting....");
		abort();
	}

    /*		Shrink and get greyscale of both images 	*/
    gettimeofday(&t[0], NULL);
	shrink_and_grey(&im0[0], &im1[0], image_left, image_right, im0_w, im0_h, shrink_factor);
	gettimeofday(&t[1], NULL);

	if(skip_depthmapping)
	{
		// If skip_depthmapping==TRUE, then small images are actually dmaps.
		disparity_image_0 = image_left;
		disparity_image_1 = image_right;
		
		/////////////////////////////////////////////////////////////////////////////////////////
		///// 			skip_depthmapping FEATURE IS NOT SUPPORTED NOW.
		/////
		/////  Since `shrink_and_grey` function doesn't (R+G+B)/3, but uses specific weights, it causes
		/////  depthmaps loaded from .png to be slightly altered before the cross-checking and occlusion
		/////  fill. It can be fixed, but ignored for now.
		/////////////////////////////////////////////////////////////////////////////////////////
	} 
	else
	{
		// Multithreaded ZNCC based depth mapping (left on right, and right on left)
		disparity_image_0 = (BYTE *)calloc(small_w * small_h);
		status_update("Computing depthmap 1 of 2...\n");
		gettimeofday(&t[2], NULL);
		get_disparity(image_left, image_right, small_w, small_h, window_size, disparity_image_0, 0, maximum_disparity, n_threads);
		gettimeofday(&t[3], NULL);

		if(output_intermediate_images){
			error = lodepng::encode(d0_filepath, disparity_image_0, small_w, small_h, LCT_GREY, 8U);
			handle_lodepng_error(error);
		}

		disparity_image_1 = (BYTE *)calloc(small_w * small_h);
		status_update("Computing depthmap 2 of 2...\n");
		gettimeofday(&t[4], NULL);
		get_disparity(image_right, image_left, small_w, small_h, window_size, disparity_image_1, -maximum_disparity, 0, n_threads);
		gettimeofday(&t[5], NULL);

		if(output_intermediate_images){
			error = lodepng::encode(d1_filepath, disparity_image_1, small_w, small_h, LCT_GREY, 8U);
			handle_lodepng_error(error);
		}
	}

	status_update("Cross checking...\n");
	gettimeofday(&t[6], NULL);
	cross_check_inplace(disparity_image_0, disparity_image_1, small_w, small_h, threshold);
	gettimeofday(&t[7], NULL);
	error = lodepng::encode(cc_filepath, disparity_image_0, small_w, small_h,  LCT_GREY, 8U);
	handle_lodepng_error(error);

	status_update("Occlusion fill...\n");
	gettimeofday(&t[8], NULL);
	occlusion_fill_inplace(disparity_image_0, small_w, small_h, neighbourhood_size);
	gettimeofday(&t[9], NULL);
	error = lodepng::encode(of_filepath, disparity_image_0, small_w, small_h, LCT_GREY, 8U);
	handle_lodepng_error(error);

	// Calculate time taken for each step
	calc_elapsed_times(t, elapsedTimes, 10);

	status_update("Done\n");

	// Show timings on screen, timings append to file
	sprintf_s(s, sizeof(s), "echo \"$(date) :: maxdisp = %02d;  thres = %02d;  winsize = %02d;  nhood = %02d, nthreads = %02d;  t_sg = %0.4lf ms;  t_d0 = %0.4lf ms;  t_d1 = %0.4lf ms;  t_cc = %0.4lf ms;  t_of = %0.4lf ms\n\"",
				maximum_disparity, threshold, window_size, neighbourhood_size, n_threads, elapsedTimes[0], elapsedTimes[1], elapsedTimes[2], elapsedTimes[3], elapsedTimes[4]);
	system(s);
	sprintf_s(s + strlen(s), sizeof(s), " >> \"%s\"", LOGFILE);
	system(s);

	// For web-access on server. 
	// system("chmod ugo+rx outputs/*.png");

	if(!skip_depthmapping){
		// These are not allocated as we would
		// be reusing image_left and image_right.
		free(disparity_image_0);
		free(disparity_image_1);
	}
	free(image_left-maximum_disparity);
	free(image_right-maximum_disparity);
}
