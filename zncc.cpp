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

int	read_png_grey_and_shrink(const char * const filepath, BYTE * * image, SIZE *size, const int reserve_size, const int shrink_factor){
	std::vector<unsigned char> temp;
	unsigned int temp_w, temp_h, i, j;
	unsigned error = lodepng::decode(temp, temp_w, temp_h, filepath);
	if (error){
		handle_lodepng_error(error); 
		return -1;
	}
	size->cx = temp_w / shrink_factor;
	size->cy = temp_h / shrink_factor;
	int allocation_size = size->cx * size->cy + 2*reserve_size;
	*image = (BYTE *)(malloc(allocation_size));
	memset(*image, 0, allocation_size);
	*image = (*image) + reserve_size;
	for (i = 0; i < temp_h; i+=shrink_factor){
		for (j = 0; j < temp_w; j+=shrink_factor){
			(*image)[(i/shrink_factor)*size->cx + (j/shrink_factor)] = BYTE(	0.2126*temp[4 * (i*temp_w + j)    ] + \
																				0.7152*temp[4 * (i*temp_w + j) + 1] + \
																				0.0722*temp[4 * (i*temp_w + j) + 2] );
		}
	}
	return 0;
}

void handle_lodepng_error(int error){
	if (error) {
		char s[128];
		sprintf_s(s, 128, "%d\n%s", error, lodepng_error_text(error));
		std::cout<<"Error: "<<s<<std::endl;
	}
}


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
	int image_width, image_height, window_size, disparity_width, minimum_disparity, maximum_disparity, disparity_range;
	int x, y, i, j, d, gap;
	float zncc;
	float mean_l = 0, std_l = 0, mean_r = 0, std_r = 0, N = 0;
	float new_zncc = 0.0;
	long long pixel_left, pixel_right;
	
	unpack_zncc_worker_args(s, x_begin, x_end, y_begin, y_end, &image_left, &image_right, &image_width, &image_height, &window_size, &minimum_disparity, &maximum_disparity, &disparity_image);
	gap = (window_size - 1) / 2;
	disparity_width = image_width - 2 * gap;
	disparity_range = (maximum_disparity-minimum_disparity)	;
	//printf("mind=%d, maxd%d\n", minimum_disparity, maximum_disparity);
	for (y = y_begin; y < y_end; y++){
		for (x = x_begin; x < x_end; x++){

			// Middle of window
			window_left = image_left + (y*image_width + x);
			window_right = image_right + (y*image_width + x);

			disparity_image[(y - gap)*disparity_width + x - gap] = 0;

			// Find mean and std of left image
			mean_l = 0; std_l = 0;
			for (i = -((window_size - 1) / 2); i <= ((window_size - 1) / 2); i++){
				for (j = -((window_size - 1) / 2); j <= ((window_size - 1) / 2); j++){
					mean_l += window_left[i*image_width + j];
				}
			}
			mean_l /= (window_size*window_size);
			
			for (j = -((window_size - 1) / 2); j <= ((window_size - 1) / 2); j++){
				for (i = -((window_size - 1) / 2); i <= ((window_size - 1) / 2); i++){
					/*
						C99 §6.5.2.1/2:
						The deﬁnition of the subscript operator [] is that
						E1[E2] is identical to (*((E1)+(E2))).
						https://stackoverflow.com/questions/3473675/are-negative-array-indexes-allowed-in-c
					*/
					pixel_left = window_left[i*image_width + j] - mean_l;
					std_l += (pixel_left * pixel_left);
				}
			}
			std_l = sqrt(std_l);
			zncc = 0;
			for (d = minimum_disparity; d < maximum_disparity; d++){
				mean_r = 0;
				for (i = -((window_size - 1) / 2) ; i <= ((window_size - 1) / 2); i++){
					for (j = -((window_size - 1) / 2); j <= ((window_size - 1) / 2); j++){
						mean_r += window_right[i*image_width + (j-d)];
					}
				}
				mean_r /= (window_size*window_size);
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
				if (new_zncc > zncc)
				{
					zncc = new_zncc;
					disparity_image[(y - gap)*disparity_width + (x - gap)] = BYTE(255*abs(float(d))/disparity_range);
				}
			}
		}
	}
	return 0;
}


void get_disparity(const BYTE * const image_left, const BYTE * const image_right, SIZE const image_size, int const window_size, BYTE * * disparity_image, SIZE * disparity_size, const int minimum_disparity, const int maximum_disparity, const int n_threads) {
	
	const int &image_width = image_size.cx;
	const int &image_height = image_size.cy;
	const int gap = (window_size - 1) / 2;
	const int disparity_width = image_width - 2 * gap;
	const int disparity_height = image_height - 2 * gap;
	int xstart, xend, ystart, yend, t;
	
	*disparity_image = new BYTE[disparity_height*disparity_width];

#ifdef _MSC_VER
	std::thread **threads = (std::thread **)(malloc(n_threads*sizeof(std::thread *)));
#else
	pthread_t *threads = new pthread_t[n_threads];
#endif
	zncc_worker_args *s = new zncc_worker_args[n_threads];
	#ifdef LOG_ALGO
	FILE *f = fopen("displog.txt", "w");
	#endif

	xstart = ystart = gap;
	xend = image_width - gap;
	yend = image_height - gap;

	for (t = 0; t < n_threads; t++){
		pack_zncc_worker_args(s+t, xstart, xend, ystart+(t*(yend-ystart)/n_threads), ystart+((t+1)*(yend-ystart)/n_threads), image_left, image_right, image_width, image_height, window_size, minimum_disparity, maximum_disparity, *disparity_image);
#ifdef _MSC_VER
		threads[t] = new std::thread{ zncc_worker, s+t };
#else
		pthread_create(threads+t, NULL, (void *(*)(void *))(zncc_worker), (void *)(s+t));
#endif
	}

	for (t = 0; t < n_threads; t++) {
#ifdef _MSC_VER
		threads[t]->join();
#else
		pthread_join(threads[t], NULL);
#endif
	}


#ifdef LOG_ALGO
	fclose(f);
#endif
	disparity_size->cx = disparity_width;
	disparity_size->cy = disparity_height;
	delete[] s;
#ifdef _MSC_VER
	for (t = 0; t < n_threads; t++){
		delete threads[t];
	}
#endif
	delete[] threads;

}

void cross_check_inplace(BYTE * const buf, const BYTE * const buf_cross_check, const SIZE size, int threshold){
	const int w = size.cx;
	const int h = size.cy;
	int i, j;
	for (i = 0; i < h; i++){
		for (j = 0; j < w; j++){
			if ( abs( buf[i*w + j] - buf_cross_check[i*w + j]) > threshold )
				buf[i*w + j] = 0;
		}
	}
}


void occlusion_fill_inplace(BYTE * image, const SIZE size, int neighbourhood_size){
	const int w = size.cx;
	const int h = size.cy;
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


void exec_project(const int maximum_disparity, const int window_size, const int n_threads, const int threshold, const int skip_depthmapping, const char *img0_fp, const char *img1_fp, const int shrink_factor){
	char d0_filepath[128], d1_filepath[128], cc_filepath[128], of_filepath[128], s[256];
	const char *ip_filepath0 = "im0.png", *ip_filepath1 = "im1.png";
	BYTE * image_left, * image_right, * disparity_image_0, * disparity_image_1;
	SIZE image_size, disparity_size;
	struct timeval t1, t2;
    double elapsedTime;
    int error;

	sprintf_s(d0_filepath, 128, "outputs/d0_%02d_%02d_%02d.png", maximum_disparity, threshold, window_size);
	sprintf_s(d1_filepath, 128, "outputs/d1_%02d_%02d_%02d.png", maximum_disparity, threshold, window_size);
	sprintf_s(cc_filepath, 128, "outputs/cc_%02d_%02d_%02d.png", maximum_disparity, threshold, window_size);
	sprintf_s(of_filepath, 128, "outputs/of_%02d_%02d_%02d.png", maximum_disparity, threshold, window_size);

	
	if (skip_depthmapping) {
		// If file not specified, default to previously made prelim dmaps.
		ip_filepath0 = d0_filepath;
		ip_filepath1 = d1_filepath;
	} 

	// If filepaths(s) specified, override any previous setting.
	ip_filepath0 = img0_fp?img0_fp:ip_filepath0;
	ip_filepath1 = img1_fp?img1_fp:ip_filepath1;
    
    gettimeofday(&t1, NULL); // <-- Starting time
	if (read_png_grey_and_shrink(ip_filepath0, &image_left, &image_size, maximum_disparity, shrink_factor) || read_png_grey_and_shrink(ip_filepath1, &image_right, &image_size, maximum_disparity, shrink_factor)) {
		std::cout << "Unable to read at least 1 input image: " << ip_filepath0 << ", "<< ip_filepath1 << std::endl;
		return;
	}

	if(skip_depthmapping)
	{
		disparity_image_0 = image_left;
		disparity_image_1 = image_right;
		disparity_size = image_size;
	} 
	else
	{
		std::cout<< "Depthmapping image 0..." << std::endl;
		get_disparity(image_left, image_right, image_size, window_size, &disparity_image_0, &disparity_size, 0, maximum_disparity, n_threads);
		error = lodepng::encode(d0_filepath, disparity_image_0, disparity_size.cx, disparity_size.cy, LCT_GREY, 8U);
		handle_lodepng_error(error);

		std::cout<< "Depthmapping image 1..." << std::endl;
		get_disparity(image_right, image_left, image_size, window_size, &disparity_image_1, &disparity_size, -maximum_disparity, 0, n_threads);
		error = lodepng::encode(d1_filepath, disparity_image_1, disparity_size.cx, disparity_size.cy, LCT_GREY, 8U);
		handle_lodepng_error(error);
	}

	std::cout << "Cross checking..." << std::endl;
	cross_check_inplace(disparity_image_0, disparity_image_1, disparity_size, threshold);
	error = lodepng::encode(cc_filepath, disparity_image_0, disparity_size.cx, disparity_size.cy, LCT_GREY, 8U);
	handle_lodepng_error(error);

	std::cout << "Occlusion fill..." << std::endl;
	occlusion_fill_inplace(disparity_image_0, disparity_size, 5);
	error = lodepng::encode(of_filepath, disparity_image_0, disparity_size.cx, disparity_size.cy, LCT_GREY, 8U);
	handle_lodepng_error(error);

	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;

	std::cout << "Done" << std::endl;

	sprintf_s(s, sizeof(s), "echo \"maxdisp=%02d; thres=%02d; winsize=%02d; nthreads=%02d; t_exec=%f\n\" >> outputs/log.txt", maximum_disparity, threshold, window_size, n_threads, elapsedTime);
	system(s);

	system("chmod ugo+rx outputs/*.png"); // To make accessible through web

	if(!skip_depthmapping){
		free(disparity_image_0);
		free(disparity_image_1);
	}
	free(image_left-maximum_disparity);
	free(image_right-maximum_disparity);
}
