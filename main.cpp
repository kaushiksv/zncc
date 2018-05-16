#include <iostream>
#include <stdio.h>
#include <memory.h>
#include "lodepng.h"
#include "zncc.h"
#include <limits.h>
#include <stdlib.h>
#include <pthread.h>
#include <CL/cl.h>
#include "cmdline.h"
#include "zncc.h"

#define in_range(x, a, b) (x>=a && x<=b)
#define MAX_MAX_DISP 230
#define MAX_THRESHOLD 50
#define MAX_WIN_SIZE 100

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
	fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

int main(int argc, char *argv[]){
	using namespace std;
	struct gengetopt_args_info args_info;
	cl_platform_id platforms[100];
	cl_device_id devices[100];
	cl_uint platforms_n = 0, devices_n = 0;
	
	// Also fills default values. See cmdline.h or "./zncc --help"
	cmdline_parser(argc, argv, &args_info);
	
	if (!in_range(args_info.maximum_disparity_arg, 0, MAX_MAX_DISP)){
		cout << "Maximum disparity out of range." << endl;
		return 0;
	}
	if (!in_range(args_info.threshold_arg, 0, MAX_THRESHOLD)){
		cout << "Threshold out of range." << endl;
		return 0;
	}
	if (!in_range(args_info.window_size_arg, 3, MAX_WIN_SIZE)){
		cout << "Window size out of range" << endl;
		return 0;
	}
	if (!in_range(args_info.nthreads_arg, 1, 35)){
		cout << "Number of threads out of range" << endl;
		return 0;
	}

	if(args_info.use_gpu_given){
		CL_CHECK(clGetPlatformIDs(100, platforms, &platforms_n));
		if (platforms_n == 0) return 1;
		CL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 100, devices, &devices_n));
		if (devices_n == 0) return 1;
		cl_context context;
		context = CL_CHECK_ERR(clCreateContext(NULL, 1, devices, &pfn_notify, NULL, &_err));
		BYTE *img;
		SIZE size;
		read_png_grey_and_shrink_gpu(devices, context, "im0.png", &img, &size, args_info.maximum_disparity_arg, args_info.shrink_by_arg);
		lodepng::encode("outputs/gpu_shrinked.png", img, size.cx, size.cy, LCT_GREY, 8U);
		free(img-args_info.maximum_disparity_arg);
		return 0;
	} else {

	exec_project(	args_info.maximum_disparity_arg,
					args_info.window_size_arg,
					args_info.nthreads_arg,
					args_info.threshold_arg,
					args_info.skip_depthmapping_given,
					args_info.image_0_given?args_info.image_0_arg:NULL,
					args_info.image_0_given?args_info.image_1_arg:NULL,
					args_info.shrink_by_arg );

	}

	return 0;
}
