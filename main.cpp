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
#include "util.h"

#define EXEC_PROJ_ARGS_COMMON	args_info.image_0_given?args_info.image_0_arg:NULL, \
								args_info.image_1_given?args_info.image_1_arg:NULL, \
								args_info.maximum_disparity_arg,                    \
								args_info.window_size_arg,                          \
								args_info.threshold_arg,                            \
								args_info.shrink_by_arg, 							\
								args_info.neighbourhood_size_arg

#define EXEC_PROJ_ARGS_CPUONLY	args_info.nthreads_arg,                             \
								(args_info.skip_depthmapping_flag && FALSE)

#define EXEC_PROJ_ARGS_GPUONLY	args_info.platform_number_arg,						\
								args_info.device_number_arg

int main(int argc, char *argv[]){

	using namespace std;

	struct gengetopt_args_info args_info;
	
	// See "./zncc --help"
	// Also fills default values.
	cmdline_parser(argc, argv, &args_info);
	
	if (!in_range(args_info.maximum_disparity_arg, 0, \
			args_info.use_gpu_given?GPU_MAX_MAX_DISP:CPU_MAX_MAX_DISP)){
		cout << "Maximum disparity out of range." << endl;
		return 0;
	}
	if (!in_range(args_info.threshold_arg, 0, MAX_THRESHOLD)){
		cout << "Threshold out of range." << endl;
		return 0;
	}
	if (!in_range(args_info.window_size_arg, 3, \
			args_info.use_gpu_given?GPU_MAX_WIN_SIZE:CPU_MAX_WIN_SIZE)){
		cout << "Window size out of range" << endl;
		return 0;
	}
	if(args_info.window_size_arg%2 == 0){
		cout << "Window size must be odd." << endl;
		return 0;
	}
	if (!in_range(args_info.nthreads_arg, 1, 35)){
		cout << "Number of threads out of range" << endl;
		return 0;
	}

	::update_status_b = args_info.show_status_flag;

	if(args_info.use_gpu_flag){
		exec_project_gpu( EXEC_PROJ_ARGS_COMMON, EXEC_PROJ_ARGS_GPUONLY );
	} else {
		exec_project_cpu( EXEC_PROJ_ARGS_COMMON, EXEC_PROJ_ARGS_CPUONLY );
	}

	return 0;
}
