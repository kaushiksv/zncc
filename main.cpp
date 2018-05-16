#include <iostream>
#include <stdio.h>
#include <memory.h>
#include "lodepng.h"
#include "zncc.h"
#include <limits.h>
#include <stdlib.h>
#include <pthread.h>
#include "cmdline.h"

#define in_range(x, a, b) (x>=a && x<=b)
#define MAX_MAX_DISP 230
#define MAX_THRESHOLD 50
#define MAX_WIN_SIZE 100

int main(int argc, char *argv[]){
	using namespace std;
	struct gengetopt_args_info args_info;

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

	exec_project(	args_info.maximum_disparity_arg,
					args_info.window_size_arg,
					args_info.nthreads_arg,
					args_info.threshold_arg,
					args_info.skip_depthmapping_given,
					args_info.image_0_given?args_info.image_0_arg:NULL,
					args_info.image_0_given?args_info.image_1_arg:NULL,
					args_info.shrink_by_arg );

	return 0;
}
