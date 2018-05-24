#include <iostream>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <sys/time.h>
#include <CL/cl.h>
#include <assert.h>
#include "zncc.h"
#include "util.h"

namespace cl_objects {
	cl_platform_id 		platforms[100];
	cl_device_id 		devices[100];
	cl_uint 			platforms_n = 0;
	cl_uint 			devices_n = 0;
	cl_device_id 		device;
	cl_context			context;
	cl_program			program;
	cl_kernel			kernel_sgrey;
	cl_kernel			kernel_zncc;
	cl_command_queue	queue;
	cl_event			kernel_completion[5];
	cl_mem				im0_raw_buffer, im0_small_buffer, im0_sub_buffer;
	cl_mem				im1_raw_buffer, im1_small_buffer, im1_sub_buffer;
	cl_mem				dmap1, dmap2, cchk, ofill;
	cl_buffer_region	buffer_region;
};

void opencl_init(int platform_number, int device_number){
	using namespace cl_objects;

	BYTE *				zncc_kernel_src;
	char 				buffer[102400];

	CL_CHECK(clGetPlatformIDs(100, platforms, &platforms_n));
	if (platforms_n == 0) {
		puts("No platform. Aborting...");
		abort();
	}
	CL_CHECK(clGetDeviceIDs(platforms[platform_number], CL_DEVICE_TYPE_GPU, 100, devices, &devices_n));
	if (devices_n == 0) {
		puts("No platform. Aborting...");
		abort();
	}
	device = devices[device_number];
	context = CL_CHECK_ERR(clCreateContext(NULL, 1, devices, &pfn_notify, NULL, &_err));
	if((zncc_kernel_src = readFile("zncc.cl")) == NULL){
		puts("Can't load zncc.cl source.");
		abort();
	}
	program = CL_CHECK_ERR(clCreateProgramWithSource(context, 1, (const char **)&zncc_kernel_src, NULL, &_err));
	if (clBuildProgram(program, 1, &device, "", NULL, NULL) != CL_SUCCESS) {
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		fprintf(stderr, "CL Compilation-1 failed:\n%s", buffer);
		abort();
	}
	CL_CHECK(clUnloadCompiler());
	queue = CL_CHECK_ERR(clCreateCommandQueue(context, device, 0, &_err));
	kernel_sgrey = CL_CHECK_ERR(clCreateKernel(program, "shrink_and_grey", &_err));
	kernel_zncc = CL_CHECK_ERR(clCreateKernel(program, "compute_disparity", &_err));
}

void opencl_create_buffers(int inpng_buffer_size, int result_buffer_size, BYTE *png1_buffer, BYTE *png2_buffer, BYTE *blank_buffer){
	using namespace cl_objects;
	im0_raw_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, inpng_buffer_size, png1_buffer, &_err));
	im1_raw_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, inpng_buffer_size, png2_buffer, &_err));
	im0_small_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, result_buffer_size, blank_buffer, &_err));
	im1_small_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, result_buffer_size, blank_buffer, &_err));
	im0_sub_buffer = CL_CHECK_ERR(clCreateSubBuffer(im0_small_buffer, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION,&buffer_region,&_err));
	im1_sub_buffer = CL_CHECK_ERR(clCreateSubBuffer(im1_small_buffer, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION,&buffer_region,&_err));
	dmap1 = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, result_buffer_size, NULL, &_err));
	dmap2 = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, result_buffer_size, NULL, &_err));
	cchk = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, result_buffer_size, NULL, &_err));
	ofill = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, result_buffer_size, NULL, &_err));
}

void opencl_create_common_objects(){

}

int exec_project_gpu	(
							const char * const img0_fp,
							const char * const img1_fp,
							const int maximum_disparity,
							const int window_size,
							const int threshold,
							const int shrink_factor,
							BYTE ** result, SIZE *size ) {
	using namespace cl_objects;

	const int &		reserve_size = maximum_disparity;
	unsigned int 	im0_w, im0_h, im1_w, im1_h, error, num_elements_small;
	int 			maxdisp_offset = 0;
	int 			result_h, result_w;
	int 			allocation_size;
	BYTE *			zncc_kernel_src = NULL;
	std::vector<unsigned char> im0_vector, im1_vector;

	assert(result);
	assert(size);

	opencl_init(0, 0);

	puts("Reading file...");
	// C++ standard requires short circuit evaluation. So this should work.
	if((error=lodepng::decode(im0_vector, im0_w, im0_h, img0_fp)) || (error=lodepng::decode(im1_vector, im1_w, im1_h, img1_fp)))
	{
		handle_lodepng_error(error);
		return -1;
	}
	puts("Done reading...");
	if((im0_w != im1_w) || (im1_h != im1_h)) return -1;
	
	result_h = int(im0_h / shrink_factor);
	result_w = int(im0_w / shrink_factor);
	num_elements_small = result_h * result_w;
	allocation_size = num_elements_small + 2*reserve_size; // For output small image
	BYTE *temp = (BYTE *)(calloc(allocation_size, 1));
	temp = temp + reserve_size;
	buffer_region = cl_buffer_region{size_t(reserve_size), size_t(num_elements_small)};
	opencl_create_buffers(4 * im0_w * im0_h, allocation_size, &im0_vector[0], &im1_vector[0], temp);

	unsigned char 	sf_c = (unsigned char)shrink_factor;
	size_t global_work_size_sgrey[2] = { size_t(result_h), size_t(result_w) };
	size_t global_work_size_zncc[3] =  { size_t(result_h), size_t(result_w), maximum_disparity};
	size_t local_size_zncc[3] = { 1, 1, maximum_disparity};
	
	CL_CHECK(clSetKernelArg(kernel_sgrey, 0, sizeof(im0_raw_buffer), &im0_raw_buffer));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 1, sizeof(im1_raw_buffer), &im1_raw_buffer));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 2, sizeof(im0_sub_buffer), &im0_sub_buffer));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 3, sizeof(im1_sub_buffer), &im1_sub_buffer));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 4, sizeof(sf_c), &sf_c));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 5, sizeof(im0_w), &im0_w));

	// For im0 on im1
	CL_CHECK(clSetKernelArg(kernel_zncc, 0, sizeof(im0_sub_buffer), &im0_sub_buffer));
	CL_CHECK(clSetKernelArg(kernel_zncc, 1, sizeof(im1_sub_buffer), &im1_sub_buffer));
	CL_CHECK(clSetKernelArg(kernel_zncc, 2, sizeof(dmap1), &dmap1));
	CL_CHECK(clSetKernelArg(kernel_zncc, 3, sizeof(result_w), &result_w));
	CL_CHECK(clSetKernelArg(kernel_zncc, 4, sizeof(result_h), &result_h));
	CL_CHECK(clSetKernelArg(kernel_zncc, 5, sizeof(window_size), &window_size));
	CL_CHECK(clSetKernelArg(kernel_zncc, 6, sizeof(maxdisp_offset), &maxdisp_offset));


	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_sgrey, 2, NULL, global_work_size_sgrey, NULL, 0, NULL, &kernel_completion[0]));
	CL_CHECK((clWaitForEvents(1, &kernel_completion[0])));
	CL_CHECK(clReleaseEvent(kernel_completion[0]));
	
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_zncc, 3, NULL, global_work_size_zncc, local_size_zncc, 0, NULL, &kernel_completion[1]));
	CL_CHECK(clWaitForEvents(1, &kernel_completion[1]));
	CL_CHECK(clReleaseEvent(kernel_completion[1]));

	// For im1 on im0
	maxdisp_offset = -64;
	CL_CHECK(clSetKernelArg(kernel_zncc, 0, sizeof(im1_sub_buffer), &im1_sub_buffer));
	CL_CHECK(clSetKernelArg(kernel_zncc, 1, sizeof(im0_sub_buffer), &im0_sub_buffer));
	CL_CHECK(clSetKernelArg(kernel_zncc, 2, sizeof(dmap2), &dmap2));
	CL_CHECK(clSetKernelArg(kernel_zncc, 3, sizeof(result_w), &result_w));
	CL_CHECK(clSetKernelArg(kernel_zncc, 4, sizeof(result_h), &result_h));
	CL_CHECK(clSetKernelArg(kernel_zncc, 5, sizeof(window_size), &window_size));
	CL_CHECK(clSetKernelArg(kernel_zncc, 6, sizeof(maxdisp_offset), &maxdisp_offset));
	
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_zncc, 3, NULL, global_work_size_zncc, local_size_zncc, 0, NULL, &kernel_completion[2]));
	CL_CHECK(clWaitForEvents(1, &kernel_completion[2]));
	CL_CHECK(clReleaseEvent(kernel_completion[2]));


	cl_mem copy_from_buf_cl = dmap2;
	printf("im0_small_buffer==im0_sub_buffer? %d\n", im0_small_buffer==im0_sub_buffer);
	puts("Copying result...");
	size->cy = result_h;
	size->cx = result_w;
	*result = temp;
	CL_CHECK(clEnqueueReadBuffer(	queue,
									copy_from_buf_cl,
									CL_TRUE,
									0 , //i*sizeof(int)
									num_elements_small,
									*result,
									0,
									NULL,
									NULL	));
	

	CL_CHECK(clReleaseMemObject(im0_raw_buffer));
	CL_CHECK(clReleaseMemObject(im1_raw_buffer));
	CL_CHECK(clReleaseMemObject(im0_small_buffer));
	CL_CHECK(clReleaseMemObject(im1_small_buffer));
	CL_CHECK(clReleaseMemObject(dmap1));
	CL_CHECK(clReleaseMemObject(dmap2));
	CL_CHECK(clReleaseKernel(kernel_sgrey));
	CL_CHECK(clReleaseKernel(kernel_zncc));
	CL_CHECK(clReleaseProgram(program));
	CL_CHECK(clReleaseContext(context));

	return 0;
}