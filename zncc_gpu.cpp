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

void shrink_and_grey_kcopy(		const unsigned char *i1,
								const unsigned char *i2,
								unsigned char *o1,
								unsigned char *o2,
								int const shrink_factor, // MAKE CHAR
								int const w, SIZE ndrange ){
	long int ws, addr;
	const int Bpp = 4;
	for(long int i=0; i<ndrange.cy; i++){
		for(long int j=0; j<ndrange.cx; j++){
			ws = (w/shrink_factor);
			addr = (i*w + j) * shrink_factor * Bpp;
			o1[i*ws + j] = (unsigned char)(0.2126*i1[addr] + 0.7152*i1[addr + 1] + 0.0722*i1[addr + 2]);
			//o2[i*ws + j] = (unsigned char)(0.2126*i2[addr] + 0.7152*i2[addr + 1] + 0.0722*i2[addr + 2]);
		}
	}
}


int read_png_grey_and_shrink_gpu(	cl_device_id device,
									cl_context context,
									const char * const filepath1,
									const char * const filepath2,
									const int maximum_disparity,
									const int shrink_factor,
									BYTE * * result,
									SIZE *size	){
	cl_mem		im0_raw_buf_cl;
	cl_mem		im0_small_buf_cl;
	cl_mem		im1_raw_buf_cl;
	cl_mem		im1_small_buf_cl;
	cl_mem		result_buf_cl;
	// cl_mem		znccs_g_cl;
	cl_program	program;
	

	const int &		reserve_size = maximum_disparity;
	const int		inpng_bytes_per_pixel = 4; //RGBA
	int 			window_size = 9;
	int 			maxdisp_offset = 0;
	int 			result_h, result_w;
	int 			allocation_size;
	unsigned int 	im0_w, im0_h, im1_w, im1_h, error, num_elements_small;
	BYTE *			zncc_kernel_src = NULL;
	char 			buffer[102400];
	std::vector<unsigned char> im0_vector, im1_vector;

	// C++ standard requires short circuit evaluation. So this should work.
	puts("Reading file...");
	if((error=lodepng::decode(im0_vector, im0_w, im0_h, filepath1)) || (error=lodepng::decode(im1_vector, im1_w, im1_h, filepath2)))
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
	*result = (BYTE *)(calloc(allocation_size, 1));
	*result = (*result) + reserve_size;
	
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
	//CL_CHECK(clUnloadCompiler());

	//BYTE *b1 = (BYTE *)calloc(im0_vector.capacity());
	//BYTE *b2 = (BYTE *)calloc(im1_vector.capacity());
	//std::copy(im0_vector.begin(), im0_vector.end(), b1);
	//std::copy(im1_vector.begin(), im1_vector.end(), b2);
	BYTE *b1 = &im0_vector[0];
	BYTE *b2 = &im1_vector[0];
	im0_raw_buf_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE,  4*im0_w * im0_h, NULL, &_err));
	im1_raw_buf_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE,  4*im0_w * im0_h, NULL, &_err));
	im0_small_buf_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE,   allocation_size, NULL, &_err));
	im1_small_buf_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE,   allocation_size, NULL, &_err));
	//znccs_g_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, maximum_disparity*num_elements_small, NULL, &_err));
	result_buf_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, num_elements_small, *result, &_err));

	cl_mem im0_subbuf, im1_subbuf; //SUBBUFFER WEIRD
	cl_buffer_region bufregion = cl_buffer_region{reserve_size, num_elements_small};

	//VARLOG(bufregion.origin);
	//VARLOG(bufregion.size);
	VARLOG(reserve_size);
	VARLOG(num_elements_small);
	VARLOG(allocation_size);

	im0_subbuf = CL_CHECK_ERR(clCreateSubBuffer(im0_small_buf_cl, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION,&bufregion,&_err));
	im1_subbuf = CL_CHECK_ERR(clCreateSubBuffer(im1_small_buf_cl, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION,&bufregion,&_err));

	unsigned char 	sf_c = (unsigned char)shrink_factor;
	cl_command_queue queue;
	queue = CL_CHECK_ERR(clCreateCommandQueue(context, device, 0, &_err));
	
	cl_kernel kernel_sgrey = CL_CHECK_ERR(clCreateKernel(program, "shrink_and_grey", &_err));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 0, sizeof(im0_raw_buf_cl), &im0_raw_buf_cl));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 1, sizeof(im1_raw_buf_cl), &im1_raw_buf_cl));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 2, sizeof(im0_subbuf), &im0_subbuf));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 3, sizeof(im1_subbuf), &im1_subbuf));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 4, sizeof(sf_c), &sf_c));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 5, sizeof(im0_w), &im0_w));
	//CL_CHECK(clSetKernelArg(kernel_sgrey, 1, sizeof(im0_raw_buf_cl), &im0_raw_buf_cl));
	//CL_CHECK(clSetKernelArg(kernel_sgrey, 1, sizeof(im1_raw_buf_cl), &im1_raw_buf_cl));
	//CL_CHECK(clSetKernelArg(kernel_sgrey, 3, sizeof(im0_subbuf), &im0_subbuf));
	//CL_CHECK(clSetKernelArg(kernel_sgrey, 3, sizeof(im1_subbuf), &im1_subbuf));

	cl_kernel kernel_zncc = CL_CHECK_ERR(clCreateKernel(program, "compute_disparity", &_err));
	CL_CHECK(clSetKernelArg(kernel_zncc, 0, sizeof(im0_subbuf), &im0_subbuf));
	CL_CHECK(clSetKernelArg(kernel_zncc, 1, sizeof(im1_subbuf), &im1_subbuf));
	CL_CHECK(clSetKernelArg(kernel_zncc, 2, sizeof(result_buf_cl), &result_buf_cl));
	CL_CHECK(clSetKernelArg(kernel_zncc, 3, sizeof(result_w), &result_w));
	CL_CHECK(clSetKernelArg(kernel_zncc, 4, sizeof(result_h), &result_h));
	CL_CHECK(clSetKernelArg(kernel_zncc, 5, sizeof(window_size), &window_size));
	CL_CHECK(clSetKernelArg(kernel_zncc, 6, sizeof(maxdisp_offset), &maxdisp_offset));
	//CL_CHECK(clSetKernelArg(kernel_zncc, 7, sizeof(znccs_g_cl), &znccs_g_cl));
	



	CL_CHECK(clEnqueueWriteBuffer(	queue, im0_raw_buf_cl, CL_TRUE,	0, 4*im0_h * im0_w, b1, 0, NULL, NULL ));
	CL_CHECK(clEnqueueWriteBuffer(	queue, im1_raw_buf_cl, CL_TRUE,	0, 4*im1_h * im1_w, b2, 0, NULL, NULL ));
/*	CL_CHECK(clEnqueueWriteBuffer(	queue, im1_raw_buf_cl, CL_TRUE, 0, 4*sizeof(BYTE)*num_elements_small, \
										b2, 0, NULL, NULL ));
*/
	cl_event kernel_completion[2];
	size_t global_work_size_sgrey[2] = { size_t(result_h), size_t(result_w) };

	puts("Enqueing 1...");
	VARLOG(result_h);
	VARLOG(result_w);
	VARLOG(im0_w);
	//shrink_and_grey_kcopy(b1, b2, *result, NULL, shrink_factor, im0_w, SIZE{result_w, result_h});
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_sgrey, 2, NULL, global_work_size_sgrey, NULL, 0, NULL, &kernel_completion[0]));
	CL_CHECK((clWaitForEvents(1, &kernel_completion[0])));
	CL_CHECK(clReleaseEvent(kernel_completion[0]));
	
#define ZNCC_WORKER_DIMS 3
	size_t global_work_size_zncc[ZNCC_WORKER_DIMS] = { size_t(result_h), size_t(result_w), maximum_disparity};
	size_t local_size_zncc[ZNCC_WORKER_DIMS] = { 1, 1, maximum_disparity};
	puts("Enqueing 2...");
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_zncc, ZNCC_WORKER_DIMS, NULL, global_work_size_zncc, local_size_zncc, 0, NULL, &kernel_completion[1]));
	CL_CHECK(clWaitForEvents(1, &kernel_completion[1]));
	CL_CHECK(clReleaseEvent(kernel_completion[1]));

	cl_mem copy_from_buf_cl = result_buf_cl;
	printf("im0_small_buf_cl==im0_subbuf? %d\n", im0_small_buf_cl==im0_subbuf);
	puts("Copying result...");
	size->cy = (im0_h / shrink_factor);
	size->cx = (im0_w / shrink_factor);
	CL_CHECK(clEnqueueReadBuffer(	queue,
									copy_from_buf_cl,
									CL_TRUE,
									0 , //i*sizeof(int)
									num_elements_small,
									*result,
									0,
									NULL,
									NULL	));
	

	CL_CHECK(clReleaseMemObject(im0_raw_buf_cl));
	CL_CHECK(clReleaseMemObject(im1_raw_buf_cl));
	CL_CHECK(clReleaseMemObject(im0_small_buf_cl));
	CL_CHECK(clReleaseMemObject(im1_small_buf_cl));
	CL_CHECK(clReleaseMemObject(result_buf_cl));
	CL_CHECK(clReleaseKernel(kernel_sgrey));
	CL_CHECK(clReleaseKernel(kernel_zncc));
	CL_CHECK(clReleaseProgram(program));
	CL_CHECK(clReleaseContext(context));

	return 0;
}