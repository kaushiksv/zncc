#include "includes.h"

// To store CL objects globally yet neatly.
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
	cl_kernel			kernel_cc;
	cl_kernel			kernel_ofill;
	cl_command_queue	queue;
	cl_event			kernel_completion[6];
	cl_mem				im0_raw_buffer, im0_small_buffer, im0_sub_buffer;
	cl_mem				im1_raw_buffer, im1_small_buffer, im1_sub_buffer;
	cl_mem				dmap1, dmap2, cchk, ofill;
	cl_buffer_region	buffer_region;
};


/*
	Function to initialize opencl objects
*/
void opencl_init(int platform_number, int device_number, char *build_options){
	
	using namespace cl_objects;

	BYTE *				zncc_kernel_src;
	char 				buffer[102400];

	////////////////////////////////////////////////////////////////////////
	////////
	//////// Get platforms, devices, context, yada yada....
	////////
	////////////////////////////////////////////////////////////////////////

	CL_CHECK(clGetPlatformIDs(100, platforms, &platforms_n));
	if (platforms_n == 0) {
		puts("No platform. Aborting...");
		abort();
	}
	CL_CHECK(clGetDeviceIDs(platforms[platform_number], CL_DEVICE_TYPE_GPU, 100, devices, &devices_n));
	if (devices_n == 0 || !in_range(device_number, 0, devices_n-1)) {
		puts("No device or invalid device number. Aborting...");
		abort();
	}
	device = devices[device_number];
	context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device, &pfn_notify, NULL, &_err));

	////////////////////////////////////////////////////////////////
	////////
	////////  Read and compile the file containing kernels' sources.
	////////
	////////////////////////////////////////////////////////////////
	if((zncc_kernel_src = read_file("zncc.cl")) == NULL){
		puts("Can't load zncc.cl source.");
		abort();
	}

	program = CL_CHECK_ERR(clCreateProgramWithSource(context, 1, (const char **)&zncc_kernel_src, NULL, &_err));

	////////////////////////////////////////////////////////////////
	////////
	////////  Build with 'build_options' (Used for optimization)
	////////
	////////////////////////////////////////////////////////////////
	if (clBuildProgram(program, 1, &device, build_options, NULL, NULL) != CL_SUCCESS) {
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		fprintf(stderr, "CL Compilation-1 failed:\n%s", buffer);
		abort();
	}
	CL_CHECK(clUnloadCompiler());
	queue = CL_CHECK_ERR(clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &_err));

	kernel_sgrey = CL_CHECK_ERR(clCreateKernel(program, "shrink_and_grey", &_err));
	kernel_zncc = CL_CHECK_ERR(clCreateKernel(program, "compute_disparity", &_err));
	kernel_cc = CL_CHECK_ERR(clCreateKernel(program, "cross_check_inplace", &_err));
	kernel_ofill = CL_CHECK_ERR(clCreateKernel(program, "occlusion_fill_inplace", &_err));
}


/*
	Allocate buffers in the GPU.
*/
void opencl_create_buffers(int inpng_buffer_size, int result_buffer_size, BYTE *png1_buffer, BYTE *png2_buffer, BYTE *blank_buffer){
	using namespace cl_objects;

	// Copy the images png1_buffer and png2_buffer to GPU global memory
	im0_raw_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, inpng_buffer_size, png1_buffer, &_err));
	im1_raw_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, inpng_buffer_size, png2_buffer, &_err));

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////
	////////	Create small buffer to store downscaled greyscale images (init with '\0's, hence blank_buffer)
	////////
	////////	Note. This size of result_buffer_size is ( (small_width * small_height) + 2*maximum_disparity)
	////////			This helps avoid bounds checking when reading the sub-buffers created in next step.
	////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	im0_small_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, result_buffer_size, blank_buffer, &_err));
	im1_small_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, result_buffer_size, blank_buffer, &_err));

	////////////////////////////////////////////////////////////////
	////////
	//////// The sub-buffers for im0 and im1 downscaled images
	//////// This is useful in avoiding bounds checking ('if' condition )
	//////// when executing the kernel. Lesser branching and improved speed
	//////// 
	//////// Subbuffer varies from (0 + maximum_disparity)....(N-1 - maximum_disparity)
	////////
	////////////////////////////////////////////////////////////////
	im0_sub_buffer = CL_CHECK_ERR(clCreateSubBuffer(im0_small_buffer, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION,&buffer_region,&_err));
	im1_sub_buffer = CL_CHECK_ERR(clCreateSubBuffer(im1_small_buffer, CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION,&buffer_region,&_err));

	// maximum_disparity*2*4 extra unused bytes are ok for now. Hence used result_buffer_size :)
	dmap1 = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, result_buffer_size, NULL, &_err));
	dmap2 = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, result_buffer_size, NULL, &_err));
	cchk = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, result_buffer_size, NULL, &_err));
	ofill = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, result_buffer_size, NULL, &_err));
}

void exec_project_gpu	(	const char * const img0_arg,
							const char * const img1_arg,
							const int maximum_disparity,
							const int window_size,
							const int threshold,
							const int shrink_factor,
							const int neighbourhood_size,
							const int platform_number,
							const int device_number ) {
	
	using namespace cl_objects;

	// Default filepaths
	const char 			*im0_filepath = "im0.png";
	const char 			*im1_filepath = "im1.png";
	char				s[512];
	char				build_options[512]="";
	const int 			half_win = (window_size-1)/2;

	// In some places, `reserve_size' is used for better code understandability
	const int &			reserve_size = maximum_disparity;

	// Input images dimensions
	unsigned int 		im0_w, im0_h, im1_w, im1_h, error;

	// Dimensions, and number of elements, size of downscaled greyscale image.
	unsigned int		num_elements_small;
	int 				result_h, result_w;
	int 				allocation_size;

	// Passed to kernel to deal with negative disparity
	int 				maxdisp_offset = 0;
	BYTE *				zncc_kernel_src = NULL;
	std::vector<BYTE> 	im0_vector, im1_vector;

	// Use input filepaths if given in command line argument
	im0_filepath = img0_arg?img0_arg:im0_filepath;
	im1_filepath = img1_arg?img1_arg:im1_filepath;

	status_update("Reading png files...\n");
	// C++ standard requires short circuit evaluation. So this will work.
	if((error=lodepng::decode(im0_vector, im0_w, im0_h, im0_filepath)) || (error=lodepng::decode(im1_vector, im1_w, im1_h, im1_filepath)))
	{
		handle_lodepng_error(error);
	}
	
	if((im0_w != im1_w) || (im1_h != im1_h)){
		printf("Image sizes mismatch.\n");
		return;
	}
	
	// Size of scaled down images
	result_h = int(im0_h / shrink_factor);
	result_w = int(im0_w / shrink_factor);

	////////////////////////////////////////////////////////////////////////////////////////
	////////
	//////// Optimizations. Using macro instead of constant variables enables compile-time
	//////// optimizations, and hopefully better loop outrolling.
	////////
	////////////////////////////////////////////////////////////////////////////////////////
	sprintf_s(build_options, sizeof(build_options), "-D WINDOW_SIZE=%d -D IMAGE_WIDTH=%d -D IMAGE_HEIGHT=%d -cl-mad-enable", window_size, result_w, result_h);


	// Initialize on given platform, device with these build options.
	opencl_init(platform_number, device_number, build_options);

	// Create a blank buffer of size:    result_w  x  result_h (+2*reserveSize)
	num_elements_small = result_h * result_w;
	allocation_size = num_elements_small + 2*reserve_size; // For output small image
	BYTE *temp = (BYTE *)(calloc(allocation_size, 1));
	temp = temp + reserve_size;

	// Set buffer region for subbuffers to be created (without the reserves on both ends)
	buffer_region = cl_buffer_region{size_t(reserve_size), size_t(num_elements_small)};

	// Create required buffers on GPU.
	opencl_create_buffers(4 * im0_w * im0_h, allocation_size, &im0_vector[0], &im1_vector[0], temp);



	////////////////////////////////////////////////////////////////
	////////
	//////// In, compute_disparity kernel:
	////////
	//////// Each workgroup has upto 64 work items, each with unique 'd' value
	////////
	//////// All work-items on same (x,y) points on image
	////////
	//////// This helps share left image mean and std in __local memory.
	////////
	////////////////////////////////////////////////////////////////

	/* Presetting kernel args for little performance boost. */
	size_t global_work_size_zncc[3]  =  { size_t(result_h), size_t(result_w), size_t(maximum_disparity)};

	size_t local_size_zncc[3]        =  {                1,                1, size_t(maximum_disparity)};

	size_t global_work_size_sgrey[2] =  { size_t(result_h), size_t(result_w) };

	// Shrink and grey kernel
	CL_CHECK(clSetKernelArg(kernel_sgrey, 0, sizeof(im0_raw_buffer), &im0_raw_buffer));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 1, sizeof(im1_raw_buffer), &im1_raw_buffer));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 2, sizeof(im0_sub_buffer), &im0_sub_buffer));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 3, sizeof(im1_sub_buffer), &im1_sub_buffer));
	unsigned char 	sf_c = (unsigned char)shrink_factor;
	CL_CHECK(clSetKernelArg(kernel_sgrey, 4, sizeof(sf_c), &sf_c));
	CL_CHECK(clSetKernelArg(kernel_sgrey, 5, sizeof(im0_w), &im0_w));

	// Preliminary depthmap 1 (compute_disparity)
	CL_CHECK(clSetKernelArg(kernel_zncc, 0, sizeof(im0_sub_buffer), &im0_sub_buffer));
	CL_CHECK(clSetKernelArg(kernel_zncc, 1, sizeof(im1_sub_buffer), &im1_sub_buffer));
	CL_CHECK(clSetKernelArg(kernel_zncc, 2, sizeof(dmap1), &dmap1));
	CL_CHECK(clSetKernelArg(kernel_zncc, 3, sizeof(result_w), &result_w));
	CL_CHECK(clSetKernelArg(kernel_zncc, 4, sizeof(result_h), &result_h));
	CL_CHECK(clSetKernelArg(kernel_zncc, 5, sizeof(window_size), &window_size));
	CL_CHECK(clSetKernelArg(kernel_zncc, 6, sizeof(maxdisp_offset), &maxdisp_offset));


	// Enqueue both consequtively.	
	status_update("Shrink/grey...\n");
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_sgrey, 2, NULL, global_work_size_sgrey, NULL, 0, NULL, &kernel_completion[0]));
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_zncc, 3, NULL, global_work_size_zncc, local_size_zncc, 1, &kernel_completion[0], &kernel_completion[1]));

	CL_CHECK((clWaitForEvents(1, &kernel_completion[0])));
	// CL_CHECK(clReleaseEvent(kernel_completion[0]));
	status_update("Computing depthmap 1 of 2\n");

	CL_CHECK(clWaitForEvents(1, &kernel_completion[1]));
	// CL_CHECK(clReleaseEvent(kernel_completion[1]));

	//////// Same kernel, different set of args.
	//////// So WAITING before we set another set of args.
	//////// Can be changed to improve speed slightly.

	// Set args for 2nd preliminary depthmap
	// To deal with negative disparity
	maxdisp_offset = -64;
	CL_CHECK(clSetKernelArg(kernel_zncc, 0, sizeof(im1_sub_buffer), &im1_sub_buffer));
	CL_CHECK(clSetKernelArg(kernel_zncc, 1, sizeof(im0_sub_buffer), &im0_sub_buffer));
	CL_CHECK(clSetKernelArg(kernel_zncc, 2, sizeof(dmap2), &dmap2));
	CL_CHECK(clSetKernelArg(kernel_zncc, 3, sizeof(result_w), &result_w));
	CL_CHECK(clSetKernelArg(kernel_zncc, 4, sizeof(result_h), &result_h));
	CL_CHECK(clSetKernelArg(kernel_zncc, 5, sizeof(window_size), &window_size));
	CL_CHECK(clSetKernelArg(kernel_zncc, 6, sizeof(maxdisp_offset), &maxdisp_offset));

	// Set args for cross-check
	size_t global_work_size_cc[2]    =  { size_t(result_h), size_t(result_w) };
	CL_CHECK(clSetKernelArg(kernel_cc, 0, sizeof(dmap1), &dmap1));
	CL_CHECK(clSetKernelArg(kernel_cc, 1, sizeof(dmap2), &dmap2));
	CL_CHECK(clSetKernelArg(kernel_cc, 2, sizeof(threshold), &threshold));

	// Set args for occlusion-fill
	size_t global_work_size_ofill[2] =	{
	// 										Ignore boundaries with black (pixel value 0)
											size_t(result_h) - 2 * half_win,
											size_t(result_w) - 2 * half_win
										};
	CL_CHECK(clSetKernelArg(kernel_ofill, 0, sizeof(dmap1), &dmap1));
	CL_CHECK(clSetKernelArg(kernel_ofill, 1, sizeof(half_win), &half_win));
	CL_CHECK(clSetKernelArg(kernel_ofill, 2, sizeof(neighbourhood_size), &neighbourhood_size));
	// Select cl_mem to copy from (variable added for easy debugging)
	cl_mem copy_from_buf_cl = dmap1;


	// Enque all the three kernels consecutively.
	status_update("Computing depthmap 2 of 2\n");
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_zncc, 3, NULL, global_work_size_zncc, local_size_zncc, 0, NULL, &kernel_completion[2]));
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_cc, 2, NULL, global_work_size_cc, NULL, 1, &kernel_completion[2], &kernel_completion[3]));
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_ofill, 2, NULL, global_work_size_ofill, NULL, 1, &kernel_completion[3], &kernel_completion[4]));

	// Enqueue copying to host memory from GPU memory.
	CL_CHECK(clEnqueueReadBuffer(	queue,
									copy_from_buf_cl,
									CL_TRUE,
									0 , //i*sizeof(int)
									num_elements_small,
									temp,
									1,
									&kernel_completion[4],
									&kernel_completion[5]	));

	CL_CHECK(clWaitForEvents(1, &kernel_completion[2]));
	// CL_CHECK(clReleaseEvent(kernel_completion[2]));
	status_update("Cross checking...\n");

	CL_CHECK(clWaitForEvents(1, &kernel_completion[3]));
	// CL_CHECK(clReleaseEvent(kernel_completion[3]));
	status_update("Occlusion fill...\n");

	CL_CHECK(clWaitForEvents(1, &kernel_completion[4]));
	// CL_CHECK(clReleaseEvent(kernel_completion[4]));
	status_update("Copying to host memory....\n");

	CL_CHECK(clWaitForEvents(1, &kernel_completion[5]));
	// CL_CHECK(clReleaseEvent(kernel_completion[5]));
	status_update("Done.\n");

	clFinish(queue);

	cl_ulong time_start[6];
	cl_ulong time_end[6];
	double running_time[6];

	for(int i=0; i<6; i++){
		clGetEventProfilingInfo(kernel_completion[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start[i], NULL);
		clGetEventProfilingInfo(kernel_completion[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end[i], NULL);
		CL_CHECK(clReleaseEvent( kernel_completion[i] ));
		running_time[i] = (time_end[i]-time_start[i])/1000000; // To ms
	}

	// Print timings on screen and append to file.
	sprintf_s(s, sizeof(s), "echo \"$(date) ::  maxdisp = %02d;  winsize = %02d;  thres = %02d;  nhood = %02d, t_sg = %0.4lf ms;  t_d0 = %0.4lf ms;  t_d1 = %0.4lf ms;  t_cc = %0.4lf ms;  t_of = %0.4lf ms\n\"",
				maximum_disparity, window_size, threshold, neighbourhood_size, running_time[0], running_time[1], running_time[2], running_time[3], running_time[4]);
	system(s);
	sprintf_s(s + strlen(s), sizeof(s), " >> \"%s\"", LOGFILE);
	system(s);

	// Write output image to disk
	lodepng::encode(	"outputs/depthmap.png"	, temp, result_w, result_h, LCT_GREY, 8U);


	// Free buffers.
	free(temp-reserve_size);
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

}
