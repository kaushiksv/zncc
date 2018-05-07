# ZNCC Multithreaded CPU Implementation


Example usage:
	./zncc --nthreads=25
	./zncc --nthreads=20 -w 9
	./zncc --nthreads=20 --image-0="path/to/im0.png"

im0.png and im1.png are used by default. Outputs 4 images (2 prelim disparity maps, cross-checked, and occlusion filled images) to 'outputs/'. See 'exec_project' in zncc.pp for details. Also, the params, and timing information is appended to outputs/log.txt


## Options

	$ ./zncc --help
	zncc 1.0

	Multithreaded implementation

	Usage: zncc [OPTIONS]

	Simple ZNCC depthmap implementation. Looks for im0.png and im1.png in working
	directory. Outputs to outputs/ relative to working directory. See zncc.cpp for
	details.

	  -h, --help                   Print help and exit
	  -V, --version                Print version and exit
	  -d, --maximum-disparity=INT  The maximum disparity between images.
	                                 (default=`65')
	  -t, --threshold=INT          The threshold used for cross-checking.
	                                 (default=`8')
	  -w, --window-size=INT        The length of zncc window. This parameter
	                                 represents one side of the window used for
	                                 zncc. (Block size is square of the value
	                                 specified here)  (default=`8')
	      --nthreads=INT           Number of threads for zncc computation.
	                                 (default=`1')
	  -s, --skip-depthmapping      Skip computation of disparity images. Skip
	                                 computation of disparity images. This option
	                                 will use supplied images, and if none is
	                                 supplied, looks for previously output files at
	                                 ./output/ directory. Missing files will cause
	                                 the program to terminate.  (default=off)
	      --image-0=STRING         Image 0 filepath
	      --image-1=STRING         Image 1 filepath
	      --shrink-by=INT          Shrink factor to downscale image. Typically set
	                                 to 1 when skipping depthmapping step.
	                                 (default=`4')

	Author: Kaushik Sundarajayaraman Venkat
	E-mail: speak2kaushik@gmail.com, kaushik.sv@student.oulu.fi
