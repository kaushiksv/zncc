# ZNCC Based Depthmapping with Multithreading (CPU) and OpenCL (GPU)


Example usage for GPU based computation:
```
      $ ./zncc --use-gpu
      la 26.5.2018 10.32.04 +0300 ::  maxdisp = 64;  winsize = 09;  thres = 08;  nhood = 08, t_sg = 0.0000 ms;  t_d0 = 38.0000 ms;  t_d1 = 38.0000 ms;  t_cc = 0.0000 ms;  t_of = 0.0000 ms

      $ ./zncc --use-gpu --window-size=19 --threshold=25 --show-status
      Reading png files...
      Shrink/grey...
      Computing depthmap 1 of 2
      Computing depthmap 2 of 2
      Cross checking...
      Occlusion fill...
      Copying to host memory....
      Done.
      la 26.5.2018 10.30.45 +0300 ::  maxdisp = 64;  winsize = 19;  thres = 25;  nhood = 08, t_sg = 0.0000 ms;  t_d0 = 162.0000 ms;  t_d1 = 148.0000 ms;  t_cc = 0.0000 ms;  t_of = 0.0000 ms
      
      $ ./zncc --use-gpu --platform-number=0 --device-number=1
      No device or invalid device number. Aborting...
      Aborted (core dumped)

      $ ./zncc --use-gpu --image-0="path/to/im0.png"
      Error: 78
      failed to open file for reading
      Aborted (core dumped)

```

Example commands for CPU only computation:
```
      $ ./zncc --nthreads=25
      $ ./zncc --nthreads=20 --image-0="path/to/im0.png"
      $ ./zncc --nthreads=20 -w 9 --show-status
```

Compilation:
```
      $ echo $LD_LIBRARY_PATH
      :/usr/local/cuda-9.1/lib64

      ## CPU ONLY VERSION:
      $ make cpu
      ./zncc --use-gpu
      Recompile with GPU support :)

      $ make
      ./zncc --use-gpu
      la 26.5.2018 10.42.28 +0300 ::  maxdisp = 64;  winsize = 09;  thres = 08;  nhood = 08, t_sg = 0.0000 ms;  t_d0 = 39.0000 ms;  t_d1 = 39.0000 ms;  t_cc = 0.0000 ms;  t_of = 0.0000 ms
```


im0.png and im1.png are used by default. Every call will append a line to performance_log.txt with timestamp, parameters, and timing information. In CPU mode, set the shell variable INTIMG=1 to get preliminary depthmaps before cross-checking and occlusion filling. Try "zncc --help" for more available options.


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
