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
	zncc 2.0

	Multithreaded and OpenCL (GPU) implementation

	Usage: zncc [OPTIONS]

	Computes ZNCC based depthmap. Looks for im0.png and im1.png in working
	directory. Outputs to outputs/depthmap.png relative to working directory. See
	zncc.cpp for details.

	  -h, --help                    Print help and exit
	  -V, --version                 Print version and exit
	      --use-gpu                 Use GPU for computation.  (default=off)
	  -d, --maximum-disparity=INT   The maximum disparity between images.
	                                  (default=`64')
	  -t, --threshold=INT           The threshold used for cross-checking.
	                                  (default=`8')
	  -w, --window-size=INT         Side of the window used for zncc. Must be odd.
	                                  (Ex: 11, window has 121 elements)
	                                  (default=`9')
	  -n, --neighbourhood-size=INT  The neighbourhood size for occlusion filling.
	                                  (default=`8')
	      --show-status             Print status-update messages that describe the
	                                  ongoing activity.  (default=off)
	      --platform-number=INT     The platform number (different from platform
	                                  ID) varies from 0..N_PLATFORMS-1. Use a tool
	                                  like clinfo to customize this.  (default=`0')
	      --device-number=INT       The device number (different from device ID)
	                                  varies from 0..N_DEVICES-1. Use a tool like
	                                  clinfo to customize this.  (default=`0')
	      --nthreads=INT            Number of threads for zncc computation. Has no
	                                  effect when using GPU.  (default=`1')
	  -s, --skip-depthmapping       OBSELETE. Previously, this flag had been used
	                                  to skip computation of preliminary depthmaps,
	                                  and reuse previously output images. Has no
	                                  effect when using GPU. This option will use
	                                  images specified by --image-0 and --image-1
	                                  options. if ommitted, it looks for previously
	                                  output files at ./outputs/ directory, and use
	                                  them to perform just cross-checking and
	                                  occlusion-filling. Missing files would cause
	                                  the program to terminate. `d0_filepath` and
	                                  `d1_filepath` in zncc.cpp define the default
	                                  files that will be looked for.  (default=off)
	      --image-0=STRING          Image 0 filepath
	      --image-1=STRING          Image 1 filepath
	      --shrink-by=INT           Shrink factor to downscale image. Typically set
	                                  to 1 when skipping depthmapping step.
	                                  (default=`4')

	In CPU mode, set shell variable INTIMG=1 before invoking to output intermediary
	files.

	Author: Kaushik Sundarajayaraman Venkat
	E-mail: speak2kaushik@gmail.com, kaushik.sv@student.oulu.fi
