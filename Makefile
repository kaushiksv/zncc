CC=g++
CFLAGS=-g -std=gnu++11 -pthread -I. -o3
CLFLAGS=-lOpenCL -DGPU_SUPPORT
#-Wall
gpu: main.cpp zncc.cpp lodepng.cpp cmdline.c util.cpp zncc_gpu.cpp
	$(CC) -o zncc main.cpp zncc.cpp lodepng.cpp cmdline.c util.cpp zncc_gpu.cpp $(CFLAGS) $(CLFLAGS)

cpu: main.cpp zncc.cpp lodepng.cpp cmdline.c util.cpp
	$(CC) -o zncc main.cpp zncc.cpp lodepng.cpp cmdline.c util.cpp $(CFLAGS) 
