CC=g++
CFLAGS=-g -std=gnu++11 -pthread -Wall -I. -lOpenCL

zncc: main.cpp zncc.cpp lodepng.cpp cmdline.c zncc.h lodepng.h cmdline.h
	$(CC) -o zncc main.cpp zncc.cpp lodepng.cpp cmdline.c $(CFLAGS) 
