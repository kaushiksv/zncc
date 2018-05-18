CC=g++
CFLAGS=-g -std=gnu++11 -pthread -I. -lOpenCL
#-Wall
zncc: main.cpp zncc.cpp lodepng.cpp cmdline.c util.cpp zncc.h lodepng.h cmdline.h util.h
	$(CC) -o zncc main.cpp zncc.cpp lodepng.cpp cmdline.c util.cpp $(CFLAGS) 
