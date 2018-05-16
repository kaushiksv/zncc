CC=g++
CFLAGS=-g -std=gnu++11 -pthread -Wall -I.

zncc: main.cpp zncc.cpp lodepng.cpp cmdline.c
	$(CC) -o zncc main.cpp zncc.cpp lodepng.cpp cmdline.c $(CFLAGS) 
