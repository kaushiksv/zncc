#ifndef INCLUDES_H
#define INCLUDES_H
#include <iostream>
#include <stdio.h>
#include <memory.h>
#include <limits.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <algorithm>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "cmdline.h"
#include "lodepng.h"

#ifdef GPU_SUPPORT
#include <CL/cl.h>
#include "clerrmacros.h"
#endif

#include "zncc.h"
#include "util.h"


#endif
