#pragma once
#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <mach/mach_time.h>
#include <mach/mach.h>
#include <CoreServices/CoreServices.h>
#else
#include <CL/cl.h>
#if defined(WIN32) || defined (__WIN32)
#include <Windows.h>
#else
#include <sys/time.h>
#include <stdio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <new>
#endif
#endif
#include <vector>
#include <fstream>
#include <string>
#include <exception>
#include <map>
#include <memory>
#include <sstream>
#include <iostream>
#include <cmath>