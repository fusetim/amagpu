// Sylvain Lefebvre - 2012-03-02

#include <iostream>

#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__ 
#include <OpenCL/opencl.h> //Mac OSX has a different name for the CL header file
#include "clv128.hpp" // manually included C++ bindings
#else
#include "CL/cl.h"
#include "CL/cl.hpp"
#endif

// ----------------------------------------------------------

extern cl::Context             *clu_Context;
extern std::vector<cl::Device>  clu_Devices;
extern cl::CommandQueue        *clu_Queue;

// ----------------------------------------------------------

void          cluInit(cl_device_type devtype = CL_DEVICE_TYPE_GPU);
cl::Program  *cluLoadProgram(const char *);
cl::Program  *cluLoadProgramFromString(const char *);
cl::Kernel   *cluLoadKernel(cl::Program *,const char *); 
void          cluCheckError(cl_int, const char *);
double        cluDisplayEventMilliseconds(const char *msg,const cl::Event& ev);
double        cluEventMilliseconds(const cl::Event& ev);
std::string   cluLoadFileIntoString(const char *file);
long long     cluCPUMilliseconds();

// ----------------------------------------------------------

