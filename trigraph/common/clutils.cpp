// Sylvain Lefebvre - 2012-03-02

#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;


#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__ 
#include <OpenCL/opencl.h> //Mac OSX has a different name for the CL header file
#include "clv128.hpp" // manually included C++ bindings
#else
#include "CL/cl.h"
#include "CL/cl.hpp"
#endif

// ----------------------------------------------------------

void cluCheckError(cl_int err, const char * name)
{
  if (err != CL_SUCCESS) {
    fprintf(stderr,"ERROR: %d - %s\n",err,name);
    exit (-1);
  }
}

// ----------------------------------------------------------

string cluLoadFileIntoString(const char *file)
{
  ifstream infile( file , ios::binary );
  if (!infile) {
    fprintf(stderr,"[loadFileIntoString] - file '%s' not found\n",file);
    exit (-1);
  }
  ostringstream strstream;
  while (infile) {
    ifstream::int_type c = infile.get();
    if (c != EOF)
      strstream << char(c);
    else
      break;
  } 
  return strstream.str();
}

// ----------------------------------------------------------

cl::Context             *clu_Context;
std::vector<cl::Device>  clu_Devices;
cl::CommandQueue        *clu_Queue;

// ----------------------------------------------------------

void cluInit(cl_device_type devtype)
{
  cl_int err;

  // get platform information
  std::vector<cl::Platform> platformList;
  cl::Platform::get(&platformList);
  cluCheckError(platformList.size()!=0 ? CL_SUCCESS : -1, "cl::Platform::get");
  std::cerr << "Number of CL platforms: " << platformList.size() << std::endl;

  std::string platformVendor;
  platformList.back().getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
  std::cerr << "CL platform is by: " << platformVendor << "\n";
  
  // ask for OpenCL context creation
  cl_context_properties cprops[3] = 
  {CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList.back())(), 0};
  clu_Context = new cl::Context(
    devtype,
    cprops,
    NULL,
    NULL,
    &err);
  cluCheckError(err, "Context::Context()");

  // get available devices
  clu_Devices = clu_Context->getInfo<CL_CONTEXT_DEVICES>();
  cluCheckError( clu_Devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

  cerr << "Vendor ID:\t\t\t"            << clu_Devices[0].getInfo<CL_DEVICE_VENDOR_ID>() << std::endl;
  cerr << "Max compute units:\t\t"      << clu_Devices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
  cerr << "Max work item dimensions:\t" << clu_Devices[0].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
  size_t size[3];
  clu_Devices[0].getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES,&size);
  cerr << "Max work item sizes:\t\t"    << "( " << size[0] << ", " << size[1] << ", " << size[2] << " )\n";
  cerr << "Max work group size:\t\t"    << clu_Devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
  cerr << "Max clock requency:\t\t"     << clu_Devices[0].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << " MHz\n";
  cerr << "Max mem alloc size:\t\t"     << clu_Devices[0].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()/(1<<20) << " MB\n";

  // create a processing queue
  clu_Queue = new cl::CommandQueue(*clu_Context, clu_Devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
  cluCheckError(err, "CommandQueue::CommandQueue()");

}

// ----------------------------------------------------------

cl::Program  *cluLoadProgram(const char *progFile)
{
  cl_int err;
  cl::Program *prg;
  // load program
  std::string prog = cluLoadFileIntoString(progFile);
  cl::Program::Sources source( 1, std::make_pair(prog.c_str(), prog.length()));
  prg = new cl::Program(*clu_Context, source);
  err = prg->build(clu_Devices,"");
  
  string str = prg->getBuildInfo<CL_PROGRAM_BUILD_LOG>(clu_Devices[0]);
  cerr << str << endl;
  cluCheckError(err, "Program::build()");

  return prg;
}

// ----------------------------------------------------------

cl::Program  *cluLoadProgramFromString(const char *progString)
{
  cl_int err;
  cl::Program *prg;
  // load program
  cl::Program::Sources source( 1, std::make_pair(progString, strlen(progString)+1));
  prg = new cl::Program(*clu_Context, source);
  err = prg->build(clu_Devices,"");
  
  string str = prg->getBuildInfo<CL_PROGRAM_BUILD_LOG>(clu_Devices[0]);
  cerr << str << endl;
  cluCheckError(err, "Program::build()");

  return prg;
}

// ----------------------------------------------------------

cl::Kernel   *cluLoadKernel(cl::Program *prg,const char *kname)
{
  cl_int err;
  cl::Kernel *krnl;

  // load kernel (a program may have multiple kernels)
  krnl = new cl::Kernel(*prg, kname, &err);
  if (err != CL_SUCCESS) {
    char str[2048];
    prg->getBuildInfo(clu_Devices[0],CL_PROGRAM_BUILD_LOG,str);
    cerr << str << endl;
  }
  cluCheckError(err, "Kernel::Kernel()");

  return krnl;
}

// ----------------------------------------------------------

double cluDisplayEventMilliseconds(const char *msg,const cl::Event& ev)
{
  cl_int err;
  // get performance info
  long long start, end;
  double total;
  err = ev.getProfilingInfo( CL_PROFILING_COMMAND_START, &start );
  cluCheckError(err, "clGetEventProfilingInfo");
  err = ev.getProfilingInfo( CL_PROFILING_COMMAND_END, &end );
  cluCheckError(err, "clGetEventProfilingInfo");
  total = (double)(end - start) / 1e6;
  fprintf(stderr,"[%s] %5.2f msecs.\n", msg, total);
  return total;
}

// ----------------------------------------------------------

double        cluEventMilliseconds(const cl::Event& ev)
{
  cl_int err;
  // get performance info
  long long start, end;
  double total;
  err = ev.getProfilingInfo( CL_PROFILING_COMMAND_START, &start );
  cluCheckError(err, "clGetEventProfilingInfo");
  err = ev.getProfilingInfo( CL_PROFILING_COMMAND_END, &end );
  cluCheckError(err, "clGetEventProfilingInfo");
  total = (double)(end - start) / 1e6;
  return total;
}

// ----------------------------------------------------------

typedef long long      t_time;

#ifndef WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

t_time cluCPUMilliseconds()
{
#ifdef WIN32

  static bool   init = false;
  static t_time freq;
  if (!init) {
    init = true;
    LARGE_INTEGER lfreq;
    QueryPerformanceFrequency(&lfreq);
    freq = t_time(lfreq.QuadPart);
  }
  LARGE_INTEGER tps;
  QueryPerformanceCounter(&tps);
  return t_time(t_time(tps.QuadPart)*1000/freq);

#else

  struct timeval        now;
  uint                  ticks;
  static struct timeval start;

  static bool           init=false;
  if (!init) { // TODO / FIXME move into a global timer init ?
    gettimeofday(&start, NULL);
    init=true;
  }

  gettimeofday(&now, NULL);
  uint ms=uint((now.tv_sec-start.tv_sec)*1000+(now.tv_usec-start.tv_usec)/1000);
  return (ms);

#endif
}
// ----------------------------------------------------------
