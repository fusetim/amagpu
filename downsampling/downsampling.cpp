// ----------------------------------------------------------

// ----------------------------------------------------------


#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <chrono>

using namespace std;

// ----------------------------------------------------------

// We provide a small library so we can easily setup OpenCL
#include "clutils.h"

// ----------------------------------------------------------

int main(int argc, char **argv)
{
	const char *clu_File = SRC_PATH "base.cl";  // path to file containing OpenCL kernel(s) code

	// Initialize OpenCL
	cluInit();

	// After this call you have access to
	// clu_Context;      <= OpenCL context (pointer)
	// clu_Devices;      <= OpenCL device list (vector)
	// clu_Queue;        <= OpenCL queue (pointer)

	// Load Program
	cl::Program *program = cluLoadProgram(clu_File);
	cl::Kernel *kernel = cluLoadKernel(program, "mainKernel");

	// Allocate memory/buffers
	const int size = 32;
	const int local_size = 8;
	cl::Buffer a_buffer(*clu_Context, CL_MEM_READ_ONLY, sizeof(int) * size);
	cl::Buffer b_buffer(*clu_Context, CL_MEM_READ_ONLY, sizeof(int) * size);
	cl::Buffer c_buffer(*clu_Context, CL_MEM_WRITE_ONLY, sizeof(int) * size);

	// Create input data
	int *a = new int[size];
	int *b = new int[size];
	int *c = new int[size];
	for (int i = 0; i < size; i++) {
		a[i] = i;
		b[i] = 2;
	}

	// Transfer input data to device
	clu_Queue->enqueueWriteBuffer(a_buffer, CL_TRUE, 0, sizeof(int) * size, a);
	clu_Queue->enqueueWriteBuffer(b_buffer, CL_TRUE, 0, sizeof(int) * size, b);
	
	// Set the kernel arguments
	kernel->setArg(0, a_buffer);
	kernel->setArg(1, b_buffer);
	kernel->setArg(2, c_buffer);
	kernel->setArg(3, cl::__local(sizeof(int) * local_size)); // local memory argument

	// Launch the kernel
	cl::NDRange global(size);
	cl::NDRange local(local_size);
	cl_int clerr = clu_Queue->enqueueNDRangeKernel(*kernel, cl::NullRange, global, local);
	cluCheckError(clerr, "Failed to launch kernel");
	clu_Queue->finish();

	// Read back the results
	clu_Queue->enqueueReadBuffer(c_buffer, CL_TRUE, 0, sizeof(int) * size, c);

	// Print the results
	for (int i = 0; i < size; i++) {
		cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
	}

	delete[] a;
	delete[] b;
	delete[] c;
}
