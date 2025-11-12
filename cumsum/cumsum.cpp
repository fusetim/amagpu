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
	const char *clu_File = SRC_PATH "cumsum.cl";
	// Initialize OpenCL
	cluInit();

	// Load Program
	cl::Program *program = cluLoadProgram(clu_File);
	cl::Kernel *kernel = cluLoadKernel(program, "cumsum");

	// Prepare buffers
	int n = 4;
	cl::Buffer *buf1;
	cl::Buffer *buf2;
	buf1 = new cl::Buffer(
		*clu_Context,
		CL_MEM_READ_WRITE,
		sizeof(int) * n);
	buf2 = new cl::Buffer(
		*clu_Context,
		CL_MEM_READ_WRITE,
		sizeof(int) * n);

	// Intialize input data
	int *inputData = new int[n];
	for (int i = 0; i < n; i++)
	{
		inputData[i] = i + 1; // For simplicity, all ones
	}

	// Copy input image to buf1
	clu_Queue->enqueueWriteBuffer(
		*buf1,
		CL_TRUE,
		0,
		sizeof(int) * n,
		inputData);

	// Recursively cumsum
	cl::Buffer *inputBuf = buf1;
	cl::Buffer *outputBuf = buf2;
	unsigned int steps = (unsigned int)log2((double)n);
	for (unsigned int k = 0; k < steps; k++)
	{
		cout << "Cumsum step " << k << endl;
		// Set kernel arguments
		kernel->setArg(0, *inputBuf);
		kernel->setArg(1, *outputBuf);
		kernel->setArg(2, (unsigned int)(1 << k));

		const unsigned int globalSize = n;
		cl::NDRange global(globalSize);
		// Execute the kernel
		clu_Queue->enqueueNDRangeKernel(
			*kernel,
			cl::NullRange,
			global,
			cl::NullRange);

		// Ping-pong buffers
		cl::Buffer *temp = inputBuf;
		inputBuf = outputBuf;
		outputBuf = temp;
	}

	// Read back the result
	int *outputData = new int[n];
	clu_Queue->enqueueReadBuffer(
		*inputBuf,
		CL_TRUE,
		0,
		sizeof(int) * n,
		outputData);

	// Print the result
	cout << "Input Data: " << endl;
	for (int i = 0; i < n; i++)
	{
		cout << inputData[i] << " ";
	}
	cout << endl << "Output Data: " << endl;
	for (int i = 0; i < n; i++)
	{
		cout << outputData[i] << " ";
	}


	return 0;
}