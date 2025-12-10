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

// #define GPU_DEBUG_PRINT

// ----------------------------------------------------------

const int matA_one[64] = {
	0,
	1,
	0,
	0,
	1,
	0,
	1,
	0,
	1,
	0,
	0,
	0,
	1,
	1,
	0,
	0,
	0,
	0,
	0,
	0,
	1,
	1,
	1,
	1,
	0,
	0,
	0,
	0,
	1,
	0,
	0,
	0,
	1,
	1,
	1,
	1,
	0,
	0,
	1,
	0,
	0,
	1,
	1,
	0,
	0,
	0,
	1,
	0,
	1,
	0,
	1,
	0,
	1,
	1,
	0,
	0,
	0,
	0,
	1,
	0,
	0,
	0,
	0,
	0,
};
const unsigned int expected_one = 4;

/// @brief Generate a random undirected graph adjacency matrix of size N
/// @param N Size of the graph (number of nodes)
int *genRandomGraphMat(const unsigned int N)
{
	int *mat = new int[N * N];
	// Initialize to zero
	for (unsigned int i = 0; i < N * N; i++)
	{
		mat[i] = 0;
	}
	// Fill upper triangular part randomly
	for (unsigned int i = 0; i < N; i++)
	{
		for (unsigned int j = i + 1; j < N; j++)
		{
			int edge = rand() % 2;
			mat[i * N + j] = edge;
			mat[j * N + i] = edge;
		}
	}
	return mat;
}

/// @brief Generate a complete graph adjacency matrix of size N
/// @param N Size of the graph (number of nodes)
/// @return Pointer to the adjacency matrix (must be deleted by caller)
int *genCompleteGraphMat(const unsigned int N)
{
	int *mat = new int[N * N];
	for (unsigned int i = 0; i < N; i++)
	{
		for (unsigned int j = 0; j < N; j++)
		{
			if (i != j)
			{
				mat[i * N + j] = 1;
			}
			else
			{
				mat[i * N + j] = 0;
			}
		}
	}
	return mat;
}

/// @brief Compute the nimber of edges in a complete graph of size N
/// @param N Size of the graph (number of nodes)
/// @return Number of edges
unsigned int computeCompleteGraphNimEdges(const unsigned int N)
{
	return (N * (N - 1) * (N - 2)) / 6;
}

/// @brief Print a square matrix
/// @param A Input matrix
/// @param N Size of the matrix (NxN)
void matPrint(const int *A, const unsigned int N)
{
	for (unsigned int i = 0; i < N; i++)
	{
		for (unsigned int j = 0; j < N; j++)
		{
			cout << setw(2) << (int)A[i * N + j] << " ";
		}
		cout << endl;
	}
}

/// @brief Compute the trace of a square matrix
/// @param A Input matrix
/// @param N Size of the matrix (NxN)
int matTraceCPU(const int *A, const unsigned int N)
{
	int trace = 0;
	for (unsigned int i = 0; i < N; i++)
	{
		trace += (int)A[i * N + i];
	}
	return trace;
}

/// @brief Matrix multiplication C = A * B
/// @param A First matrix
/// @param B Second matrix
/// @param C Result matrix
/// @param N Size of the matrices (NxN)
void matDotCPU(const int *A, const int *B, int *C, const unsigned int N)
{
	for (unsigned int i = 0; i < N; i++)
	{
		for (unsigned int j = 0; j < N; j++)
		{
			int sum = 0;
			for (unsigned int k = 0; k < N; k++)
			{
				sum += A[i * N + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
}

/// @brief Count the number of triangle in an undirected graph using CPU
/// @param mat Adjacency matrix of the graph
/// @param N Number of nodes in the graph
/// @return Number of triangles found
unsigned int computeCPUTri(const int *mat, const unsigned int N)
{
	// n_tri = trace(A^3) / 6

	// Compute A^2 then A^3
	int *matA2 = new int[N * N];
	int *matA3 = new int[N * N];
	matDotCPU(mat, mat, matA2, N);
	matDotCPU(matA2, mat, matA3, N);
	delete[] matA2;

	// Compute trace
	int trace = matTraceCPU(matA3, N);
	delete[] matA3;

	// Return number of triangles
	if (trace < 0)
	{
		cout << "computeCPUTri - error: trace is negative!" << endl;
		return 0;
	}
	return (unsigned int)(trace / 6);
}

/// @brief Count the number of triangle in an undirected graph using CPU and measure elapsed time
/// @param mat Adjacency matrix of the graph
/// @param N Number of nodes in the graph
/// @param elapsed Reference to store elapsed time (in millis)
unsigned int computeTimedCPUTri(const int *mat, const unsigned int N, double &elapsed)
{
	auto start = chrono::high_resolution_clock::now();
	unsigned int n_tri = computeCPUTri(mat, N);
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double, std::nano> diff = end - start;
	elapsed = diff.count() / 1e6; // Convert to milliseconds
	return n_tri;
}

int computeGPUTri(const uint8_t *mat, const unsigned int N, double &elapsed)
{
	// Reset elapsed time
	elapsed = 0.0;
	// Load Program
	const char *clu_File = SRC_PATH "trigraph.cl";
	cl::Program *program = cluLoadProgram(clu_File);
	cl::Kernel *kMatMul = cluLoadKernel(program, "matmul");
	cl::Kernel *kDiag = cluLoadKernel(program, "diag");
	cl::Kernel *kSumStep = cluLoadKernel(program, "sumStep");

	// Prepare buffers
	cl::Buffer *bufA;
	cl::Buffer *bufA2;
	cl::Buffer *bufA3;
	cl::Buffer *bufTraceOne;
	cl::Buffer *bufTraceTwo;

	bufA = new cl::Buffer(
		*clu_Context,
		CL_MEM_READ_ONLY,
		sizeof(int) * N * N);
	bufA2 = new cl::Buffer(
		*clu_Context,
		CL_MEM_READ_WRITE,
		sizeof(int) * N * N);
	bufA3 = new cl::Buffer(
		*clu_Context,
		CL_MEM_READ_WRITE,
		sizeof(int) * N * N);
	bufTraceOne = new cl::Buffer(
		*clu_Context,
		CL_MEM_READ_WRITE,
		sizeof(int) * N);
	bufTraceTwo = new cl::Buffer(
		*clu_Context,
		CL_MEM_READ_WRITE,
		sizeof(int) * N);

	// Copy input matrix to bufA
	cl::Event eventWriteA; // To measure
	clu_Queue->enqueueWriteBuffer(
		*bufA,
		CL_TRUE,
		0,
		sizeof(int) * N * N,
		mat,
		nullptr,
		&eventWriteA);
	elapsed += cluDisplayEventMilliseconds("Write Matrix A to GPU", eventWriteA);

	// Prepare commands
	// - Compute A^2
	kMatMul->setArg(0, *bufA);
	kMatMul->setArg(1, *bufA);
	kMatMul->setArg(2, *bufA2);
	kMatMul->setArg(3, N);
	const unsigned int globalSizeMatMul = N * N;
	cl::NDRange globalMatMul(globalSizeMatMul);
	cl::Event eventMatMulA2; // To measure
	clu_Queue->enqueueNDRangeKernel(
		*kMatMul,
		cl::NullRange,
		globalMatMul,
		cl::NullRange,
		nullptr,
		&eventMatMulA2);
	elapsed += cluDisplayEventMilliseconds("Matrix Multiplication A^2", eventMatMulA2);

// DEBUG - Read back A^2
#ifdef GPU_DEBUG_PRINT
	int *bufA2_host = new int[N * N];
	clu_Queue->enqueueReadBuffer(
		*bufA2,
		CL_TRUE,
		0,
		sizeof(int) * N * N,
		bufA2_host);
	cout << "A^2 matrix:" << endl;
	matPrint(bufA2_host, N);
	delete[] bufA2_host;
#endif
	// DEBUG - END Read back A^2

	// - Compute A^3
	kMatMul->setArg(0, *bufA2);
	kMatMul->setArg(1, *bufA);
	kMatMul->setArg(2, *bufA3);
	kMatMul->setArg(3, N);
	cl::Event eventMatMulA3;
	clu_Queue->enqueueNDRangeKernel(
		*kMatMul,
		cl::NullRange,
		globalMatMul,
		cl::NullRange,
		NULL,
		&eventMatMulA3);
	eventMatMulA3.wait();
	elapsed += cluDisplayEventMilliseconds("Matrix Multiplication A^3", eventMatMulA3);

// DEBUG - Read back A^3
#ifdef GPU_DEBUG_PRINT
	int *bufA3_host = new int[N * N];
	clu_Queue->enqueueReadBuffer(
		*bufA3,
		CL_TRUE,
		0,
		sizeof(int) * N * N,
		bufA3_host);
	cout << "A^3 matrix:" << endl;
	matPrint(bufA3_host, N);
	delete[] bufA3_host;
#endif
	// DEBUG - END Read back A^3

	// - Compute diag(A^3)
	kDiag->setArg(0, *bufA3);
	kDiag->setArg(1, *bufTraceOne);
	kDiag->setArg(2, N);
	const unsigned int globalSizeDiag = N;
	cl::Event eventDiag; // To measure
	cl::NDRange globalDiag(globalSizeDiag);
	clu_Queue->enqueueNDRangeKernel(
		*kDiag,
		cl::NullRange,
		globalDiag,
		cl::NullRange,
		nullptr,
		&eventDiag);
	eventDiag.wait();
	elapsed += cluDisplayEventMilliseconds("Diagonal Extraction", eventDiag);

// DEBUG - Read back diag(A^3)
#ifdef GPU_DEBUG_PRINT
	int *bufTrace_host = new int[N];
	clu_Queue->enqueueReadBuffer(
		*bufTraceOne,
		CL_TRUE,
		0,
		sizeof(int) * N,
		bufTrace_host);
	cout << "diag(A^3):" << endl;
	for (unsigned int i = 0; i < N; i++)
	{
		cout << setw(2) << bufTrace_host[i] << " ";
	}
	cout << endl;
	delete[] bufTrace_host;
#endif
	// DEBUG - END Read back diag(A^3)

	// - Compute sum of diag(A^3)
	int iter = 0;
	int step = 4;
	int sumSize = N;
	cl::Buffer *bufTraceRead = bufTraceOne;
	cl::Buffer *bufTraceWrite = bufTraceTwo;
	while (sumSize > 1)
	{
		kSumStep->setArg(0, *bufTraceRead);
		kSumStep->setArg(1, *bufTraceWrite);
		kSumStep->setArg(2, sumSize);
		kSumStep->setArg(3, step);
		const unsigned int globalSizeSum = (unsigned int)ceil((float)sumSize / (float)step);
		cl::NDRange globalSum(globalSizeSum);
		cl::Event eventSumStep;
		clu_Queue->enqueueNDRangeKernel(
			*kSumStep,
			cl::NullRange,
			globalSum,
			cl::NullRange,
			nullptr,
			&eventSumStep);
		elapsed += cluDisplayEventMilliseconds("Sum Step", eventSumStep);
		// Swap buffers
		bufTraceRead = (iter % 2 == 0) ? bufTraceTwo : bufTraceOne;
		bufTraceWrite = (iter % 2 == 0) ? bufTraceOne : bufTraceTwo;
		// Update sum size
		sumSize = globalSizeSum;
		iter++;
	}
	// Read back trace value
	int trace = 0;
	cl::Event eventReadTrace;
	clu_Queue->enqueueReadBuffer(
		*bufTraceRead,
		CL_TRUE,
		0,
		sizeof(int),
		&trace,
		nullptr,
		&eventReadTrace);
	elapsed += cluDisplayEventMilliseconds("Read Back Trace", eventReadTrace);

	// Clean up
	delete bufA;
	delete bufA2;
	delete bufA3;
	delete bufTraceOne;
	delete bufTraceTwo;

	// Return number of triangles
	if (trace < 0)
	{
		cout << "computeGPUTri - error: trace is negative!" << endl;
		return 0;
	}
	return (unsigned int)(trace / 6);
}

int main(int argc, char **argv)
{
	// Initialize OpenCL
	cluInit();

	// Step 1 - Test the matA_one matrix
	cout << "Testing small graph (8 nodes, 4 triangles)..." << endl;
	double gpuTime, cpuTime;

	// Compute number of triangles using CPU
	const unsigned int N = 8;
	unsigned int n_tri_cpu = computeCPUTri(matA_one, N);
	cout << "Number of triangles (CPU): " << n_tri_cpu;

	// Compute number of triangles using GPU
	unsigned int n_tri_gpu = computeGPUTri((const uint8_t *)matA_one, N, gpuTime);
	cout << "Number of triangles (GPU): " << n_tri_gpu << endl;

	assert(n_tri_cpu == expected_one);
	assert(n_tri_gpu == expected_one);

	// Step 2 - Test a complete graph of size N=2,4,8,16,32,64
	const unsigned int testSizes[] = {2, 4, 8, 16, 32, 64};
	const unsigned int nTestSizes = 6;
	for (unsigned int t = 0; t < nTestSizes; t++)
	{
		const unsigned int N = testSizes[t];
		cout << "Testing complete graph of size " << N << "..." << endl;

		// Generate complete graph adjacency matrix
		int *matComplete = genCompleteGraphMat(N);

		// Compute expected number of triangles
		const unsigned int expectedNimEdges = computeCompleteGraphNimEdges(N);

		cout << "Expected number of triangles: " << expectedNimEdges << endl;

		// Compute number of triangles using CPU
		unsigned int n_tri_cpu = computeCPUTri(matComplete, N);
		cout << "Number of triangles (CPU): " << n_tri_cpu;

		// Compute number of triangles using GPU
		unsigned int n_tri_gpu = computeGPUTri((const uint8_t *)matComplete, N, gpuTime);
		cout << "Number of triangles (GPU): " << n_tri_gpu << endl;

		assert(n_tri_cpu == expectedNimEdges);
		assert(n_tri_gpu == expectedNimEdges);

		// Clean up
		delete[] matComplete;
	}

	// Step 3 - Time for some randomness
	// Ensure the seed is fixed for reproducibility
	srand(12345);

	// Test with random undirected graph of size N=16
	unsigned int N_random = 16;
	for (int i = 0; i < 5; i++)
	{
		cout << "Testing random graph " << i << " of size " << N_random << "..." << endl;
		int *matRandom = genRandomGraphMat(N_random);
		unsigned int n_tri_random_cpu = computeCPUTri(matRandom, N_random);
		cout << "Number of triangles (CPU): " << n_tri_random_cpu;
		unsigned int n_tri_random_gpu = computeGPUTri((const uint8_t *)matRandom, N_random, gpuTime);
		cout << "Number of triangles (GPU): " << n_tri_random_gpu << endl;
		assert(n_tri_random_cpu == n_tri_random_gpu);
		delete[] matRandom;
	}

	// Test with random undirected graph of size N=64
	// N_random = 512;
	// for (int i = 0; i < 5; i++)
	// {
	// 	cout << "Testing random graph " << i << " of size " << N_random << "..." << endl;
	// 	int *matRandom = genRandomGraphMat(N_random);
	// 	// unsigned int n_tri_random_cpu = computeTimedCPUTri(matRandom, N_random, cpuTime);
	// 	// cout << "Number of triangles (CPU): " << n_tri_random_cpu << " (Time: " << cpuTime << " ms)";
	// 	unsigned int n_tri_random_gpu = computeGPUTri((const uint8_t *)matRandom, N_random, gpuTime);
	// 	cout << "Number of triangles (GPU): " << n_tri_random_gpu << " (Time: " << fixed << setprecision(15) << gpuTime << " ms)" << endl;
	// 	//assert(n_tri_random_cpu == n_tri_random_gpu);
	// 	delete[] matRandom;
	// }

	// Running the benchmark and saving the result to a file
	cout << "Running benchmark for random graphs of increasing size..." << endl;
	ofstream benchmarkFile("results.txt");

	benchmarkFile << "N,CPU_Time_ms,GPU_Time_ms" << endl;

	for (unsigned int N = 8; N <= 512; N *= 2)
	{
		cout << "Benchmarking random graph of size " << N << "..." << endl;

		for (unsigned int k = 0; k < 5; k++)
		{
			int *matRandom = genRandomGraphMat(N);

			// Time CPU computation
			unsigned int n_tri_random_cpu = computeTimedCPUTri(matRandom, N, cpuTime);
			cout << "Number of triangles (CPU): " << n_tri_random_cpu << " (Time: " << fixed << setprecision(15) << cpuTime << " ms)";

			// Time GPU computation
			unsigned int n_tri_random_gpu = computeGPUTri((const uint8_t *)matRandom, N, gpuTime);
			cout << "Number of triangles (GPU): " << n_tri_random_gpu << " (Time: " << fixed << setprecision(15) << gpuTime << " ms)" << endl;

			assert(n_tri_random_cpu == n_tri_random_gpu);

			// Save to file
			benchmarkFile << N << "," << fixed << setprecision(15) << cpuTime << "," << fixed << setprecision(15) << gpuTime << endl;

			delete[] matRandom;
		}
	}

	return 0;
}