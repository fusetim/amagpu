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
#include "lodepng.h"

// ----------------------------------------------------------

typedef struct {
    unsigned char* data;
    unsigned width;
    unsigned height;
} image_t;

bool loadFromFile(const char *filename, image_t &img)
{
    std::vector<unsigned char> image; // the raw pixels
    unsigned width, height;

    // decode
    unsigned error = lodepng::decode(image, width, height, filename);

    // if there's an error, display it
    if (error) {
        std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        return false;
    }

    img.width = width;
    img.height = height;
    img.data = new unsigned char[image.size()];
    std::copy(image.begin(), image.end(), img.data);
    return true;
}

bool saveToFile(const char *filename, const image_t &img)
{
    // encode
    unsigned error = lodepng::encode(filename, img.data, img.width, img.height);

    // if there's an error, display it
    if (error) {
        std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        return false;
    }
    return true;
}

/// Prepare the two ping-ponging buffers
bool prepareBuffers(cl::Buffer *&buf1, cl::Buffer *&buf2, const unsigned int n)
{
	// Create the two buffers for ping-ponging
	unsigned int side = 1 << n;
	buf1 = new cl::Buffer(
		*clu_Context,
		CL_MEM_READ_WRITE,
		sizeof(unsigned char) * side * side * 4);
	buf2 = new cl::Buffer(
		*clu_Context,
		CL_MEM_READ_WRITE,
		sizeof(unsigned char) * side * side * 4);
	return true;
}

bool saveDownsampled(const char *outputDir, const unsigned int n, image_t &img)
{
	stringstream ss;
	ss << outputDir << "/downsampled_" << setfill('0') << setw(2) << n << ".png";
	if (!saveToFile(ss.str().c_str(), img))
	{
		cerr << "Error saving output image!" << endl;
		return false;
	}
	return true;
}

int main(int argc, char **argv)
{
	const char *clu_File = SRC_PATH "downsampling.cl";
	// Initialize OpenCL
	cluInit();

	// Load Program
	cl::Program *program = cluLoadProgram(clu_File);
	cl::Kernel *kernel = cluLoadKernel(program, "downsample");

	// Get the input file and output directory from args
	if (argc < 3)
	{
		cerr << "Usage: " << argv[0] << " <input file> <output dir>" << endl;
		return 1;
	}
	const char *inputFile = argv[1];
	const char *outputDir = argv[2];

	// Load input image
	image_t inputImage;
	cout << "Loading input image: " << inputFile << endl;
	if (!loadFromFile(inputFile, inputImage))
	{
		cerr << "Error loading input image!" << endl;
		return 1;
	}

	// Check the image is indeed a square and width/height are power of two
	cout << "Input image size: " << inputImage.width << " x " << inputImage.height << endl;
	if (inputImage.width != inputImage.height)
	{
		cerr << "Error: input image is not square!" << endl;
		return 1;
	}
	unsigned size = inputImage.width;
	if ((size & (size - 1)) != 0)
	{
		cerr << "Error: input image size is not a power of two!" << endl;
		return 1;
	}
	unsigned n = (unsigned)log2((double)size);
	cout << "Image is square with size " << size << " = 2^" << n << endl;

	// Save the original image to output dir
	saveDownsampled(outputDir, n, inputImage);

	// Prepare buffers
	cl::Buffer *buf1;
	cl::Buffer *buf2;
	prepareBuffers(buf1, buf2, n);

	// Copy input image to buf1
	clu_Queue->enqueueWriteBuffer(
		*buf1,
		CL_TRUE,
		0,
		sizeof(unsigned char) * inputImage.width * inputImage.height * 4,
		inputImage.data);

	cl::Buffer* inputBuf = buf1;
	cl::Buffer* outputBuf = buf2;
	unsigned int k = n;
	image_t downsampled;
	downsampled.data = new unsigned char[inputImage.width * inputImage.height * 4];
	while (k > 2)
	{
		k--;
		// Downsampled one time
		downsampled.width = 1 << k;
		downsampled.height = 1 << k;

		// Set kernel arguments
		kernel->setArg(0, *inputBuf);
		kernel->setArg(1, *outputBuf);
		kernel->setArg(2, (unsigned int)downsampled.width);

		const unsigned int globalSize = downsampled.width;
		cl::NDRange global(globalSize, globalSize);
		// Execute the kernel
		clu_Queue->enqueueNDRangeKernel(
			*kernel,
			cl::NullRange,
			global,
			cl::NullRange);

		// Read back the result
		clu_Queue->enqueueReadBuffer(
			*outputBuf,
			CL_TRUE,
			0,
			sizeof(unsigned char) * downsampled.width * downsampled.height * 4,
			downsampled.data);

		// Save the downsampled image
		saveDownsampled(outputDir, k, downsampled);

		// Ping-pong buffers
		cl::Buffer* temp = inputBuf;
		inputBuf = outputBuf;
		outputBuf = temp;
	}

	// Free image data
	free(inputImage.data);
	free(downsampled.data);
	return 0;
}