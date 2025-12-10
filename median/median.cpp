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
#include <algorithm>

// #define GPU_DEBUG_PRINT

// ----------------------------------------------------------
typedef struct {
    unsigned char* data;
    unsigned width;
    unsigned height;
    unsigned channels;
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
    img.channels = 4; // RGBA
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

/// Clamp an integer value between min and max 
int iclamp(int value, int min, int max)
{
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/// Compute the median of the pixel values in the window centered at (x, y)
/// for a specific channel with the provided radius
char pixel_median(const image_t &img, int x, int y, unsigned int radius, unsigned int channel)
{
    const unsigned int diameter = 2 * radius + 1;
    const unsigned int windowSize = diameter * diameter;
    char* window = new char[windowSize];

    // Extract the window
    for (unsigned int wy = 0; wy < diameter; wy++)
    {
        for (unsigned int wx = 0; wx < diameter; wx++)
        {
            int imgX = iclamp(x + wx - radius, 0, img.width - 1);
            int imgY = iclamp(y + wy - radius, 0, img.height - 1);
            window[wy * diameter + wx] = img.data[(imgY * img.width + imgX) * img.channels + channel];
        }
    }

    // Sort the window to find the median
    std::sort(window, window + windowSize);
    char median = window[windowSize / 2];
    delete[] window;
    return median;
}

/// Apply a median filter (CPU) with given radius on the input image and store the result in the output image.
void cpu_median_filter(const image_t &input, image_t &output, unsigned int radius)
{
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data = new unsigned char[output.width * output.height * output.channels];

    for (unsigned int y = 0; y < input.height; y++)
    {
        for (unsigned int x = 0; x < input.width; x++)
        {
            for (unsigned int c = 0; c < input.channels; c++)
            {
                char median = pixel_median(input, x, y, radius, c);
                output.data[(y * output.width + x) * output.channels + c] = median;
            }
        }
    }
}

// Apply a median filter (GPU) with given radius on the input image and store the result in the output image.
void gpu_median_filter(const image_t &input, image_t &output, unsigned int radius)
{
    // Init / Load the kernel
    const char *clu_File = SRC_PATH "median.cl";
	cl::Program *program = cluLoadProgram(clu_File);
	cl::Kernel *kernel = cluLoadKernel(program, "median");

    // Prepare buffers
    const size_t imgSize = input.width * input.height * input.channels * sizeof(char);

    cl::Buffer inputBuf = cl::Buffer(
		*clu_Context,
		CL_MEM_READ_ONLY,
		imgSize);
    cl::Buffer outputBuf = cl::Buffer(
        *clu_Context,
        CL_MEM_WRITE_ONLY,
        imgSize);

    size_t actualSize = inputBuf.getInfo<CL_MEM_SIZE>();
    std::cout << "Input buffer size: " << actualSize << " bytes\n";
    actualSize = outputBuf.getInfo<CL_MEM_SIZE>();
    std::cout << "Output buffer size: " << actualSize << " bytes\n";

    // Upload input image to GPU
    int err = clu_Queue->enqueueWriteBuffer(
        inputBuf,
        CL_TRUE,
        0,
        input.width * input.height * input.channels * sizeof(char),
        input.data);
    assert(err == CL_SUCCESS);
    cout << "Input image uploaded to GPU." << endl;

    // Precompute the window (for local-memory based sort)
    const unsigned int diameter = 2 * radius + 1;
    const unsigned int windowSize = diameter * diameter;

    // Set kernel arguments
    kernel->setArg(0, inputBuf); // Input image buffer -- Global
    kernel->setArg(1, outputBuf); // Output image buffer -- Global
    kernel->setArg(2, cl::Local(windowSize * sizeof(char))); // Local memory for window
    kernel->setArg(3, input.width);
    kernel->setArg(4, input.height);
    kernel->setArg(5, input.channels);
    kernel->setArg(6, radius);

    // Launch the kernel
    cl::NDRange globalSize(input.width * input.height * input.channels * windowSize);
    cl::NDRange localSize(windowSize);
    err = clu_Queue->enqueueNDRangeKernel(
        *kernel,
        cl::NullRange,
        globalSize,
        localSize);
    assert(err == CL_SUCCESS);
    cout << "Kernel setup." << endl;

    // Read back the result
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data = new unsigned char[output.width * output.height * output.channels];

    cout << "Reading back the result from GPU..." << endl;
    err = clu_Queue->enqueueReadBuffer(
        outputBuf,
        CL_TRUE,
        0,
        output.width * output.height * output.channels * sizeof(char),
        output.data);
    cout << "err code: " << err << endl;
    assert(err == CL_SUCCESS);
    cout << "Output image downloaded from GPU." << endl;
    
    // Cleanup
    delete kernel;
    delete program;
}

int main(int argc, char **argv)
{
	// Initialize OpenCL
	cluInit();

	// Get the input file and output directory from args
	if (argc < 4)
	{
		cerr << "Usage: " << argv[0] << " <radius k> <input file> <output dir>" << endl;
		return 1;
	}
    
    // Check that radius is actually a positive integer
    for (int k = 0; argv[1][k] != '\0'; k++)
    {
        if (!isdigit(argv[1][k]))
        {
            cerr << "Radius must be a positive integer!" << endl;
            cerr << "Usage: " << argv[0] << " <radius k> <input file> <output dir>" << endl;
            return 1;
        }
    }

    const unsigned int radius = atoi(argv[1]);
	const char *inputFile = argv[2];
	const char *outputDir = argv[3];

    // Check if r is reasonable
    if (radius > 20) {
        cerr << "Radius must be a positive integer between 1 and 20!" << endl;
        return 1;
    }

	// Load input image
	image_t inputImage;
	cout << "Loading input image: " << inputFile << endl;
	if (!loadFromFile(inputFile, inputImage))
	{
		cerr << "Error loading input image!" << endl;
		return 1;
	}
	cout << "Input image size: " << inputImage.width << " x " << inputImage.height << endl;

    // Apply median filter on CPU
    image_t outputCPUImage;
    cout << "Applying median filter with radius " << radius << " on CPU..." << endl;
    auto startCPU = std::chrono::high_resolution_clock::now();
    cpu_median_filter(inputImage, outputCPUImage, radius);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = endCPU - startCPU;
    cout << "CPU median filter completed in " << cpu_duration.count() << " seconds." << endl;

    // Save output image
    std::string outputCPUFile = std::string(outputDir) + "/output_radius_cpu_" + std::to_string(radius) + ".png";
    cout << "Saving output image to: " << outputCPUFile << endl;
    if (!saveToFile(outputCPUFile.c_str(), outputCPUImage))
    {
        cerr << "Error saving output image!" << endl;
        return 1;
    }

    image_t outputGPUImage;
    cout << "Applying median filter with radius " << radius << " on GPU..." << endl;
    auto startGPU = std::chrono::high_resolution_clock::now();
    gpu_median_filter(inputImage, outputGPUImage, radius);
    auto endGPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = endGPU - startGPU;
    cout << "GPU median filter completed in " << gpu_duration.count() << " seconds." << endl;

    // Save output image
    std::string outputGPUFile = std::string(outputDir) + "/output_radius_gpu_" + std::to_string(radius) + ".png";
    cout << "Saving output image to: " << outputGPUFile << endl;
    if (!saveToFile(outputGPUFile.c_str(), outputGPUImage))
    {
        cerr << "Error saving output image!" << endl;
        return 1;
    }

	// Free the image buffer
    free(outputCPUImage.data);
    free(outputGPUImage.data);
	free(inputImage.data);
	return 0;
}