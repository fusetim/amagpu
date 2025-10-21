#include "imageadp.hpp"
#include "lodepng.h"
#include <iostream>

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