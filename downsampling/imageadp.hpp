#ifndef __IMAGEADP_H__
#define __IMAGEADP_H__

typedef struct {
    unsigned char* data;
    unsigned width;
    unsigned height;
} image_t;


bool loadFromFile(const char *filename, image_t &img);
bool saveToFile(const char *filename, const image_t &img);
#endif // __IMAGEADP_H__