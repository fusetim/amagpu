__kernel void downsample(__global unsigned char *inputImage,
                         __global unsigned char *outputImage,
                         const unsigned int side) {
  // x,y are coordinates in the output image
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);
  unsigned int input_width = side * 2;

  int idx = (y * side + x) * 4;

  // Simple downsampling by averaging 2x2 pixels
  unsigned int topleft = ((2 * y) * input_width + (2 * x)) * 4;
  unsigned int topright = ((2 * y) * input_width + (2 * x + 1)) * 4;
  unsigned int bottomleft = ((2 * y + 1) * input_width + (2 * x)) * 4;
  unsigned int bottomright = ((2 * y + 1) * input_width + (2 * x + 1)) * 4;

  unsigned int sumR = inputImage[topleft] + inputImage[topright] +
                      inputImage[bottomleft] + inputImage[bottomright];

  unsigned int sumG = inputImage[topleft + 1] + inputImage[topright + 1] +
                      inputImage[bottomleft + 1] + inputImage[bottomright + 1];

  unsigned int sumB = inputImage[topleft + 2] + inputImage[topright + 2] +
                      inputImage[bottomleft + 2] + inputImage[bottomright + 2];

  unsigned int sumA = inputImage[topleft + 3] + inputImage[topright + 3] +
                      inputImage[bottomleft + 3] + inputImage[bottomright + 3];

  outputImage[idx] = sumR / 4;
  outputImage[idx + 1] = sumG / 4;
  outputImage[idx + 2] = sumB / 4;
  outputImage[idx + 3] = sumA / 4;
}