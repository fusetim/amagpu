__kernel void cumsum(__global int *inbuf,
                    __global int *outbuf,
                    const unsigned int step) {
  unsigned int gid = (unsigned int) get_global_id(0);
  
  //step 0 means we have to calculate S(xi-1 to xi)
  //step k means we have to calculate S(xi-k-1 to xi)

  unsigned int k = gid - step + 1;
  if (k <= 0) {
    // We just copy the input to output for the first step+1 elements
    outbuf[gid] = inbuf[gid];
  } else {
    // For step 0, yi <- xi-1 + xi
    // For step 1, zi <- yi-2 + yi = xi-3 + xi-2 + xi-1 + xi
    // For step k, qi <- pi-k-1 + pi
    outbuf[gid] = inbuf[k - 1] + inbuf[gid];
  }
}