/// Median filter OpenCL kernel
///
/// Applies a median filter to an input image with a specified radius.
///
/// Image layout is assumed to be in HWC (Height-Width-Channels) format.
/// Each work-group processes one pixel (of a channel) in the output image.
/// Each work-group is composed of D*D work-items, where D is the diameter of the filter window (2*radius + 1).
/// 
/// Responsability of a work item:
/// - Load its pixel value into the group local memory.
/// - Synchronize with other work-items in the group.
/// - Perform a local sort to find the median value (odd-even sort) using barrier for group sync.
/// - Synchronize again with the other work items.
/// - The work-item at the center of the group writes the median value to the output image
__kernel void median(
    __global char const * input,
    __global char * output,
    __local char * localMem,
    const unsigned int width,
    const unsigned int height,
    const unsigned int channels,
    const unsigned int radius
) {
    // Work-items number should be equal to the diameter squared
    const unsigned int diameter = 2 * radius + 1;
    const unsigned int windowSize = diameter * diameter;
    const unsigned int workerId = get_local_id(0);
    const unsigned int groupId = get_group_id(0);

    const unsigned int channel = groupId % channels;
    const unsigned int pixelIndex = groupId / channels;
    const unsigned int x = pixelIndex % width;
    const unsigned int y = pixelIndex / width;

    const unsigned int dx = (workerId % diameter) - radius;
    const unsigned int dy = (workerId / diameter) - radius;

    // Load pixel value into local memory
    if (workerId < windowSize) { 
        localMem[workerId] = input[((y+dy) * width + (x+dx)) * channels + channel];
    }
    //Synchronize work-items in the group
    barrier(CLK_LOCAL_MEM_FENCE);

    // Odd-even sort to find the median
    for (unsigned int i = 0; i < windowSize; i++) {
        // If this worker is in the correct phase (odd/even)
        if (workerId % 2 == i % 2) {
            if (workerId > 0) {
                // Compare with the left neighbor
                char leftValue = localMem[workerId - 1];
                char currentValue = localMem[workerId];
                if (currentValue < leftValue) {
                    // Swap values
                    char temp = localMem[workerId];
                    localMem[workerId] = localMem[workerId - 1];
                    localMem[workerId - 1] = temp;
                }
            }
        }
        // Wait for all work-items to reach the end of step i
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // At that point, the median is at the center of the sorted window
    if (workerId == 0) {
        if (channel != 3) {
            output[(y * width + x) * channels + channel] = localMem[windowSize / 2];
        } else {
            output[(y * width + x) * channels + channel] = input[(y * width + x) * channels + channel];
        }
        //output[(y * width + x) * channels + channel] = (char) workerId % 256;
    }
}