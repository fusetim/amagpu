__kernel void matmul(__global int const * matA,
                    __global int const * matBtr,
                    __global int * matC,
                    const unsigned int N) {
    // Get global thread IDs
    const unsigned int id = get_global_id(0);
    const unsigned int row = id / N;
    const unsigned int col = id % N;

    // Compute element C[row, col]
    int sum = 0;
    for (unsigned int k = 0; k < N; k++) {
        sum += matA[row * N + k] * matBtr[col * N + k];
    }
    matC[row * N + col] = sum;
}

__kernel void diag(__global int const * matA,
                   __global int * diagA,
                   const unsigned int N) {
    // Get global thread ID
    const unsigned int i = get_global_id(0);

    // Extract diagonal element
    diagA[i] = matA[i * N + i];
}

// Kernel to perform a sum reduction on an array
// Will divide sum step elements into step parts
__kernel void sumStep(__global int const * src,
                   __global int * result,
                   const unsigned int N, 
                   const unsigned int step) {
    // Get global thread ID - also the index of the result element
    const unsigned int i = get_global_id(0);

    // Compute sum for this step
    int sum = 0;
    for (unsigned int j = 0; j < step; j++) {
        unsigned int idx = i * step + j;
        if (idx < N) {
            sum += src[idx];
        }
    }
    result[i] = sum;
}