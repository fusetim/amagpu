__kernel void mainKernel(
    __global const int *a,
    __global const int *b,
    __global int *c,
    __local int *local_table)
{
    int id = get_global_id( 0 );
    c[ id ] = a[ id ] + b[ id ];

    // prefetch 
    int lid = get_local_id( 0 );
    local_table[lid] = c[id];

    // barrier to ensure all writes to local memory are done
    // note this only blocks work-items in the same work-group until local memory is coherent
    barrier( CLK_LOCAL_MEM_FENCE );
    // if you need global memory to be coherent, use: barrier( CLK_GLOBAL_MEM_FENCE );
    // however this will never block work-items from other work-groups

    if (lid == 0) {
        // only work-item 0 in each work-group does the cumulative some of the work group
        int local_sum = 0;
        int local_size = get_local_size(0);
        for (int i = 0; i < local_size; i++) {
            local_sum += local_table[i];
        }
        c[id] = local_sum;
    }
}