/* Repeating from the tutorial, just in case you haven't looked at it.
   "kernels" or __global__ functions are the entry points to code that executes on the GPU.
   The keyword __global__ indicates to the compiler that this function is a GPU entry point.
   __global__ functions must return void, and may only be called or "launched" from code that
   executes on the CPU.
*/

typedef unsigned char uchar;
typedef unsigned int uint;
static const uint MAX_GRID_DIM = 65535;

// This kernel implements a per element shift
// by naively loading one byte and shifting it
__global__ void shift_char(const uchar* input_array, uchar* output_array,
                           uchar shift_amount, uint array_length) {
    uint bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint tid = bid * blockDim.x + threadIdx.x;
    if (tid < array_length)
        output_array[tid] = input_array[tid] + shift_amount;
}

//Here we load 4 bytes at a time instead of just 1
//to improve the bandwidth due to a better memory
//access pattern
__global__ void shift_int(const uint* input_array, uint* output_array,
                          uint shift_amount, uint array_length) {
    uint bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint tid = bid * blockDim.x + threadIdx.x;
    if (tid < array_length)
        output_array[tid] = input_array[tid] + shift_amount;
}

//Here we go even further and load 8 bytes
//does it make a further improvement?
__global__ void shift_int2(const uint2* input_array, uint2* output_array,
                           uint shift_amount, uint array_length) {
    uint bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint tid = bid * blockDim.x + threadIdx.x;
    if (tid < array_length) {
        output_array[tid].x = input_array[tid].x + shift_amount;
        output_array[tid].y = input_array[tid].y + shift_amount;
    }
}

//the following three kernels launch their respective kernels
//and report the time it took for the kernel to run
double doGPUShiftChar(const uchar* d_input, uchar* d_output,
                      uchar shift_amount, uint text_size, uint block_size) {
    // compute grid dimensions (NOTE: block size is 256 in tests)
    dim3 blocks_per_grid(1, 1);         // 2D grid
    dim3 threads_per_block(block_size); // 1D block
    // compute number of blocks needed
    uint num_blocks = ceil((float)text_size / (float)block_size);
    if (num_blocks > MAX_GRID_DIM) {
        blocks_per_grid.x = MAX_GRID_DIM;
        blocks_per_grid.y = ceil((float)num_blocks / (float)MAX_GRID_DIM);
    } else {
        blocks_per_grid.x = num_blocks;
    }

    event_pair timer;
    start_timer(&timer);

    // launch kernel
    shift_char<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, shift_amount, text_size);
    check_launch("gpu shift cipher char");
    return stop_timer(&timer);
}

double doGPUShiftUInt(const uchar* d_input, uchar* d_output,
                      uchar shift_amount, uint text_size, uint block_size) {
    // compute grid dimensions
    dim3 blocks_per_grid(1, 1);         // 2D grid
    dim3 threads_per_block(block_size); // 1D block
    // compute number of blocks needed
    uint num_blocks = ceil((float)text_size / (float)block_size);
    uint scale = sizeof(uint) / sizeof(uchar);
    num_blocks = ceil((float)num_blocks / (float)scale);
    if (num_blocks > MAX_GRID_DIM) {
        blocks_per_grid.x = MAX_GRID_DIM;
        blocks_per_grid.y = ceil((float)num_blocks / (float)MAX_GRID_DIM);
    } else {
        blocks_per_grid.x = num_blocks;
    }
    // compute 4 byte shift value
    uint shift = shift_amount | (shift_amount << 8) | (shift_amount << 16) | (shift_amount << 24);

    event_pair timer;
    start_timer(&timer);

    // launch kernel
    uint array_length = ceil((float)text_size / (float)scale);
    shift_int<<<blocks_per_grid, threads_per_block>>>((const uint *)d_input, (uint *)d_output,
                                                       shift, array_length);
    check_launch("gpu shift cipher uint");
    return stop_timer(&timer);
}

double doGPUShiftUInt2(const uchar* d_input, uchar* d_output
                       , uchar shift_amount, uint text_size, uint block_size) {
    // compute grid dimensions
    dim3 blocks_per_grid(1, 1);         // 2D grid
    dim3 threads_per_block(block_size); // 1D block
    // compute number of blocks needed
    uint num_blocks = ceil((float)text_size / (float)block_size);
    uint scale = sizeof(uint2) / sizeof(uchar);
    num_blocks = ceil((float)num_blocks / (float)scale);
    if (num_blocks > MAX_GRID_DIM) {
        blocks_per_grid.x = MAX_GRID_DIM;
        blocks_per_grid.y = ceil((float)num_blocks / (float)MAX_GRID_DIM);
    } else {
        blocks_per_grid.x = num_blocks;
    }
    // compute 4 byte shift value
    uint shift = shift_amount | (shift_amount << 8) | (shift_amount << 16) | (shift_amount << 24);

    event_pair timer;
    start_timer(&timer);

    // launch kernel
    uint array_length = ceil((float)text_size / (float)scale);
    shift_int2<<<blocks_per_grid, threads_per_block>>>((const uint2 *)d_input, (uint2 *)d_output,
                                                       shift, array_length);

    check_launch("gpu shift cipher uint2");
    return stop_timer(&timer);
}
