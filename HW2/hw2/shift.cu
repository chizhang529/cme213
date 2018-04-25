/* Repeating from the tutorial, just in case you haven't looked at it.
   "kernels" or __global__ functions are the entry points to code that executes on the GPU.
   The keyword __global__ indicates to the compiler that this function is a GPU entry point.
   __global__ functions must return void, and may only be called or "launched" from code that
   executes on the CPU.
*/

typedef unsigned char uchar;
typedef unsigned int uint;

// This kernel implements a per element shift
// by naively loading one byte and shifting it
__global__ void shift_char(const uchar* input_array, uchar* output_array,
                           uchar shift_amount, uint array_length) {
    // TODO: fill in
}

//Here we load 4 bytes at a time instead of just 1
//to improve the bandwidth due to a better memory
//access pattern
__global__ void shift_int(const uint* input_array, uint* output_array,
                          uint shift_amount, uint array_length) {
    // TODO: fill in
}

//Here we go even further and load 8 bytes
//does it make a further improvement?
__global__ void shift_int2(const uint2* input_array, uint2* output_array,
                           uint shift_amount, uint array_length) {
    // TODO: fill in
}

//the following three kernels launch their respective kernels
//and report the time it took for the kernel to run

double doGPUShiftChar(const uchar* d_input, uchar* d_output,
                      uchar shift_amount, uint text_size, uint block_size) {
    // TODO: compute your grid dimensions

    event_pair timer;
    start_timer(&timer);

    // TODO: launch kernel

    check_launch("gpu shift cipher char");
    return stop_timer(&timer);
}

double doGPUShiftUInt(const uchar* d_input, uchar* d_output,
                      uchar shift_amount, uint text_size, uint block_size) {
    // TODO: compute your grid dimensions

    // TODO: compute 4 byte shift value

    event_pair timer;
    start_timer(&timer);

    // TODO: launch kernel

    check_launch("gpu shift cipher uint");
    return stop_timer(&timer);
}

double doGPUShiftUInt2(const uchar* d_input, uchar* d_output
                       , uchar shift_amount, uint text_size, uint block_size) {
    // TODO: compute your grid dimensions

    // TODO: compute 4 byte shift value

    event_pair timer;
    start_timer(&timer);

    // TODO: launch kernel

    check_launch("gpu shift cipher uint2");
    return stop_timer(&timer);
}
