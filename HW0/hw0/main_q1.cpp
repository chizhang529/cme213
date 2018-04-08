#include <iostream>
#include <utility>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <chrono>
#include <stdexcept>

#define SIZE 1000000
#define A_SCALE 1.0
#define B_SCALE 2.0

void swapData(double *a, double *b)
{
    double *c = (double *)std::malloc(SIZE*sizeof(double));
    if (c == nullptr)
        throw std::runtime_error("Memory cannot be allocated.");
    // swap
    auto start = std::chrono::high_resolution_clock::now();
    std::memcpy(c, a, SIZE*sizeof(double));
    std::memcpy(a, b, SIZE*sizeof(double));
    std::memcpy(b, c, SIZE*sizeof(double));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> time = end - start;
    std::cout << "Time to swap data: " << time.count() << "us" << std::endl;
    // free memory
    std::free(c);
}

void testDataSwap(double *a, double *b)
{
    std::cout << "Testing data swapping... ";
    for (int i = 0; i < SIZE; ++i) {
        assert(a[i] == B_SCALE * i);
        assert(b[i] == A_SCALE * i);
    }
    std::cout << "Done" << std::endl;
}

void swapPointer(double **a_ptr, double **b_ptr)
{
    auto start = std::chrono::high_resolution_clock::now();
    double *c = *a_ptr;
    *a_ptr = *b_ptr;
    *b_ptr = c;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> time = end - start;
    std::cout << "Time to swap pointers: " << time.count() << "us" << std::endl;
}

void testPointerSwap(std::pair<double *, double *> prev, std::pair<double *, double *> curr)
{
    // std::cout << "Before swapping, a: " << prev.first << " b: " << prev.second << std::endl;
    // std::cout << "After swapping, a: " << curr.first << " b: " << curr.second << std::endl;
    std::cout << "Testing pointer swapping... ";
    assert(prev.first == curr.second);
    assert(prev.second == curr.first);
    std::cout << "Done" << std::endl;
}

int main(int argc, char** argv)
{
    double *a = (double *)std::malloc(SIZE*sizeof(double));
    double *b = (double *)std::malloc(SIZE*sizeof(double));
    if (a == nullptr || b == nullptr)
        throw std::runtime_error("Memory cannot be allocated.");
    // initialize data
    for (int i = 0; i < SIZE; ++i) {
        a[i] = A_SCALE * i;
        b[i] = B_SCALE * i;
    }

    // swap data of two memory chunks
    swapData(a, b);
    testDataSwap(a, b);

    // swap values of two pointers
    std::pair<double *, double *> prev = std::make_pair(a, b);
    swapPointer(&a, &b);
    std::pair<double *, double *> curr = std::make_pair(a, b);
    testPointerSwap(prev, curr);

    // free memory
    std::free(a);
    std::free(b);
    return 0;
}
