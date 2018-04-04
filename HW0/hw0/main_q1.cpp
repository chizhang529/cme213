#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {
    double *a = (double *)std::malloc(1000000*sizeof(double));
    double *b = (double *)std::malloc(1000000*sizeof(double));
    std::cout << "Before swapping, a: " << a << " b: " << b << std::endl;

    // swap values of two pointers
    double *c = a;
    a = b;
    b = c;
    std::cout << "After swapping, a: " << a << " b: " << b << std::endl;

    // free memory
    std::free(a);
    std::free(b);
    return 0;
}
