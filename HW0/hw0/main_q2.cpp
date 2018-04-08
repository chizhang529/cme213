#include <iostream>
#include <cassert>
#include "matrix_lt.hpp"

int main(int argc, char** argv) {
    MatrixLt<double> mat1(3);
    mat1.print();
    std::cout << "L0 Norm: " << mat1.l0_norm() << std::endl;

    mat1(0, 0) = 3.1;
    mat1(2, 1) = 4.2;
    mat1(1, 1) = 5.3;
    mat1.print();
    std::cout << "L0 Norm: " << mat1.l0_norm() << std::endl;
    return 0;
}
