#include <cassert>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <utility>
#include <stdexcept>
#include "matrix_lt.hpp"

const std::string INVALID_SIZE_EXP = "Invalid matrix size: should be larger than zero.";
const std::string OUT_OF_RANGE_EXP = "The matrix index is out of range.";
const std::string UPPER_TRI_ACCESS_EXP = "Not allowed to access upper triangular zero-value elements.";

void testInitialization()
{
    std::cout << "Test 1: initializing MatrixLt objects...";
    // test invalid matrices
    try {
        MatrixLt<double> invalid_mat1(0);
    } catch (const std::runtime_error &e) {
        assert(e.what() == INVALID_SIZE_EXP);
    }

    try {
        MatrixLt<double> invalid_mat2(-1);
    } catch (const std::exception &e) {
        assert(e.what() == INVALID_SIZE_EXP);
    }

    // test different data types (floating points and integers)
    // and different matrix sizes
    MatrixLt<float> mat1(2);
    MatrixLt<double> mat2(4);
    MatrixLt<char> mat3(8);
    MatrixLt<short> mat4(16);
    MatrixLt<int> mat5(32);
    MatrixLt<long> mat6(64);
    MatrixLt<long long> mat7(128);

    assert(mat1.l0_norm() == 0 && mat1.size() == 2);
    assert(mat2.l0_norm() == 0 && mat2.size() == 4);
    assert(mat3.l0_norm() == 0 && mat3.size() == 8);
    assert(mat4.l0_norm() == 0 && mat4.size() == 16);
    assert(mat5.l0_norm() == 0 && mat5.size() == 32);
    assert(mat6.l0_norm() == 0 && mat6.size() == 64);
    assert(mat7.l0_norm() == 0 && mat7.size() == 128);

    // test large matrix
    MatrixLt<double> mat8(10000);

    std::cout << "Done" << std::endl;
}

double random_num(double min, double max)
{
    srand(time(0));
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
}

void testDataRW()
{
    std::cout << "Test 2: read and write data...";
    MatrixLt<double> mat(3);
    // test indices that are out of range
    std::vector<std::pair<int, int>> indices {std::make_pair(-1, -2),
                                              std::make_pair(3, 1),
                                              std::make_pair(1, 3),
                                              std::make_pair(4, 5)};
    for (auto index_pair : indices) {
        try {
            mat(index_pair.first, index_pair.second);
        } catch (const std::exception &e) {
            assert(e.what() == OUT_OF_RANGE_EXP);
        }
    }

    // test accessing upper triangular elements
    try {
        mat(1, 2);
    } catch (const std::exception &e) {
        assert(e.what() == UPPER_TRI_ACCESS_EXP);
    }

    // test writing and reading data
    double num1 = random_num(1.0, 2.0);
    double num2 = random_num(4.0, 5.0);
    assert(mat(1, 1) == 0);
    assert(mat(2, 0) == 0);
    assert(mat.l0_norm() == 0);
    mat(1, 1) = num1;
    assert(mat(1, 1) == num1);
    assert(mat.l0_norm() == 1);
    mat(2, 0) = num2;
    assert(mat(2, 0) == num2);
    assert(mat.l0_norm() == 2);
    // swap values and test again
    double temp = mat(1, 1);
    mat(1, 1) = mat(2, 0);
    mat(2, 0) = temp;
    assert(mat(1, 1) == num2);
    assert(mat(2, 0) == num1);
    assert(mat.l0_norm() == 2);

    std::cout << "Done" << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "Running tests on MatrixLt class" << std::endl;

    testInitialization();
    testDataRW();

    std::cout << "All tests passed." << std::endl;
    return 0;


    // put element
    // read element
    // l0_norm
    // different types
    // out of range
    // empty matrices
}
