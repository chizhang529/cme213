#ifndef MATRIX_LT_HPP
#define MATRIX_LT_HPP

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "matrix.hpp"

template <class T>
class MatrixLt : public Matrix<T> {
public:
    // constructor
    MatrixLt() {};
    MatrixLt(const size_t sz);
    // destructor
    ~MatrixLt() {};

    int l0_norm();
    size_t size();
    T &operator ()(const unsigned int i, const unsigned int j);
    void print();

private:
    size_t sz;
    std::vector<T> data;
    // no actual data stored for upper triangular elements
    T ZERO = 0.0;
};

template <typename T>
MatrixLt<T>::MatrixLt(const size_t sz)
{
    this->sz = sz;
    data = std::vector<T>(sz * (sz + 1) / 2, 0);
}

template <typename T>
bool non_zero(const T value)
{
    return std::abs(value) > 1e-5;
}

template <typename T>
int MatrixLt<T>::l0_norm()
{
    return std::count_if(data.begin(), data.end(), non_zero<T>);
}

template <typename T>
size_t MatrixLt<T>::size()
{
    return sz;
}

template <typename T>
T &MatrixLt<T>::operator ()(const unsigned int i, const unsigned int j)
{
    if (i < 0 || i >= sz || j < 0 || j >= sz)
        throw std::out_of_range("The matrix index is out of range.");
    if (i < j)
        throw std::runtime_error("Not allowed to access upper triangular zero-value elements.");

    size_t index = i * (i + 1) / 2 + j;
    return (i >= j) ? data[index] : ZERO;
}

template <typename T>
void MatrixLt<T>::print()
{
    for (size_t i = 0; i < sz; ++i) {
        std::cout << '[';
        for (size_t j = 0; j < sz; ++j) {
            std::cout << ((i >= j) ? data[i * (i + 1) / 2 + j] : ZERO);
            if (j != sz - 1)
                std::cout << "\t";
        }
        std::cout << ']' << std::endl;
    }
}

#endif /* MATRIX_LT_HPP */
