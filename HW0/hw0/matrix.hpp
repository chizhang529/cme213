#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstdlib>

template <class T>
class Matrix {
public:
    virtual int l0_norm() = 0;
    virtual size_t size() = 0;
    virtual T &operator ()(const int i, const int j) = 0;
};

#endif  /* MATRIX_HPP */
