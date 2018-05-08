#include "Grid.h"
#include <iostream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <fstream>

using std::setw;
using std::setprecision;
using std::endl;
using std::cout;

//construct the grid - set dimensions and allocate memory on host and device
Grid::Grid(int gx, int gy) : gx_(gx), gy_(gy) {
    //resize and set ICs
    hGrid_.resize(gx_ * gy_);

    cudaError_t err = cudaMalloc(&dGrid_, gx_ * gy_ * sizeof(float));

    if(err != cudaSuccess) {
        std::cerr << "Could not allocate memory for Grid!" << std::endl;
        exit(1);
    }
}

//host side memory will automatically freed in vector destructor
//we must manually free the device memory
Grid::~Grid() {
    cudaFree(dGrid_);
}

//copy constructor, copy both the host contents and gpu contents
Grid::Grid(const Grid& other) : gx_(other.gx()), gy_(other.gy()),
    hGrid_(other.hGrid_) {
    cudaError_t err = cudaMalloc(&dGrid_, gx_ * gy_ * sizeof(float));

    if(err != cudaSuccess) {
        std::cerr << "Error allocating Grid!" << std::endl;
        exit(1);
    }

    cudaMemcpy(dGrid_, other.dGrid_, gx_ * gy_ * sizeof(float),
               cudaMemcpyDeviceToDevice);
}

//copy from host -> device
void Grid::toGPU() {
    cudaError_t err = cudaMemcpy(dGrid_, &hGrid_[0], gx_ * gy_ * sizeof(float),
                                 cudaMemcpyHostToDevice);

    if(err != cudaSuccess) {
        std::cerr << "Error copying grid to GPU" << std::endl;
        exit(1);
    }
}

//copy from device -> host
void Grid::fromGPU() {
    cudaError_t err = cudaMemcpy(&hGrid_[0], dGrid_, gx_ * gy_ * sizeof(float),
                                 cudaMemcpyDeviceToHost);

    if(err != cudaSuccess) {
        std::cerr << "Error copying grid from GPU" << std::endl;
        exit(1);
    }
}

//swap two grids by exchanging pointers
//host and device pointers
//std::vector does this under the hood with a specialized version of swap
void Grid::swap(Grid& a, Grid& b) {
    std::swap(a.hGrid_, b.hGrid_);
    std::swap(a.dGrid_, b.dGrid_);
}

//save the host grid to a file for debugging / visualization
void Grid::saveStateToFile(const std::string& identifier) {
    std::stringstream ss;
    ss << "grid" << "_" << identifier << ".txt";
    std::ofstream ofs(ss.str().c_str());

    ofs << *this << std::endl;

    ofs.close();
}

std::ostream& operator<<(std::ostream& os, const Grid& grid) {
    os << setprecision(7);

    for(int y = 0; y < grid.gy(); ++y) {
        for(int x = 0; x < grid.gx(); x++) {
            os << setw(5) << x << " " << setw(5) << y << " " << setw(
                   15) << grid.hGrid_[x + grid.gx() * y] << endl;
        }

        os << endl;
    }

    os << endl;
    return os;
}
