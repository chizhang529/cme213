#include "simParams.h"
#include <fstream>
#include <iostream>
#include <cassert>
#include <stdlib.h>

simParams::simParams(const char* filename) {
    std::ifstream ifs(filename);

    if(!ifs.good()) {
        std::cerr << "Couldn't open parameter file!" << std::endl;
        exit(1);
    }

    ifs >> nx_ >> ny_;
    assert(nx_ > 0);
    assert(ny_ > 0);
    ifs >> lx_ >> ly_;
    assert(lx_ > 0);
    assert(ly_ > 0);
    ifs >> iters_;
    assert(iters_ >= 0);
    ifs >> order_;
    assert(order_ == 2 || order_ == 4 || order_ == 8);

    ifs.close();

    borderSize_ = 0;

    if(order_ == 2) {
        borderSize_ = 1;
    } else if(order_ == 4) {
        borderSize_ = 2;
    } else if(order_ == 8) {
        borderSize_ = 4;
    }

    assert(borderSize_ == 1 || borderSize_ == 2 || borderSize_ == 4);

    gx_ = nx_ + 2 * borderSize_;
    gy_ = ny_ + 2 * borderSize_;

    assert(gx_ > 2 * borderSize_);
    assert(gy_ > 2 * borderSize_);

    dx_ = lx_ / (gx_ - 1);
    dy_ = ly_ / (gy_ - 1);

    calcDtCFL();

}

void simParams::calcDtCFL() {
    //check cfl number and make sure it is ok
    if(order_ == 2) {
        //make sure we come in just under the limit
        dt_ = (.5 - .01) * (dx_ * dx_ * dy_ * dy_) / (dx_ * dx_ + dy_ * dy_);
        xcfl_ = (dt_) / (dx_ * dx_);
        ycfl_ = (dt_) / (dy_ * dy_);
    } else if(order_ == 4) {
        dt_ = (.5 - .01) * (12 * dx_ * dx_ * dy_ * dy_) / (16 *
                (dx_ * dx_ + dy_ * dy_));
        xcfl_ = (dt_) / (12 * dx_ * dx_);
        ycfl_ = (dt_) / (12 * dy_ * dy_);
    } else if(order_ == 8) {
        dt_ = (.5 - .01) * (5040 * dx_ * dx_ * dy_ * dy_) / (8064 *
                (dx_ * dx_ + dy_ * dy_));
        xcfl_ = (dt_) / (5040 * dx_ * dx_);
        ycfl_ = (dt_) / (5040 * dy_ * dy_);
    } else {
        std::cerr << "Unsupported discretization order." << std::endl;
        exit(1);
    }

    assert(xcfl_ > 0);
    assert(ycfl_ > 0);
}

size_t simParams::calcBytes() const {
    int stencilWords = 0;

    assert(order_ == 2 || order_ == 4 || order_ == 8);

    if(order_ == 2) {
        stencilWords = 6;
    } else if(order_ == 4) {
        stencilWords = 10;
    } else if(order_ == 8) {
        stencilWords = 18;
    }

    return (size_t)iters_ * nx_ * ny_ * stencilWords * sizeof(float);
}
