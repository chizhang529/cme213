#ifndef SIM_PARAMS_H__
#define SIM_PARAMS_H__
#include <stdlib.h>

class simParams {
  public:
    simParams(const char* filename); //parse command line
    //does no error checking

    int    nx()         const {
        return nx_;
    }
    int    ny()         const {
        return ny_;
    }
    int    gx()         const {
        return gx_;
    }
    int    gy()         const {
        return gy_;
    }
    double lx()         const {
        return lx_;
    }
    double ly()         const {
        return ly_;
    }
    int    iters()      const {
        return iters_;
    }
    double dx()         const {
        return dx_;
    }
    double dy()         const {
        return dy_;
    }
    int    order()      const {
        return order_;
    }
    int    borderSize() const {
        return borderSize_;
    }
    double xcfl()       const {
        return xcfl_;
    }
    double ycfl()       const {
        return ycfl_;
    }
    double dt()         const {
        return dt_;
    }
    size_t calcBytes()  const;

  private:
    int    nx_, ny_;     //number of grid points in each dimension
    int    gx_, gy_;     //number of grid points including halos
    double lx_, ly_;     //extent of physical domain in each dimension
    double dt_;          //timestep
    int    iters_;       //number of iterations to do
    double dx_, dy_;     //size of grid cell in each dimension
    double xcfl_, ycfl_; //cfl numbers in each dimension
    int    order_;       //order of discretization
    int    borderSize_;  //number of halo points

    void calcDtCFL();
};

#endif
