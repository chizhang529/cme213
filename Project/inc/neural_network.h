#ifndef UTILS_TWO_LAYER_NET_H_
#define UTILS_TWO_LAYER_NET_H_

#include <armadillo>
#include <cmath>
#include <iostream>

class NeuralNetwork {
public:
    const int num_layers = 2;
    // H[i] is the number of neurons in layer i (where i=0 implies input layer)
    std::vector<int> H;
    // Weights of the neural network
    // W[i] are the weights of the i^th layer
    std::vector<arma::mat> W;
    // Biases of the neural network
    // b[i] is the row vector biases of the i^th layer
    std::vector<arma::colvec> b;

    NeuralNetwork(std::vector<int> _H) {
        W.resize(num_layers);
        b.resize(num_layers);
        H = _H;

        for(int i = 0; i < num_layers; i++) {
            arma::arma_rng::set_seed(arma::arma_rng::seed_type(i));
            W[i] = 0.01 * arma::randn(H[i+1], H[i]);
            b[i].zeros(H[i+1]);
        }
    }
};

void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& bpcache);
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg);
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads);
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads);
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg = 0.0, const int epochs = 15,
           const int batch_size = 800, bool grad_check = false, int print_every = -1,
           int debug = 0);
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label);

void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg = 0.0, const int epochs = 15,
                    const int batch_size = 800, bool grad_check = false, int print_every = -1,
                    int debug = 0);

// NOT USING THIS FUNCTION
/*
void gpu_gradient_descent(NeuralNetwork& nn, const double learning_rate, struct grads &bpgrads) {
    for(size_t i = 0; i < nn.W.size(); ++i) {
        int M = nn.W[i].n_rows,
            N = nn.W[i].n_cols;
        double *d_Wi, *d_dWi;
        cudaMalloc((void **)&d_Wi, (M * N) * sizeof(double));
        cudaMalloc((void **)&d_dWi, (M * N) * sizeof(double));
        cudaMemcpy(d_Wi, nn.W[i].memptr(), (M * N) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dWi, bpgrads.dW[i].memptr(), (M * N) * sizeof(double), cudaMemcpyHostToDevice);

        double alpha = -learning_rate, beta = 1.0;
        gpu_linear(d_dWi, d_Wi, alpha, beta, M, N, 0);
        cudaMemcpy(nn.W[i].memptr(), d_Wi, (M * N) * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_Wi);
        cudaFree(d_dWi);
    }

    for(size_t i = 0; i < nn.b.size(); ++i) {
        int M = nn.W[i].n_rows,
            N = 1;
        double *d_bi, *d_dbi;
        cudaMalloc((void **)&d_bi, (M * N) * sizeof(double));
        cudaMalloc((void **)&d_dbi, (M * N) * sizeof(double));
        cudaMemcpy(d_bi, nn.b[i].memptr(), (M * N) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dbi, bpgrads.db[i].memptr(), (M * N) * sizeof(double), cudaMemcpyHostToDevice);

        double alpha = -learning_rate, beta = 1.0;
        gpu_linear(d_dbi, d_bi, alpha, beta, M, N, 0);
        cudaMemcpy(nn.b[i].memptr(), d_bi, (M * N) * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_bi);
        cudaFree(d_dbi);
    }
}
*/

#endif
