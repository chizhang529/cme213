#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

double norms(NeuralNetwork& nn) {
    double norm_sum = 0;

    for(int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    nn.W[0].save(s.str(), arma::raw_ascii);
    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    nn.W[1].save(t.str(), arma::raw_ascii);
    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    nn.b[0].save(u.str(), arma::raw_ascii);
    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
    arma::mat A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
    double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
    double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
    double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
    double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

    int ow = 15;

    if(iter == 0) {
        error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(
                       ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1"
                   << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(
                       ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left
                   << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(
                       ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left << std::setw(
                   ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 <<
               std::left << std::setw(ow) << max_errb0 << std::left << std::setw(
                   ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left <<
               std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 <<
               std::left<< std::setw(ow) << L2_errb1 << "\n";

}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    cache.z.resize(2);
    cache.a.resize(2);

    // std::cout << W[0].n_rows << "\n";
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    arma::mat z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
    cache.z[0] = z1;
    // std::cout << "z0  " << cache.z[0](0,0) << std::endl;

    arma::mat a1;
    sigmoid(z1, a1);
    cache.a[0] = a1;
    // std::cout << "a0  " << cache.a[0](0,0) << std::endl;

    assert(a1.n_rows == nn.W[1].n_cols);
    arma::mat z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
    cache.z[1] = z2;
    // std::cout << "z1  " << cache.z[1](0,0) << std::endl;

    arma::mat a2;
    softmax(z2, a2);
    cache.a[1] = cache.yc = a2;
    // std::cout << "a1  " << cache.a[1](0,0) << std::endl;
    // std::cout << "yc  " << cache.yc(0,0) << std::endl;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;

    // std::cout << "backprop " << bpcache.yc << "\n";
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    // std::cout << "dW1  " << bpgrads.dW[1](0,0) << std::endl;
    bpgrads.db[1] = arma::sum(diff, 1);
    // std::cout << "db1  " << bpgrads.db[1](0,0) << std::endl;
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    // std::cout << "dW0  " << bpgrads.dW[0](0,0) << std::endl;
    bpgrads.db[0] = arma::sum(dz1, 1);
    // std::cout << "db0  " << bpgrads.db[0](0,0) << std::endl;
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg) {
    int N = yc.n_cols;
    double ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

    double data_loss = ce_sum / N;
    double reg_loss = 0.5 * reg * norms(nn);
    double loss = data_loss + reg_loss;
    // std::cout << "Loss: " << loss << "\n";
    return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label) {
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for(int i = 0; i < X.n_cols; ++i) {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads) {
    double h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for(int j = 0; j < nn.W[i].n_rows; ++j) {
            for(int k = 0; k < nn.W[i].n_cols; ++k) {
                double oldval = nn.W[i](j,k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                double fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                double fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for(int j = 0; j < nn.b[i].size(); ++j) {
            double oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            double fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            double fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2*h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg,
           const int epochs, const int batch_size, bool grad_check, int print_every,
           int debug) {
    int N = X.n_cols;
    int iter = 0;
    int print_flag = 0;

    for(int epoch = 0 ; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if(print_every > 0 && iter % print_every == 0) {
                if(grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for(int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for(int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
               for the first batch of each epoch to avoid saving too many large files.
               Note that for the first time, you have to run debug and serial modes together.
               This will run the following function and write out files to CPUmats folder.
               In the later runs (with same parameters), you can use just the debug flag to
               output diff b/w CPU and GPU without running CPU version */
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if(debug && print_flag) {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}

/* GPU IMPLEMENTATIONS */
void gpu_feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    // two-layer neural network
    cache.z.resize(2);
    cache.a.resize(2);
    cache.X = X;

    // dimensions: W1(M1, K1) x(K1, N1) b1(M1, 1)
    int M1 = nn.W[0].n_rows,
        N1 = X.n_cols,
        K1 = X.n_rows;
    cache.z[0].zeros(M1, N1);
    cache.a[0].zeros(M1, N1);
    assert(nn.W[0].n_cols == X.n_rows);
    /*----- layer 1: z1 = W1*x + b1, a1 = sigmoid(z1) -----*/
    double *d_W1, *d_X, *d_b1;
    cudaMalloc((void **)&d_W1, (M1 * K1) * sizeof(double));
    cudaMalloc((void **)&d_X, (K1 * N1) * sizeof(double));
    cudaMalloc((void **)&d_b1, (M1 * N1) * sizeof(double));
    arma::mat b1_mat = arma::repmat(nn.b[0], 1, N1); // b1(M1, 1) -> b1_mat(M1, N1)
    cudaMemcpy(d_W1, nn.W[0].memptr(), (M1 * K1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X.memptr(), (K1 * N1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1_mat.memptr(), (M1 * N1) * sizeof(double), cudaMemcpyHostToDevice);

    double alpha = 1.0, beta = 1.0;
    myGEMM(d_W1, d_X, d_b1, &alpha, &beta, M1, N1, K1);
    // z1 is in-place caculated
    cudaMemcpy(cache.z[0].memptr(), d_b1, (M1 * N1) * sizeof(double), cudaMemcpyDeviceToHost);
    // std::cout << "z0  " << cache.z[0](0,0) << std::endl;
    gpu_sigmoid(d_b1, M1, N1);
    // a1 is in-place caculated
    cudaMemcpy(cache.a[0].memptr(), d_b1, (M1 * N1) * sizeof(double), cudaMemcpyDeviceToHost);
    // std::cout << "a0  " << cache.a[0](0,0) << std::endl;

    // dimensions: W2(M2, K2=M1) a1(K2=M1, N2=N1) b2(M2, 1)
    int M2 = nn.W[1].n_rows,
        N2 = N1,
        K2 = M1;
    cache.z[1].zeros(M2, N2);
    cache.a[1].zeros(M2, N2);
    cache.yc.zeros(M2, N2);
    assert(nn.W[1].n_cols == cache.a[0].n_rows);
    /*----- layer 2: z2 = W2*a1 + b2, yc = a2 = softmax(z2) -----*/
    double *d_W2, *d_b2;
    cudaMalloc((void **)&d_W2, (M2 * K2) * sizeof(double));
    cudaMalloc((void **)&d_b2, (M2 * N2) * sizeof(double));
    arma::mat b2_mat = arma::repmat(nn.b[1], 1, N2); // b2(M2, 1) -> b2_mat(M2, N2)
    cudaMemcpy(d_W2, nn.W[1].memptr(), (M2 * K2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2_mat.memptr(), (M2 * N2) * sizeof(double), cudaMemcpyHostToDevice);

    myGEMM(d_W2, d_b1, d_b2, &alpha, &beta, M2, N2, K2);
    // z2 is in-place caculated
    cudaMemcpy(cache.z[1].memptr(), d_b2, (M2 * N2) * sizeof(double), cudaMemcpyDeviceToHost);
    // std::cout << "z1  " << cache.z[1](0,0) << std::endl;
    gpu_softmax(d_b2, M2, N2);
    // yc = a2 is in-place calculated
    cudaMemcpy(cache.a[1].memptr(), d_b2, (M2 * N2) * sizeof(double), cudaMemcpyDeviceToHost);
    // std::cout << "a1  " << cache.a[1](0,0) << std::endl;
    cudaMemcpy(cache.yc.memptr(), d_b2, (M2 * N2) * sizeof(double), cudaMemcpyDeviceToHost);
    // std::cout << "yc  " << cache.yc(0,0) << std::endl;
    // free memory
    cudaFree(d_W1);
    cudaFree(d_X);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
}

void gpu_backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    // dimensions
    int M1 = nn.W[0].n_rows,
        N1 = y.n_cols,
        K1 = bpcache.X.n_rows;
    bpgrads.dW[0].zeros(M1, K1);
    bpgrads.db[0].zeros(M1, 1);

    int M2 = nn.W[1].n_rows,
        N2 = N1,
        K2 = M1;
    bpgrads.dW[1].zeros(M2, K2);
    bpgrads.db[1].zeros(M2, 1);

    /*----- layer 2 -----*/
    double *d_diff, *d_yc, *d_dW2, *d_db2;
    cudaMalloc((void **)&d_diff, (M2 * N2) * sizeof(double));
    cudaMalloc((void **)&d_yc, (M2 * N2) * sizeof(double));
    cudaMalloc((void **)&d_dW2, (M2 * K2) * sizeof(double));
    cudaMalloc((void **)&d_db2, (M2 * 1) * sizeof(double));
    cudaMemcpy(d_diff, y.memptr(), (M2 * N2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yc, bpcache.yc.memptr(), (M2 * N2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dW2, nn.W[1].memptr(), (M2 * K2) * sizeof(double), cudaMemcpyHostToDevice);

    /*----- layer 1 -----*/
    double *d_X, *d_a1, *d_da1, *d_dz1, *d_dW1, *d_db1;
    cudaMalloc((void **)&d_X, (K1 * N1) * sizeof(double));
    cudaMalloc((void **)&d_a1, (M1 * N1) * sizeof(double));
    cudaMalloc((void **)&d_da1, (M1 * N1) * sizeof(double));
    cudaMalloc((void **)&d_dz1, (M1 * N1) * sizeof(double));
    cudaMalloc((void **)&d_dW1, (M1 * K1) * sizeof(double));
    cudaMalloc((void **)&d_db1, (M1 * 1) * sizeof(double));
    cudaMemcpy(d_X, bpcache.X.memptr(), (K1 * N1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a1, bpcache.a[0].memptr(), (M1 * N1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dW1, nn.W[0].memptr(), (M1 * K1) * sizeof(double), cudaMemcpyHostToDevice);

    // transpose (X.T, a1.T, W2.T)
    double *d_X_t;
    cudaMalloc((void **)&d_X_t, (K1 * N1) * sizeof(double));
    gpu_transpose(d_X, d_X_t, K1, N1);

    double *d_a1_t;
    cudaMalloc((void **)&d_a1_t, (M1 * N1) * sizeof(double));
    gpu_transpose(d_a1, d_a1_t, M1, N1);

    double *d_dW2_t;
    cudaMalloc((void **)&d_dW2_t, (M2 * K2) * sizeof(double));
    gpu_transpose(d_dW2, d_dW2_t, M2, K2);

    // dW2 = 1/N * (yc - y) * a1.T + reg * W2
    double factor = 1.0 / (double)N1;
    gpu_linear(d_yc, d_diff, factor, -factor, M2, N2, 0);   // diff = 1/N * (yc - y)
    double alpha = 1.0, beta = reg;
    myGEMM(d_diff, d_a1_t, d_dW2, &alpha, &beta, M2, K2, N2);
    cudaMemcpy(bpgrads.dW[1].memptr(), d_dW2, (M2 * K2) * sizeof(double), cudaMemcpyDeviceToHost);
    // std::cout << "dW1  " << bpgrads.dW[1](0,0) << std::endl;
    // db2 = 1/N * (yc - y)
    gpu_row_sum(d_diff, d_db2, M2, N2);   // gpu_sum(d_diff, d_db2, M2, N2, 1);
    cudaMemcpy(bpgrads.db[1].memptr(), d_db2, (M2 * 1) * sizeof(double), cudaMemcpyDeviceToHost);
    // std::cout << "db1  " << bpgrads.db[1](0,0) << std::endl;

    // da1 = W2.T * diff
    alpha = 1.0, beta = 0.0;
    myGEMM(d_dW2_t, d_diff, d_da1, &alpha, &beta, M1, N2, M2);
    // dz1 = da1 .* a1 .* (1 - a1)
    gpu_linear(d_a1, d_dz1, -1.0, 1.0, M1, N1, 1);
    gpu_elem_mult(d_a1, d_dz1, 1.0, M1, N1);
    gpu_elem_mult(d_da1, d_dz1, 1.0, M1, N1);
    // dW1 = dz1 * X.T + reg * W1
    alpha = 1.0, beta = reg;
    myGEMM(d_dz1, d_X_t, d_dW1, &alpha, &beta, M1, K1, N1);
    cudaMemcpy(bpgrads.dW[0].memptr(), d_dW1, (M1 * K1) * sizeof(double), cudaMemcpyDeviceToHost);
    // std::cout << "dW0  " << bpgrads.dW[0](0,0) << std::endl;
    // db1 = dz1
    gpu_row_sum(d_dz1, d_db1, M1, N1);   // gpu_sum(d_dz1, d_db1, M1, N1, 1);
    cudaMemcpy(bpgrads.db[0].memptr(), d_db1, (M1 * 1) * sizeof(double), cudaMemcpyDeviceToHost);
    // std::cout << "db0  " << bpgrads.db[0](0,0) << std::endl;

    // free memory
    cudaFree(d_diff);
    cudaFree(d_yc);
    cudaFree(d_dW2);
    cudaFree(d_db2);

    cudaFree(d_X);
    cudaFree(d_a1);
    cudaFree(d_da1);
    cudaFree(d_dz1);
    cudaFree(d_dW1);
    cudaFree(d_db1);

    cudaFree(d_X_t);
    cudaFree(d_a1_t);
    cudaFree(d_dW2_t);
}

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

/*
 * TODO
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0)?X.n_cols:0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;

    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in NeuralNetwork &nn of rank 0 before returning from the function. */

    /* iter is a variable used to manage debugging. It increments in the inner loop
       and therefore goes from 0 to epochs*num_batches */
    int iter = 0;

    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */
            if (rank == 0) {
                int last_col = std::min((batch + 1)*batch_size-1, N-1);
                arma::mat X_batch = X.cols(batch * batch_size, last_col);
                arma::mat y_batch = y.cols(batch * batch_size, last_col);

                struct cache bpcache;
                gpu_feedforward(nn, X_batch, bpcache);

                struct grads bpgrads;
                gpu_backprop(nn, y_batch, reg, bpcache, bpgrads);

                if(print_every > 0 && iter % print_every == 0) {
                    if(grad_check) {
                        struct grads numgrads;
                        numgrad(nn, X_batch, y_batch, reg, numgrads);
                        assert(gradcheck(numgrads, bpgrads));
                    }

                    std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                              epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
                }

                // gpu_gradient_descent(nn, learning_rate, bpgrads);
                // Gradient descent step
                for(int i = 0; i < nn.W.size(); ++i) {
                    nn.W[i] -= learning_rate * bpgrads.dW[i];
                }

                for(int i = 0; i < nn.b.size(); ++i) {
                    nn.b[i] -= learning_rate * bpgrads.db[i];
                }
            }

            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            /* Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  */
            if(debug && rank == 0 && print_flag) {
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;
        }
    }

    error_file.close();
}
