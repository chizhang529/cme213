// utility function for checking cuda errors
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

static const uint MAX_GRID_DIM = 65535;
/* Each kernel handles the update of one pagerank score. In other
 * words, each kernel handles one row of the update:
 *
 *      pi(t+1) = (1/2) A pi(t) + (1 / (2N))
 *
 * You may assume that num_nodes <= blockDim.x * 65535
 *
 */
__global__
void device_graph_propagate(const uint* graph_indices
                            , const uint* graph_edges
                            , const float* graph_nodes_in
                            , float* graph_nodes_out
                            , const float* inv_edges_per_node
                            , int num_nodes) {
    uint bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint tid = bid * blockDim.x + threadIdx.x;
    if (tid < num_nodes) {
        float sum = 0.0f;
        // iterate all edges
        uint start = graph_indices[tid], end = graph_indices[tid+1];
        for (uint i = start; i < end; ++i) {
            uint node = graph_edges[i];
            sum += 0.5f * inv_edges_per_node[node] * graph_nodes_in[node];
        }
        // add constant term
        sum += 1.0f / (2 * num_nodes);
        // update
        graph_nodes_out[tid] = sum;
    }
}

// swap pointers to input and output array
void swapio(float **in, float **out) {
    float *temp;
    temp = *in;
    *in = *out;
    *out = temp;
}

/* This function executes a specified number of iterations of the
 * pagerank algorithm. The variables are:
 *
 * h_graph_indices, h_graph_edges:
 *     These arrays describe the indices of the neighbors of node i.
 *     Specifically, node i is adjacent to all nodes in the range
 *     h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]].
 *
 * h_node_values_input:
 *     An initial guess of pi(0).
 *
 * h_gpu_node_values_output:
 *     Output array for the pagerank vector.
 *
 * h_inv_edges_per_node:
 *     The i'th element in this array is the reciprocal of the
 *     out degree of the i'th node.
 *
 * nr_iterations:
 *     The number of iterations to run the pagerank algorithm for.
 *
 * num_nodes:
 *     The number of nodes in the whole graph (ie N).
 *
 * avg_edges:
 *     The average number of edges in the graph. You are guaranteed
 *     that the whole graph has num_nodes * avg_edges edges.
 *
 */
double device_graph_iterate(const uint* h_graph_indices
                            , const uint* h_graph_edges
                            , const float* h_node_values_input
                            , float* h_gpu_node_values_output
                            , const float* h_inv_edges_per_node
                            , int nr_iterations
                            , int num_nodes
                            , int avg_edges) {
    // allocate GPU memory
    const uint num_indices = num_nodes + 1;
    const uint num_edges = num_nodes * avg_edges;

    uint *d_graph_indices, *d_graph_edges;
    float *d_input, *d_output, *d_inv_edges_per_node;

    cudaMalloc((void **)&d_graph_indices, num_indices * sizeof(uint));
    cudaMalloc((void **)&d_graph_edges, num_edges * sizeof(uint));
    cudaMalloc((void **)&d_input, num_nodes * sizeof(float));
    cudaMalloc((void **)&d_output, num_nodes * sizeof(float));
    cudaMalloc((void **)&d_inv_edges_per_node, num_nodes * sizeof(float));

    // check for allocation failure
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    // copy data to the GPU
    cudaMemcpy(d_graph_indices, h_graph_indices, num_indices * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_edges, h_graph_edges, num_edges * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_node_values_input, num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_edges_per_node, h_inv_edges_per_node,
               num_nodes * sizeof(float), cudaMemcpyHostToDevice);

    start_timer(&timer);

    // compute grid dimensions
    const int block_size = 192;
    // compute grid dimensions
    dim3 blocks_per_grid(1, 1);         // 2D grid
    dim3 threads_per_block(block_size); // 1D block
    // compute number of blocks needed
    uint num_blocks = ceil((float)num_nodes / (float)block_size);
    if (num_blocks > MAX_GRID_DIM) {
        blocks_per_grid.x = MAX_GRID_DIM;
        blocks_per_grid.y = ceil((float)num_blocks / (float)MAX_GRID_DIM);
    } else {
        blocks_per_grid.x = num_blocks;
    }

    // launch kernel the appropriate number of iterations
    for (int i = 0; i < nr_iterations; ++i) {
        device_graph_propagate<<<blocks_per_grid, threads_per_block>>>(d_graph_indices,
                                                                       d_graph_edges,
                                                                       d_input, d_output,
                                                                       d_inv_edges_per_node,
                                                                       num_nodes);
        swapio(&d_input, &d_output);
    }

    check_launch("gpu graph propagate");
    double gpu_elapsed_time = stop_timer(&timer);

    // copy final data back to the host for correctness checking
    cudaMemcpy(h_gpu_node_values_output, d_input, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_graph_indices);
    cudaFree(d_graph_edges);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_inv_edges_per_node);
    // error checking
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    return gpu_elapsed_time;
}
