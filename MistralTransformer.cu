#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// CUDA kernel for generating positions  
__global__ void generate_positions(int* positions, int* seqlens, int total_length) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < total_length) {
        int offset = 0;
        for (int i = 0; i < idx; ++i) {
            offset += seqlens[i];
        }
        positions[idx] = idx - offset;
    }
}

// Class definition for SimpleInputMetadata
class SimpleInputMetadata {
public:
    int* positions;

    SimpleInputMetadata(int total_length) {
        // Allocate memory on GPU for positions
        cudaMalloc((void**)&positions, total_length * sizeof(int));
    }

    ~SimpleInputMetadata() {
        // Free GPU memory
        cudaFree(positions);
    }

    // Static function to create positions from sequence lengths
    static SimpleInputMetadata from_seqlens(const std::vector<int>& seqlens) {
        int total_length = 0;

        // Calculate total length
        for (int seqlen : seqlens) {
            total_length += seqlen;
        }

        // Allocate GPU memory for sequence lengths
        int* d_seqlens;
        cudaMalloc((void**)&d_seqlens, seqlens.size() * sizeof(int));
        cudaMemcpy(d_seqlens, seqlens.data(), seqlens.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Create an object of SimpleInputMetadata
        SimpleInputMetadata metadata(total_length);

        // Launch the kernel to generate positions
        int blockSize = 256;
        int numBlocks = (total_length + blockSize - 1) / blockSize;
        generate_positions<<<numBlocks, blockSize>>>(metadata.positions, d_seqlens, total_length);

        // Free sequence lengths memory
        cudaFree(d_seqlens);

        return metadata;
    }
};
