// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}
// TL {"workspace_files": []}

#include <cuda.h>
#include <cuda_bf16.h>
#include <random>
#include <stdio.h>

#include "tma-interface.cuh"

// Type alias for bfloat16
typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 0: Single Block, Single Tile TMA Load
////////////////////////////////////////////////////////////////////////////////

template <int TILE_M, int TILE_N>
__global__ void single_tma_load(__grid_constant__ const CUtensorMap src_map,
                                bf16 *dest) {
    __shared__ alignas(128) bf16 smem_tile[TILE_M * TILE_N];
    __shared__ alignas(8) uint64_t mbar;

    if (threadIdx.x == 0) {
        init_barrier(&mbar, 1);
    }
    __syncthreads();

    async_proxy_fence();

    if (threadIdx.x == 0) {
        const uint32_t bytes = static_cast<uint32_t>(TILE_M * TILE_N * sizeof(bf16));
        expect_bytes_and_arrive(&mbar, bytes);

        cp_async_bulk_tensor_2d_global_to_shared(
            smem_tile, &src_map, 0, 0, &mbar);

        wait(&mbar,0);

        // Store the tile back to global memory using regular CUDA stores
        for (int i = 0; i < TILE_M; ++i) {
            #pragma unroll
            for (int j = 0; j < TILE_N; ++j) {
                dest[i * TILE_N + j] = smem_tile[i * TILE_N + j];
            }
        }
    }
}

template <int TILE_M, int TILE_N>
void launch_single_tma_load(bf16 *src, bf16 *dest) {
    CUtensorMap src_map;

    // Tensor rank and dimensions (fastest to slowest): {N, M}
    constexpr cuuint32_t rank = 2;
    const cuuint64_t globalDim[rank] = {static_cast<cuuint64_t>(TILE_N), static_cast<cuuint64_t>(TILE_M)};
    const cuuint64_t globalStrides[rank - 1] = {static_cast<cuuint64_t>(TILE_N * sizeof(bf16))};
    const cuuint32_t boxDim[rank] = {static_cast<cuuint32_t>(TILE_N), static_cast<cuuint32_t>(TILE_M)};

    // consecutive elements
    const cuuint32_t elementStrides[rank] = {1, 1};

    // setting up the CUtensorMap
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &src_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, // bf16 datatype
        rank,
        src,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // Launch single block, single thread (single-lane) kernel
    single_tma_load<TILE_M, TILE_N><<<1, 1>>>(src_map, dest);
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main() {
    const int M = 64;
    const int N = 128;
    const uint64_t total_size = M * N;

    // Allocate host and device memory
    bf16 *matrix = (bf16 *)malloc(total_size * sizeof(bf16));
    bf16 *d_matrix;
    bf16 *d_dest;
    cudaMalloc(&d_matrix, total_size * sizeof(bf16));
    cudaMalloc(&d_dest, total_size * sizeof(bf16));

    // Zero out destination buffer
    for (int i = 0; i < total_size; i++) {
        matrix[i] = 0;
    }
    cudaMemcpy(d_dest, matrix, total_size * sizeof(bf16),
               cudaMemcpyHostToDevice);

    // Initialize source matrix on host
    std::default_random_engine generator(0);
    std::normal_distribution<float> dist(0, 1);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = dist(generator);
            matrix[i * N + j] = __float2bfloat16(val);
        }
    }
    cudaMemcpy(d_matrix, matrix, total_size * sizeof(bf16),
               cudaMemcpyHostToDevice);

    printf("\n\nRunning TMA load kernel...\n\n");

    // Launch the TMA kernel
    launch_single_tma_load<M, N>(d_matrix, d_dest);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    bf16 *final_output = (bf16 *)malloc(total_size * sizeof(bf16));
    cudaMemcpy(final_output, d_dest, total_size * sizeof(bf16),
               cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = true;
    for (int x = 0; x < M * N; x++) {
        int i = x / N;
        int j = x % N;
        float ref = (float)matrix[i * N + j];
        float computed = (float)final_output[i * N + j];
        if (ref != computed) {
            correct = false;
            printf("Mismatch at (%d, %d): expected %f, got %f \n", i, j, ref,
                   computed);
            break;
        }
    }

    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    // Cleanup resources
    cudaFree(d_matrix);
    cudaFree(d_dest);
    free(matrix);
    free(final_output);

    return 0;
}