// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}
// TL {"workspace_files": []}

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "tma-interface.cuh"

typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 3: TMA Memcpy
////////////////////////////////////////////////////////////////////////////////

__global__ void tma_copy(__grid_constant__ const CUtensorMap tensor_map,
                         __grid_constant__ const CUtensorMap dest_tensor_map,
                         const int total_cols, const int total_rows) {
    constexpr int TILE_COLS = 128;
    constexpr int TILE_ROWS = 64;
    constexpr int TILE_ELEMS = TILE_COLS * TILE_ROWS;
    constexpr uint32_t TILE_BYTES =
        static_cast<uint32_t>(TILE_ELEMS * sizeof(bf16));

    __shared__ alignas(128) bf16 smem_tiles[2][TILE_ELEMS];
    __shared__ alignas(8) uint64_t mbarriers[2];

    const int tiles_per_row =
        (total_cols + TILE_COLS - 1) / TILE_COLS;
    const int row_tiles = (total_rows + TILE_ROWS - 1) / TILE_ROWS;
    const size_t total_tiles =
        static_cast<size_t>(tiles_per_row) * static_cast<size_t>(row_tiles);

    if (blockIdx.x >= total_tiles) {
        return;
    }

    // Initialize barriers once per block.
    if (threadIdx.x == 0) {
        init_barrier(&mbarriers[0], 1);
        init_barrier(&mbarriers[1], 1);
    }
    __syncthreads();
    async_proxy_fence();

    auto issue_tma_load = [&](size_t tile_idx, int buffer) {
        const int tile_col = static_cast<int>(tile_idx % tiles_per_row);
        const int tile_row = static_cast<int>(tile_idx / tiles_per_row);
        const int c0 = tile_col * TILE_COLS;
        const int c1 = tile_row * TILE_ROWS;
        bf16 *smem_ptr = smem_tiles[buffer];
        uint64_t *bar = &mbarriers[buffer];

        expect_bytes_and_arrive(bar, TILE_BYTES);
        cp_async_bulk_tensor_2d_global_to_shared(
            smem_ptr, &tensor_map, c0, c1, bar);
    };

    int phase_parity[2] = {0, 0};
    const size_t stride = gridDim.x;
    size_t tile_idx = blockIdx.x;

    if (threadIdx.x == 0) {
        issue_tma_load(tile_idx, tile_idx & 1);
    }
    __syncthreads();

    for (; tile_idx < total_tiles; tile_idx += stride) {
        const int buffer = static_cast<int>(tile_idx & 1);
        const int tile_col = static_cast<int>(tile_idx % tiles_per_row);
        const int tile_row = static_cast<int>(tile_idx / tiles_per_row);
        const int c0 = tile_col * TILE_COLS;
        const int c1 = tile_row * TILE_ROWS;
        bf16 *smem_ptr = smem_tiles[buffer];
        uint64_t *bar = &mbarriers[buffer];

        if (threadIdx.x == 0) {
            wait(bar, phase_parity[buffer]);
            phase_parity[buffer] ^= 1;
            cp_async_bulk_tensor_2d_shared_to_global(&dest_tensor_map, c0, c1,
                                                     smem_ptr);
            tma_commit_group();
        }
        __syncthreads();

        const size_t next_tile = tile_idx + stride;
        if (next_tile < total_tiles && threadIdx.x == 0) {
            const int next_buffer = static_cast<int>(next_tile & 1);
            issue_tma_load(next_tile, next_buffer);
        }

        if (threadIdx.x == 0) {
            // Allow one outstanding commit group to keep the pipeline full.
            tma_wait_until_pending<1>();
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        tma_wait_until_pending<0>();
    }
}

void launch_tma_copy(bf16 *dest, bf16 *src, int num_elements) {
    constexpr int TILE_COLS = 128;
    constexpr int TILE_ROWS = 64;

    CUDA_CHECK(cuInit(0));

    CUtensorMap src_map;
    CUtensorMap dest_map;

    const int cols = TILE_COLS;
    const int rows = num_elements / cols;

    const cuuint64_t globalDim[2] = {static_cast<cuuint64_t>(cols),
                                     static_cast<cuuint64_t>(rows)};
    const cuuint64_t globalStrides[1] = {
        static_cast<cuuint64_t>(cols * sizeof(bf16))};
    const cuuint32_t boxDim[2] = {static_cast<cuuint32_t>(TILE_COLS),
                                  static_cast<cuuint32_t>(TILE_ROWS)};
    const cuuint32_t elementStrides[2] = {1, 1};

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &src_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, src, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &dest_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dest, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count,
                                      cudaDevAttrMultiProcessorCount, 0));

    const int tiles_per_row =
        (cols + TILE_COLS - 1) / TILE_COLS;
    const int row_tiles = (rows + TILE_ROWS - 1) / TILE_ROWS;
    const size_t total_tiles =
        static_cast<size_t>(tiles_per_row) * static_cast<size_t>(row_tiles);

    const int max_blocks = sm_count * 4;
    const int grid_blocks =
        static_cast<int>(std::min(total_tiles, static_cast<size_t>(max_blocks)));

    dim3 block_dim(32);
    dim3 grid_dim(grid_blocks);

    tma_copy<<<grid_dim, block_dim>>>(src_map, dest_map, cols, rows);
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

const int elem_per_block = 16384;
__global__ void simple_vector_copy(bf16 *__restrict__ dest,
                                   const bf16 *__restrict__ src, int N) {
    constexpr int VEC_ELEMS = 8;
    using VecT = uint4;

    int total_vecs = elem_per_block / VEC_ELEMS;
    int start_vec = (blockIdx.x * blockDim.x) * total_vecs;

    const VecT *src_vec = reinterpret_cast<const VecT *>(src);
    VecT *dest_vec = reinterpret_cast<VecT *>(dest);

    for (int i = threadIdx.x; i < blockDim.x * total_vecs; i += blockDim.x) {
        dest_vec[start_vec + i] = src_vec[start_vec + i];
    }
}

#define BENCHMARK_KERNEL(kernel_call, num_iters, size_bytes, label)            \
    do {                                                                       \
        cudaEvent_t start, stop;                                               \
        CUDA_CHECK(cudaEventCreate(&start));                                   \
        CUDA_CHECK(cudaEventCreate(&stop));                                    \
        CUDA_CHECK(cudaEventRecord(start));                                    \
        for (int i = 0; i < num_iters; i++) {                                  \
            kernel_call;                                                       \
        }                                                                      \
        CUDA_CHECK(cudaEventRecord(stop));                                     \
        CUDA_CHECK(cudaEventSynchronize(stop));                                \
        float elapsed_time;                                                    \
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));          \
        float time_per_iter = elapsed_time / num_iters;                        \
        float bandwidth_gb_s = (2.0 * size_bytes * 1e-6 / time_per_iter);      \
        printf("%s - Time: %.4f ms, Bandwidth: %.2f GB/s\n", label,            \
               time_per_iter, bandwidth_gb_s);                                 \
        CUDA_CHECK(cudaEventDestroy(start));                                   \
        CUDA_CHECK(cudaEventDestroy(stop));                                    \
    } while (0)

int main() {
    const size_t size = 132 * 10 * 32 * 128 * 128;

    // Allocate and initialize host memory
    bf16 *matrix = (bf16 *)malloc(size * sizeof(bf16));
    const int N = 128;
    for (int idx = 0; idx < size; idx++) {
        int i = idx / N;
        int j = idx % N;
        // Don't want to use a random number generator, takes too long.
        float val = fmodf((i * 123 + j * 37) * 0.001f, 2.0f) - 1.0f;
        matrix[idx] = __float2bfloat16(val);
    }

    // Allocate device memory
    bf16 *d_src, *d_dest;
    CUDA_CHECK(cudaMalloc(&d_src, size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_dest, size * sizeof(bf16)));
    CUDA_CHECK(
        cudaMemcpy(d_src, matrix, size * sizeof(bf16), cudaMemcpyHostToDevice));

    // Test TMA copy correctness
    printf("Testing TMA copy correctness...\n");
    CUDA_CHECK(cudaMemset(d_dest, 0, size * sizeof(bf16)));
    launch_tma_copy(d_dest, d_src, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bf16 *tma_result = (bf16 *)malloc(size * sizeof(bf16));
    CUDA_CHECK(cudaMemcpy(tma_result, d_dest, size * sizeof(bf16),
                          cudaMemcpyDeviceToHost));

    bool tma_correct = true;
    for (int idx = 0; idx < size; idx++) {
        if (tma_result[idx] != matrix[idx]) {
            printf("First mismatch at [%d]: %.4f != %.4f\n", idx,
                   __bfloat162float(tma_result[idx]),
                   __bfloat162float(matrix[idx]));
            tma_correct = false;
            break;
        }
    }
    printf("TMA Copy: %s\n\n", tma_correct ? "PASSED" : "FAILED");
    free(tma_result);

    // Test simple copy correctness
    printf("Testing simple copy correctness...\n");
    CUDA_CHECK(cudaMemset(d_dest, 0, size * sizeof(bf16)));
    simple_vector_copy<<<size / (elem_per_block * 32), 32>>>(d_dest, d_src,
                                                             size);
    CUDA_CHECK(cudaDeviceSynchronize());

    bf16 *simple_result = (bf16 *)malloc(size * sizeof(bf16));
    CUDA_CHECK(cudaMemcpy(simple_result, d_dest, size * sizeof(bf16),
                          cudaMemcpyDeviceToHost));

    bool simple_correct = true;
    for (int idx = 0; idx < size; idx++) {
        if (simple_result[idx] != matrix[idx]) {
            printf("First mismatch at [%d]: %.4f != %.4f\n", idx,
                   __bfloat162float(tma_result[idx]),
                   __bfloat162float(matrix[idx]));

            simple_correct = false;
            break;
        }
    }
    printf("Simple Copy: %s\n\n", simple_correct ? "PASSED" : "FAILED");
    free(simple_result);

    // Benchmark both kernels
    const int num_iters = 10;
    const size_t size_bytes = size * sizeof(bf16);

    if (tma_correct) {
        BENCHMARK_KERNEL((launch_tma_copy(d_dest, d_src, size)), num_iters,
                         size_bytes, "TMA Copy");
    }

    if (simple_correct) {
        BENCHMARK_KERNEL(
            (simple_vector_copy<<<size / (elem_per_block * 32), 32>>>(
                 d_dest, d_src, size),
             cudaDeviceSynchronize()),
            num_iters, size_bytes, "Simple Copy");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dest));
    free(matrix);
    return 0;
}