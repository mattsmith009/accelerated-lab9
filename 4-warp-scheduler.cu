// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "tma-interface.cuh"

typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 4: Bring Your Own Warp Scheduler
////////////////////////////////////////////////////////////////////////////////

__global__ void
tma_multiwarp_pipeline(__grid_constant__ const CUtensorMap tensor_map,
                       __grid_constant__ const CUtensorMap dest_tensor_map,
                       const int total_elems) {
    constexpr int TILE_COLS = 128;
    constexpr int TILE_ROWS = 64;
    constexpr int TILE_ELEMS = TILE_COLS * TILE_ROWS;
    constexpr uint32_t TILE_BYTES =
        static_cast<uint32_t>(TILE_ELEMS * sizeof(bf16));
    constexpr int PIPE_DEPTH = 4;

    const int total_cols = TILE_COLS;
    const int total_rows = total_elems / total_cols;

    const int tiles_per_row =
        (total_cols + TILE_COLS - 1) / TILE_COLS; // == 1 with current params
    const int row_tiles = (total_rows + TILE_ROWS - 1) / TILE_ROWS;
    const size_t total_tiles =
        static_cast<size_t>(tiles_per_row) * static_cast<size_t>(row_tiles);

    if (blockIdx.x >= total_tiles) {
        return;
    }

    extern __shared__ unsigned char shared_raw[];
    uintptr_t base_ptr = reinterpret_cast<uintptr_t>(shared_raw);
    base_ptr = (base_ptr + 127) & ~static_cast<uintptr_t>(127);
    bf16 *smem_tiles = reinterpret_cast<bf16 *>(base_ptr);

    uintptr_t after_tiles =
        base_ptr + sizeof(bf16) * PIPE_DEPTH * static_cast<size_t>(TILE_ELEMS);
    after_tiles = (after_tiles + 7) & ~static_cast<uintptr_t>(7);
    uint64_t *mbarriers = reinterpret_cast<uint64_t *>(after_tiles);

    uintptr_t after_barriers =
        after_tiles + sizeof(uint64_t) * PIPE_DEPTH;
    after_barriers = (after_barriers + 3) & ~static_cast<uintptr_t>(3);
    volatile int *buffer_state = reinterpret_cast<volatile int *>(after_barriers);

    uintptr_t after_state =
        after_barriers + sizeof(int) * PIPE_DEPTH;
    after_state = (after_state + 3) & ~static_cast<uintptr_t>(3);
    unsigned long long *tile_index =
        reinterpret_cast<unsigned long long *>(after_state);

    uintptr_t after_index =
        after_state + sizeof(unsigned long long) * PIPE_DEPTH;
    after_index = (after_index + 3) & ~static_cast<uintptr_t>(3);
    int *barrier_phase = reinterpret_cast<int *>(after_index);

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    if (warp_id == 0 && lane_id == 0) {
        for (int i = 0; i < PIPE_DEPTH; ++i) {
            init_barrier(&mbarriers[i], 1);
            barrier_phase[i] = 0;
            buffer_state[i] = 0;
            tile_index[i] = 0ull;
        }
    }
    __syncthreads();
    async_proxy_fence();

    const size_t stride = gridDim.x;
    const size_t first_tile = blockIdx.x;

    const unsigned long long tiles_for_block =
        (total_tiles <= first_tile)
            ? 0ull
            : ((static_cast<unsigned long long>(total_tiles - first_tile) +
                static_cast<unsigned long long>(stride) - 1ull) /
               static_cast<unsigned long long>(stride));

    if (tiles_for_block == 0) {
        return;
    }

    if (warp_id == 0) {
        size_t tile = first_tile;
        unsigned long long issued = 0;
        while (tile < total_tiles) {
            const int buffer = static_cast<int>(issued % PIPE_DEPTH);
            bf16 *buffer_ptr =
                smem_tiles + buffer * static_cast<size_t>(TILE_ELEMS);
            uint64_t *bar = &mbarriers[buffer];

            if (lane_id == 0) {
                while (buffer_state[buffer] != 0) {
                }
                const int tile_col =
                    static_cast<int>(tile % static_cast<size_t>(tiles_per_row));
                const int tile_row =
                    static_cast<int>(tile / static_cast<size_t>(tiles_per_row));
                tile_index[buffer] = static_cast<unsigned long long>(tile);

                const int c0 = tile_col * TILE_COLS;
                const int c1 = tile_row * TILE_ROWS;

                expect_bytes_and_arrive(bar, TILE_BYTES);
                cp_async_bulk_tensor_2d_global_to_shared(
                    buffer_ptr, &tensor_map, c0, c1, bar);
                __threadfence_block();
                buffer_state[buffer] = 1;
            }
            issued++;
            tile += stride;
        }
    } else if (warp_id == 1) {
        unsigned long long consumed = 0;
        while (consumed < tiles_for_block) {
            const int buffer = static_cast<int>(consumed % PIPE_DEPTH);
            bf16 *buffer_ptr =
                smem_tiles + buffer * static_cast<size_t>(TILE_ELEMS);
            uint64_t *bar = &mbarriers[buffer];

            if (lane_id == 0) {
                while (buffer_state[buffer] != 1) {
                }
                const unsigned long long tile =
                    tile_index[buffer];
                const int tile_col =
                    static_cast<int>(tile % static_cast<unsigned long long>(tiles_per_row));
                const int tile_row =
                    static_cast<int>(tile / static_cast<unsigned long long>(tiles_per_row));
                const int c0 = tile_col * TILE_COLS;
                const int c1 = tile_row * TILE_ROWS;

                const int phase = barrier_phase[buffer];
                wait(bar, phase);
                barrier_phase[buffer] = phase ^ 1;

                cp_async_bulk_tensor_2d_shared_to_global(
                    &dest_tensor_map, c0, c1, buffer_ptr);
                tma_commit_group();
                tma_wait_until_pending<1>();
                __threadfence_block();
                buffer_state[buffer] = 0;
            }
            consumed++;
        }

        if (lane_id == 0) {
            tma_wait_until_pending<0>();
        }
    }

    __syncthreads();
}

void launch_multiwarp_pipeline(bf16 *dest, bf16 *src, const int N) {
    /*
     * IMPORTANT REQUIREMENT FOR PART 4:
     *
     * To receive credit for this part, you MUST launch the kernel with maximum
     * shared memory allocated.
     *
     * Use cudaFuncSetAttribute() with
     * cudaFuncAttributeMaxDynamicSharedMemorySize to configure the maximum
     * available shared memory before launching the kernel, and then **launch**
     * it with the maximum amount.
     */

    constexpr int TILE_COLS = 128;
    constexpr int TILE_ROWS = 64;
    constexpr int PIPE_DEPTH = 4;
    constexpr size_t TILE_ELEMS = static_cast<size_t>(TILE_COLS) * TILE_ROWS;

    CUDA_CHECK(cuInit(0));

    CUtensorMap src_map;
    CUtensorMap dest_map;

    const int cols = TILE_COLS;
    const int rows = N / cols;

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

    int max_smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_optin,
                                      cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      0));

    size_t required_tiles_bytes =
        sizeof(bf16) * TILE_ELEMS * PIPE_DEPTH;
    size_t required_barriers_bytes =
        ((required_tiles_bytes + 127) & ~static_cast<size_t>(127));
    required_barriers_bytes =
        ((required_barriers_bytes + sizeof(uint64_t) * PIPE_DEPTH + 7) &
         ~static_cast<size_t>(7));
    const size_t required_total =
        max_smem_optin; // launch reserves maximum per requirement
    (void)required_tiles_bytes;
    (void)required_barriers_bytes;

    CUDA_CHECK(cudaFuncSetAttribute(
        tma_multiwarp_pipeline, cudaFuncAttributeMaxDynamicSharedMemorySize,
        max_smem_optin));

    const int tiles_per_row =
        (cols + TILE_COLS - 1) / TILE_COLS;
    const int row_tiles = (rows + TILE_ROWS - 1) / TILE_ROWS;
    const size_t total_tiles =
        static_cast<size_t>(tiles_per_row) * static_cast<size_t>(row_tiles);

    const int max_blocks = sm_count * 4;
    const int grid_blocks =
        static_cast<int>(std::min(total_tiles, static_cast<size_t>(max_blocks)));

    dim3 block_dim(128);
    dim3 grid_dim(grid_blocks);

    tma_multiwarp_pipeline<<<grid_dim, block_dim, required_total>>>(
        src_map, dest_map, N);
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
    launch_multiwarp_pipeline(d_dest, d_src, size);
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
        BENCHMARK_KERNEL((launch_multiwarp_pipeline(d_dest, d_src, size)),
                         num_iters, size_bytes, "TMA Copy");
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