// SPDX-License-Identifier: Apache-2.0
// New host adapter following official-3bd6c364 runtime_utils.hpp and paged host.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <deep_gemm/impls/sm100_mqa_logits.cuh>

static void driver_check(CUresult r) {
    if (r != CUDA_SUCCESS) { const char* s = nullptr; cuGetErrorString(r, &s);
        throw std::runtime_error(std::string("Q5 TMA: ") + (s ? s : "unknown")); }
}
static void runtime_check(cudaError_t r) {
    if (r != cudaSuccess) throw std::runtime_error(std::string("Q5 launch: ") + cudaGetErrorString(r));
}
static CUtensorMap tma(void* ptr, CUtensorMapDataType dtype, unsigned rank,
                       const cuuint64_t* dims, const cuuint64_t* strides,
                       const cuuint32_t* box, CUtensorMapSwizzle swizzle) {
    CUtensorMap map;
    const cuuint32_t elem_strides[3] = {1,1,1};
    driver_check(cuTensorMapEncodeTiled(&map, dtype, rank, ptr, dims, strides, box, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

void k31_q5_launch(const void* q, const void* kv, const float* weights, const int* contexts,
                  const int* table, const int* schedule, float* output,
                  int rows, int pages, int logits_stride, int table_stride, int sms,
                  bool pdl, cudaStream_t stream) {
    constexpr unsigned Q_STAGES = 3, KV_STAGES = 4;
    using Shared = deep_gemm::layout::MQALogitsSharedStorage<32,128,false,5,256,
                        Q_STAGES,KV_STAGES,3,cutlass::float_e4m3_t,float>;
    static_assert(sizeof(Shared) == 199680);
    constexpr auto kernel = deep_gemm::k31_q5_score_singleblock_r5_v1<
        5,32,128,64,false,true,false,Q_STAGES,KV_STAGES,256,16,128,256,
        cutlass::float_e4m3_t,float,float,2>;
    const cuuint64_t q_dims[2] = {128, static_cast<cuuint64_t>(rows)*32}, q_stride[1] = {128};
    const cuuint32_t q_box[2] = {128, 160};
    auto q_map = tma(const_cast<void*>(q), CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
                    q_dims, q_stride, q_box, CU_TENSOR_MAP_SWIZZLE_128B);
    const cuuint64_t kv_dims[3] = {128,64,static_cast<cuuint64_t>(pages)}, kv_stride[2] = {128,8448};
    const cuuint32_t kv_box[3] = {128,64,1};
    auto kv_map = tma(const_cast<void*>(kv), CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
                     kv_dims, kv_stride, kv_box, CU_TENSOR_MAP_SWIZZLE_128B);
    const cuuint64_t sf_dims[2] = {64,static_cast<cuuint64_t>(pages)}, sf_stride[1] = {8448};
    const cuuint32_t sf_box[2] = {64,1};
    auto sf_map = tma(static_cast<char*>(const_cast<void*>(kv))+8192,
                     CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,sf_dims,sf_stride,sf_box,CU_TENSOR_MAP_SWIZZLE_NONE);
    const cuuint64_t w_dims[2] = {32,static_cast<cuuint64_t>(rows)}, w_stride[1] = {128};
    const cuuint32_t w_box[2] = {32,5};
    auto w_map = tma(const_cast<float*>(weights),CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,
                    w_dims,w_stride,w_box,CU_TENSOR_MAP_SWIZZLE_NONE);
    runtime_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared)));
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(sms); config.blockDim = dim3(384);
    config.dynamicSmemBytes = sizeof(Shared); config.stream = stream;
    cudaLaunchAttribute attr{};
    if (pdl) { attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1; config.attrs = &attr; config.numAttrs = 1; }
    runtime_check(cudaLaunchKernelEx(&config, kernel, uint32_t(rows), uint32_t(logits_stride), uint32_t(table_stride),
        reinterpret_cast<const uint32_t*>(contexts), output, reinterpret_cast<const uint32_t*>(table),
        static_cast<const uint32_t*>(nullptr), reinterpret_cast<const uint32_t*>(schedule),
        q_map, sf_map, kv_map, sf_map, w_map));
}
