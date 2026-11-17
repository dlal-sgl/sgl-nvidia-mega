// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

extern void k31_q5_launch(const void*, const void*, const float*, const int*,
                         const int*, const int*, float*, int, int, int, int, int,
                         bool, cudaStream_t);

static void run(const at::Tensor& q, const at::Tensor& kv, const at::Tensor& weights,
                const at::Tensor& contexts, const at::Tensor& table,
                const at::Tensor& schedule, const at::Tensor& output, bool pdl) {
    const c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == at::kFloat8_e4m3fn &&
                q.dim() == 4 && q.size(0) >= 1 && q.size(0) <= 8 &&
                q.size(1) == 5 && q.size(2) == 32 && q.size(3) == 128, "Q5 query contract");
    const int64_t n = q.size(0), rows = n * 5;
    TORCH_CHECK(kv.scalar_type() == at::kByte && kv.is_contiguous() && kv.dim() == 4 &&
                kv.size(1) == 64 && kv.size(2) == 1 && kv.size(3) == 132, "KV contract");
    TORCH_CHECK(weights.scalar_type() == at::kFloat && weights.is_contiguous() &&
                weights.sizes() == at::IntArrayRef({rows, 32}), "weights contract");
    TORCH_CHECK(contexts.scalar_type() == at::kInt && contexts.is_contiguous() &&
                contexts.sizes() == at::IntArrayRef({n, 5}), "contexts contract");
    TORCH_CHECK(table.scalar_type() == at::kInt && table.dim() == 2 && table.size(0) == n &&
                table.stride(1) == 1, "table contract");
    TORCH_CHECK(schedule.scalar_type() == at::kInt && schedule.is_contiguous() &&
                schedule.dim() == 2 && schedule.size(1) == 2 && schedule.size(0) > 1, "schedule contract");
    TORCH_CHECK(output.scalar_type() == at::kFloat && output.is_contiguous() &&
                output.dim() == 2 && output.size(0) == rows && output.size(1) % 256 == 0, "output contract");
    for (const auto* t : {&kv, &weights, &contexts, &table, &schedule, &output})
        TORCH_CHECK(t->device() == q.device(), "all inputs must share the CUDA device");
    k31_q5_launch(q.data_ptr(), kv.data_ptr(), weights.data_ptr<float>(), contexts.data_ptr<int>(),
                  table.data_ptr<int>(), schedule.data_ptr<int>(), output.data_ptr<float>(),
                  rows, kv.size(0), output.stride(0), table.stride(0), schedule.size(0) - 1,
                  pdl, c10::cuda::getCurrentCUDAStream(q.get_device()).stream());
}
TORCH_LIBRARY(k31_q5_score_singleblock_r1r8_v2, m) {
    m.def("run(Tensor q, Tensor kv, Tensor weights, Tensor contexts, Tensor table, Tensor schedule, Tensor(a!) output, bool pdl) -> ()", &run);
}
