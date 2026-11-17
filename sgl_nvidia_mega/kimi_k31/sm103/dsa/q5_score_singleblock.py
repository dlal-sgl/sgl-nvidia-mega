# SPDX-License-Identifier: Apache-2.0
"""Default-off R1-R8/Q5 FP32 scorer, with one query block per request.

The device core is copied from the installed 0.2.3.post2 headers. The new
host adapter follows the recipe-bound upstream TMA/launch contract; it is
not a recovered copy of the installed binary's host dispatcher.
"""
from functools import cache
import hashlib
import os
from pathlib import Path

import torch

JIT_NAME = "k31_q5_score_singleblock_r1r8_v2"


@cache
def _supported_device(device):
    return torch.cuda.get_device_capability(device) == (10, 3)


def eligible(query, gates, cache):
    return (
        query.is_cuda and gates.device == query.device and cache.device == query.device
        and _supported_device(query.device)
        and query.dtype == torch.float8_e4m3fn
        and query.ndim == 3 and 5 <= query.shape[0] <= 40 and query.shape[0] % 5 == 0
        and tuple(query.shape[1:]) == (32, 128) and query.stride() == (4096, 128, 1)
        and gates.dtype == torch.float32
        and tuple(gates.shape) == (query.shape[0], 32, 1) and gates.stride() == (32, 1, 1)
        and cache.dtype == torch.uint8 and cache.ndim == 2
        and cache.shape[0] > 0 and cache.shape[1] == 8448 and cache.stride() == (8448, 1)
        and all(x.data_ptr() % 16 == 0 for x in (query, gates, cache))
    )


@cache
def _load():
    import deep_gemm
    from torch.utils.cpp_extension import load, get_default_build_root

    src = Path(__file__).with_name("q5_score_singleblock_src")
    digest = hashlib.sha256()
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(src)).encode())
        digest.update(path.read_bytes())
    name = JIT_NAME + "_" + digest.hexdigest()[:16]
    build = Path(os.environ.get("K31_Q5_SCORE_CACHE_DIR", str(Path(get_default_build_root()) / JIT_NAME))) / name
    build.mkdir(parents=True, exist_ok=True)
    load(name=name, sources=[str(src / "binding.cpp"), str(src / "kernel.cu")],
         extra_include_paths=[str(src / "include"), str(Path(deep_gemm.__file__).parent / "include")],
         extra_cflags=["-O3", "-std=c++20"],
         extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_100f,code=sm_100f",
                           "--expt-relaxed-constexpr", "--expt-extended-lambda",
                           "--ptxas-options=--register-usage-level=10", "--ptxas-options=-v"],
         extra_ldflags=["-lcuda"], build_directory=str(build),
         is_python_module=False, verbose=True)
    return torch.ops.k31_q5_score_singleblock_r1r8_v2.run.default


def run_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta,
               max_context_len, clean_logits=False, indices=None):
    """DeepGEMM-compatible signature; unsupported metadata uses the original op."""
    import deep_gemm

    n = q.shape[0] if q.ndim == 4 else 0
    rows = 5 * n
    supported = (
        1 <= n <= 8 and tuple(q.shape[1:]) == (5, 32, 128) and q.is_contiguous()
        and kv_cache.ndim == 4 and tuple(kv_cache.shape[1:]) == (64, 1, 132)
        and kv_cache.is_contiguous() and weights.shape == (rows, 32) and weights.is_contiguous()
        and eligible(q.view(rows, 32, 128), weights.view(rows, 32, 1), kv_cache.view(-1, 8448))
        and indices is None and not clean_logits
        and context_lens.shape == (n, 5) and context_lens.dtype == torch.int32 and context_lens.is_contiguous()
        and block_table.ndim == 2 and block_table.shape[0] == n
        and block_table.dtype == torch.int32 and block_table.stride(1) == 1
        and schedule_meta.shape == (deep_gemm.get_num_sms() + 1, 2)
        and schedule_meta.dtype == torch.int32 and schedule_meta.is_contiguous()
        and max_context_len > 0 and max_context_len <= block_table.shape[1] * 64
        and all(x.device == q.device for x in (context_lens, block_table, schedule_meta))
    )
    if not supported:
        return deep_gemm.fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table,
                                             schedule_meta, max_context_len, clean_logits, indices)
    op = _load()
    stride = (max_context_len + 255) // 256 * 256
    output = torch.empty((rows, stride), dtype=torch.float32, device=q.device)
    op(q, kv_cache, weights, context_lens, block_table, schedule_meta, output, bool(deep_gemm.get_pdl()))
    return output[:, :max_context_len]
