# SPDX-License-Identifier: Apache-2.0
"""Pinned FlashInfer 0.7.2 R3-R6/Q5 reducer experiment; the default route never imports it."""

from functools import cache

import torch


@cache
def _module():
    from flashinfer.jit.attention.modules import gen_trtllm_gen_fmha_module
    from flashinfer.mla._core import setup_cubin_loader

    spec = gen_trtllm_gen_fmha_module(k31_reducer_r3_r6_q5=True)
    op = spec.build_and_load()
    for path in spec.get_library_paths():
        setup_cubin_loader(path)
    return op


def run_reducer_mla(owner, cache, local_rows, counts, scales):
    from flashinfer.mla._core import (
        TrtllmGenMlaDecodeRunner,
        _check_trtllm_gen_mla_shape,
        get_device_sm_count,
        log2e,
    )

    query = owner.query
    rows = query.size(0)
    if rows not in (15, 20, 25, 30) or tuple(query.shape[1:]) != (1, 128, 576):
        raise ValueError("K31 reducer requires 15/20/25/30 rows and 128 padded heads")
    if query.dtype != torch.float8_e4m3fn or tuple(local_rows.shape) != (rows, 2048):
        raise ValueError("K31 reducer requires FP8 query and 2048 sparse slots per row")
    if tuple(counts.shape) != (rows,) or counts.dtype != torch.int32:
        raise ValueError("K31 reducer requires one int32 local KV count per row")
    kv_cache = cache.view(torch.float8_e4m3fn).unsqueeze(1)
    if kv_cache.size(-2) != 64:
        raise ValueError("K31 reducer requires the original page64 KV pool")
    block_tables = local_rows.unsqueeze(1)
    kv_cache = _check_trtllm_gen_mla_shape(
        query, kv_cache, 512, 64, 2048, block_tables, 64,
        require_aligned_block_table=False,
    )
    out = torch.empty((rows, 1, 128, 512), dtype=torch.bfloat16, device=query.device)
    lse = torch.empty((rows, 128), dtype=torch.float32, device=query.device)
    # Match the original public dispatcher: tensor BMM1 scales are converted to
    # log2 on device; scalar conversion remains in the unchanged C++ launcher.
    bmm1_scale = scales[0] * log2e if isinstance(scales[0], torch.Tensor) else scales[0]
    runner = TrtllmGenMlaDecodeRunner(
        kv_cache=kv_cache, workspace_buffer=owner.workspace,
        sm_count=get_device_sm_count(query.device), qk_nope_head_dim=128,
        kv_lora_rank=512, qk_rope_head_dim=64, max_seq_len=owner.max_context,
        sparse_mla_top_k=2048, bmm1_scale=bmm1_scale, bmm2_scale=scales[1],
        sinks=None, skip_softmax_threshold_scale_factor=None, enable_pdl=True,
        is_var_seq=True, uses_shared_paged_kv_idx=True, return_lse=True,
        lse=lse, return_lse_base="base2",
    )
    # A distinct module/compile macro selects the reducer; no global monkeypatch
    # or environment-dependent FlashInfer dispatch can change the stock anchor.
    runner._run = _module().trtllm_paged_attention_decode
    runner.forward([query, block_tables, counts, out],
                   multi_ctas_kv_counter_buffer=owner.counter)
    return out, lse
