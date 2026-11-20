# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""The DFlash runtime of the Kimi K3.1 target (``BuiltK31Verify.dflash_runtime``).

SGLang's DFlash worker binds it through ``sglang.srt.kernel_plugins.nvidia.kimi_k31_dflash_binding``
and hands it three whole phases; it subclasses nothing of SGLang's:

* ``__init__(binding, built)``: the end of the worker's ``__init__``, before the KV pools
  are sized (the sampled accept's build comes out of the KV budget). Collective.
* ``draft_sampler(lm_head)``: before the draft cuda graph captures, the megakernel head
  folded into it (stage 2a). Collective.
* ``decode_round(batch, ...)``: the whole decode round, on every target TP rank:

      draft graph replay (MK chain + head) -> target verify graph -> sampled accept on
      the kept vocab shard (or the greedy accept on its argmax) -> simulated acceptance
      -> KDA commit -> publish -> aux graph (main stream)

  It returns None, launching nothing, for a round it does not serve (grammar, logprobs,
  top-k/top-p, adjusted logits, sampling masks, a batch size off its graphs); SGLang
  then runs its own round. Every condition is the same on every rank.

It reads the worker only through the binding's exports (``exports``) and states the
setup it serves as SGLang guard values (``invariants``); on any other setup SGLang
declines it at startup (one warning naming each mismatch) and runs its stock DFlash
round beside the target megakernel. The round is SGLang's stock round (b1b3173c7c's
DFlashRuntime steps with 20c91c3/9f1bb9f's overrides) specialized to that setup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch
from sglang.kernels.ops.memory.small_copy import try_small_copy
from sglang.kernels.ops.speculative.dflash import (
    _prepare_dflash_draft_block_unchecked,
)
from sglang.srt.kernel_plugins.nvidia.binding import (
    built_megakernel,
    megakernel_agree,
)
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import apply_dflash_simulated_acceptance
from sglang.srt.speculative.dflash_worker_v2 import (
    _commit_accept,
    _is_all_greedy,
    _resolve_dflash_embedding_module,
    _verify_logits_unadjusted,
)
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    SIMULATE_ACC_METHOD,
    SIMULATE_ACC_TOKEN_MODE,
    draft_tp_context,
)

from .aux_graphs import AuxGraphs
from .commit import KdaCommit
from .protocol import DRAFT_CHAINS, NO_DRAFT_CHAIN, offered
from .sampler import MegakernelSelectorDraftSampler

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

# The SGLang names this module reads, by file under the sglang package; test/unit/test_contract.py pins them.
SGLANG_READS = {
    "kernels/ops/memory/small_copy.py": ("try_small_copy",),
    "kernels/ops/speculative/dflash.py": ("_prepare_dflash_draft_block_unchecked",),
    "srt/kernel_plugins/nvidia/binding.py": ("built_megakernel", "megakernel_agree"),
    "srt/lora/layers.py": ("unwrap_lora_layer",),
    "srt/managers/utils.py": ("GenerationBatchResult",),
    "srt/mem_cache/kv/alloc/id_space.py": ("IdSpaceKind", "SLIDING_WINDOW"),
    "srt/mem_cache/kv/alloc/iteration_ids.py": ("write_ids", "reads_from", "bind"),
    "srt/mem_cache/kv/alloc/runner_ids.py": ("plan", "space"),
    "srt/model_executor/forward_batch_info.py": (
        "ForwardBatch",
        "enable_num_token_non_padded",
    ),
    "srt/model_executor/forward_mode.py": ("CaptureHiddenMode", "ForwardMode"),
    "srt/runtime_context.py": ("get_exec", "get_parallel"),
    "srt/speculative/dflash_info.py": ("DFlashVerifyInput", "prepare_for_verify"),
    "srt/speculative/dflash_utils.py": ("apply_dflash_simulated_acceptance",),
    "srt/speculative/dflash_worker_v2.py": (
        "_commit_accept",
        "_is_all_greedy",
        "_resolve_dflash_embedding_module",
        "_verify_logits_unadjusted",
    ),
    "srt/speculative/draft_worker_common.py": ("make_draft_input_v2",),
    "srt/speculative/spec_info.py": ("SpeculativeAlgorithm",),
    "srt/speculative/spec_utils.py": (
        "SIMULATE_ACC_LEN",
        "SIMULATE_ACC_METHOD",
        "SIMULATE_ACC_TOKEN_MODE",
        "draft_tp_context",
    ),
    "srt/layers/logits_processor.py": (
        "vocab_size",
        "final_logit_softcapping",
        "defer_plain_gather",
        "is_deferred_shard",
        "deferred_argmax",
    ),
    "srt/layers/vocab_parallel_embedding.py": ("num_embeddings_per_partition",),
    "srt/models/dflash.py": ("candidate_selector", "draft_config", "top_k"),
    "srt/managers/schedule_batch.py": (
        "ScheduleBatch",
        "has_grammar",
        "return_logprob",
        "can_run_decode_cuda_graph",
        "out_cache_loc",
    ),
    "srt/sampling/sampling_batch_info.py": (
        "temperatures",
        "need_top_k_sampling",
        "need_top_p_sampling",
    ),
}
# SGLang code moved here (test/unit/test_contract.py): while SGLang still defines it,
# its digest must match. _arm_plain_deferral and _prepare_megakernel_sampled_accept are
# copies (the deferral refusal is a guard key now); the head attach, the aux offer and
# the sampled-accept gate are specialized into draft_sampler, __init__ and _serves.
SGLANG_MOVED = {
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._prepare_megakernel_sampled_accept": "f1c3d06541d47fdf",
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._plain_deferral_refused": "a926ed410dfdf354",
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._arm_plain_deferral": "a4940f4de693d0f9",
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._maybe_attach_megakernel_head": "7a08ff6f804278e0",
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._maybe_megakernel_aux": "092aaae84c5d6358",
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._can_megakernel_sample": "b4b2f0926f05c18f",
}
# What this runtime pulls from SGLang's DFlash worker (kimi_k31_dflash_binding.EXPORTS)
# and the guard values it serves (kimi_k31_dflash_binding.GUARD);
# test/unit/test_contract.py pins both against SGLang's binding.
DFLASH_RUNTIME_API = 1
DFLASH_EXPORTS = frozenset(
    {
        "block_size",
        "device",
        "mask_token_id",
        "noise_embed_scale",
        "batched_small_copy",
        "draft_model",
        "draft_model_runner",
        "draft_parallel_layout",
        "draft_block_spec_info",
        "target_worker",
        "target_logits_processor",
        "sim_acc_rng",
        "need_mamba_verify_commit",
        "compute_compact_draft_seq_lens",
        "fill_compact_seq_lens_cpu_bound",
        "rebuild_compact_draft_cache",
        "accept_block",
    }
)
# The measured setup: TP16 GB300, B1 x 8 rows, selector sampling, compact draft window,
# FP8 draft KV, KDA target. SGLang declines the runtime on any other setup (e.g. TP8,
# decode graph bs [1, 2, 4]) and runs its stock round; K31MK_DFLASH_RUNTIME=0 always does.
DFLASH_INVARIANTS = {
    "platform": "cuda",
    "tp_size": 16,
    "pp_size": 1,
    "attn_dp": False,
    "dtype": "torch.bfloat16",
    "page_size": 64,
    "recurrent_target": True,
    "draft_architectures": ("DFlash2DraftModel",),
    "draft_kv_cache_dtype": "fp8_e4m3",
    "block_size": 8,
    "compact_draft_cache": True,
    "draft_kv_ring": False,
    "selector_draft": True,
    "selector_sampling": True,
    "lilicorr": False,
    "domino": False,
    "block_verification": False,
    "replicated_embed": False,
    "aux_graph": True,
    "decode_cuda_graph_bs": (1,),
    "ragged_verify_mode": "static",
    "eager_draft_sampler": False,
    "defer_plain_gather": frozenset({None, True}),
    "info_dump": False,
    "hierarchical_cache": False,
    "hisparse": False,
    "lora": False,
}

logger = logging.getLogger(__name__)


class K31DFlashRuntime(AuxGraphs, KdaCommit):
    """One per DFlash worker, over the target's ``BuiltK31Verify``."""

    api_version = DFLASH_RUNTIME_API
    exports = DFLASH_EXPORTS
    invariants = DFLASH_INVARIANTS

    def __init__(self, binding: Any, built: Any) -> None:
        pull = binding.pull
        self._target = built
        self._target_tp_rank = get_parallel().tp_rank
        self.device = pull("device")
        self.block_size = pull("block_size")
        self.draft_model = pull("draft_model")
        self.selector = self.draft_model.candidate_selector
        self._target_worker = pull("target_worker")
        self._target_model = self._target_worker.model_runner.model
        self._target_logits_processor = pull("target_logits_processor")
        self._mask_token_id = pull("mask_token_id")
        self._noise_embed_scale = pull("noise_embed_scale")
        self._batched_small_copy = pull("batched_small_copy")
        self._draft_parallel_layout = pull("draft_parallel_layout")
        self._draft_block_spec_info = pull("draft_block_spec_info")
        self._simulated_acc_len_rng = pull("sim_acc_rng")
        self._compute_compact_draft_seq_lens = pull("compute_compact_draft_seq_lens")
        self._fill_compact_seq_lens_cpu_bound = pull("fill_compact_seq_lens_cpu_bound")
        self._rebuild_compact_draft_cache = pull("rebuild_compact_draft_cache")
        self._accept_block = pull("accept_block")
        self._binding = binding
        # bound by draft_sampler, before the draft graph captures
        self._sampler = None
        self._draft_runner = None
        self._round_bs = frozenset()
        self._embed_module = None
        self._block_bufs = None
        self._megakernel_aux_graphs = {}
        self._megakernel_aux_caches = None
        # Megakernel mailbox slot of the aux: alternates on every call, eager or replayed,
        # in the same order on every rank.
        self._aux_parity = 0
        self._kda_commit_buffers = None
        draft_built = built_megakernel(self.draft_model)
        self._draft = NO_DRAFT_CHAIN
        if any(chain is draft_built for chain in DRAFT_CHAINS):
            self._draft = draft_built
        draft = self._draft
        # Ranks that differ would wait in different collectives below.
        megakernel_agree(
            (
                draft.draft_aux_rows,
                draft.draft_head_requests,
                draft.draft_accept_requests,
                draft.draft_embed_requests,
            ),
            what="DFlash draft chain offers",
        )
        self._megakernel_aux = offered(draft, draft.draft_aux_rows)
        _require(
            self._megakernel_aux is not None,
            "the draft is not on the megakernel chain, or its aux is off "
            "(SGLANG_K31_DRAFT_MK, SGLANG_K31_DRAFT_MK_AUX)",
        )
        _require(
            not draft.draft_accept_requests and not draft.draft_embed_requests,
            "the opt-in megakernel accept and embed are not served "
            "(unset SGLANG_K31_DRAFT_MK_ACCEPT and SGLANG_K31_DRAFT_MK_EMBED)",
        )
        # Fixed at build (K31MK_KDA_COMMIT).
        _require(built.kda_commit_enabled, "the KDA commit is off (K31MK_KDA_COMMIT)")
        self._megakernel_kda = built
        self._kda_confirmed = True
        self._megakernel_sampled = self._prepare_megakernel_sampled_accept()
        _require(
            self._megakernel_sampled is not None,
            "the sampled accept did not build (K31MK_SAMPLED_ACCEPT, selector K, "
            "shard width)",
        )
        if self._target_tp_rank == 0:
            logger.info(
                "DFLASH aux append runs on the NVIDIA megakernel for rows %s, "
                "on the main stream",
                list(self._megakernel_aux.draft_aux_rows),
            )

    # ---- init phases

    def _prepare_megakernel_sampled_accept(self):
        target_lp = self._target_logits_processor
        built = self._target
        # Declined before the build: it is collective.
        if built is None or target_lp is None:
            return None
        lm_head = unwrap_lora_layer(self._target_model.lm_head)
        if not built.prepare_sampled_accept(
            vocab_size=int(target_lp.vocab_size),
            softcap=target_lp.final_logit_softcapping or None,
            shard_width=int(lm_head.num_embeddings_per_partition),
            seed=int(torch.initial_seed()),
            topk=int(self.selector.top_k),
        ):
            return None
        self._arm_plain_deferral("sampled accept")
        return built

    def _arm_plain_deferral(self, consumer: str) -> None:
        """Must run before the graphs capture."""
        target_lp = self._target_logits_processor
        if (
            target_lp is None
            or target_lp.final_logit_softcapping
            or target_lp.defer_plain_gather
        ):
            return
        target_lp.defer_plain_gather = True
        if self._target_tp_rank == 0:
            logger.info(
                "DFLASH plain verify deferral on (the %s reads the shard)", consumer
            )

    def draft_sampler(self, lm_head: Any) -> MegakernelSelectorDraftSampler:
        """The megakernel head, which proposes inside the draft forward; before the
        draft graph captures (after the worker's attention backends). Collective."""
        _require(
            self._binding.pull("need_mamba_verify_commit"),
            "the target has no recurrent state to commit",
        )
        self._draft_runner = self._binding.pull("draft_model_runner")
        # The aux graphs write only the full pool's slots.
        _require(
            self._draft_runner.runner_ids.space(IdSpaceKind.SLIDING_WINDOW) is None,
            "the draft KV pool has a sliding-window sub-pool",
        )
        self._draft_kv_pool = self._draft_runner.token_to_kv_pool
        self._target_attn_backend = self._target_worker.model_runner.attn_backend
        self._embed_module = unwrap_lora_layer(
            _resolve_dflash_embedding_module(self.draft_model, self._target_model)
        )
        head = offered(self._draft, self._draft.draft_head_requests)
        graph_bs = set(get_exec().graph.cuda_graph_config.decode.bs)
        _require(
            head is not None
            and graph_bs <= set(head.draft_head_requests)
            and head.draft_head_sampling,
            f"the megakernel head does not serve the decode graphs {sorted(graph_bs)} "
            "with sampling (SGLANG_K31_DRAFT_MK_HEAD)",
        )
        rows = set(self._megakernel_aux.draft_aux_rows)
        self._round_bs = frozenset(
            bs for bs in graph_bs if bs * int(self.block_size) in rows
        )
        tokens, candidates, q = head.attach_draft_head(
            lm_head=lm_head,
            selector=self.draft_model.candidate_selector,
            draft_config=self.draft_model.draft_config,
        )
        if self._target_tp_rank == 0:
            logger.info(
                "DFLASH selector decode on the NVIDIA megakernel head (%s).",
                "greedy and sampling requests",
            )
        self._sampler = MegakernelSelectorDraftSampler(
            head=head,
            out=tokens,
            candidate_out=candidates,
            q_out=q,
            sampling_enabled=True,
        )
        return self._sampler

    # ---- the round

    def decode_round(
        self, batch: ScheduleBatch, *, on_publish, grammar_barrier
    ) -> Optional[GenerationBatchResult]:
        """The whole decode round, or None (nothing launched) for SGLang's."""
        if self._sampler is None:
            raise RuntimeError(
                "K3.1 DFlash runtime: the draft graph was not captured, so the "
                "megakernel head never attached (cuda graphs off or out of memory?)"
            )
        sampling_info = batch.sampling_info
        bs = len(batch.seq_lens)
        sampled = not _is_all_greedy(sampling_info)
        if not self._serves(batch, sampling_info, bs=bs, sampled=sampled):
            return None
        return self._round(
            batch, sampling_info, bs=bs, sampled=sampled, on_publish=on_publish
        )

    def _serves(self, batch, sampling_info, *, bs: int, sampled: bool) -> bool:
        """Rank-uniform, before any launch: the round runs here, on the draft graph."""
        if (
            bs not in self._round_bs
            or not batch.can_run_decode_cuda_graph
            or batch.has_grammar
            or batch.return_logprob
        ):
            return False
        if sampling_info is None:
            return True
        # Nor penalties, logit bias, custom processors or sampling masks.
        if not _verify_logits_unadjusted(sampling_info):
            return False
        return not sampled or (
            not sampling_info.need_top_k_sampling
            and not sampling_info.need_top_p_sampling
            and self._megakernel_sampled.sampled_accept_fits(bs)
        )

    def _block_buffers(self, bs: int) -> tuple:
        bufs = self._block_bufs
        if bufs is None or bufs[0].shape[0] < bs:
            cap = max(bs, 0 if bufs is None else 2 * bufs[0].shape[0])
            block, device = int(self.block_size), self.device
            bufs = self._block_bufs = (
                torch.empty((cap, block), dtype=torch.long, device=device),
                torch.empty((cap, block), dtype=torch.int64, device=device),
                torch.empty((cap, block), dtype=torch.int64, device=device),
                torch.empty((cap, block), dtype=torch.long, device=device),
                torch.empty((cap,), dtype=torch.int32, device="cpu"),
            )
        return tuple(buf[:bs] for buf in bufs)

    def _round(self, batch, sampling_info, *, bs, sampled, on_publish):
        # -- begin_round
        # `seq_lens` is carried over from the previous overlap iteration and may have been
        # produced on another stream.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        device = self.device
        draft_input = batch.spec_info
        target_runner = self._target_worker.model_runner
        # -- build_block
        block_size = int(self.block_size)
        block_ids, positions_2d, verify_out_cache_loc_2d, draft_tokens, seq_lens_cpu = (
            self._block_buffers(bs)
        )
        prefix_lens = batch.seq_lens
        _prepare_dflash_draft_block_unchecked(
            bonus_tokens=draft_input.bonus_tokens.view(-1),
            prefix_lens=prefix_lens.view(-1),
            req_pool_indices=batch.req_pool_indices.view(-1),
            req_to_token=target_runner.req_to_token_pool.req_to_token,
            block_ids_out=block_ids,
            positions_out=positions_2d,
            cache_loc_out=verify_out_cache_loc_2d,
            mask_token_id=int(self._mask_token_id),
        )
        # -- embed_block
        noise_embedding = self._embed_module(block_ids)
        if self._noise_embed_scale != 1.0:
            noise_embedding = noise_embedding * self._noise_embed_scale
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])
        # -- plan_draft
        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)
        iteration_ids = target_runner.runner_ids.plan(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            seq_lens_cpu=(
                batch.seq_lens_cpu
                if batch.seq_lens_cpu is not None
                else draft_input.nxt_kv_lens_cpu
            ),
            write_virtual=verify_out_cache_loc,
            read_extent=block_size,
        )
        # Rebuild the draft-local sliding-window view from committed target state.
        draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
        self._fill_compact_seq_lens_cpu_bound(
            batch_seq_lens_cpu=batch.seq_lens_cpu,
            nxt_kv_lens_cpu=draft_input.nxt_kv_lens_cpu,
            out=seq_lens_cpu,
        )
        self._rebuild_compact_draft_cache(
            req_pool_indices=batch.req_pool_indices,
            prefix_lens=prefix_lens,
            draft_prefix_lens=draft_prefix_lens,
            verify_out_cache_loc_2d=verify_out_cache_loc_2d,
            bs=bs,
            block_size=block_size,
        )
        draft_seq_lens = draft_prefix_lens
        draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=(
                torch.full((), bs * block_size, dtype=torch.int32, device=device)
                if enable_num_token_non_padded()
                else None
            ),
            global_num_token_non_padded_cpu=bs * block_size,
            can_run_decode_cuda_graph=batch.can_run_decode_cuda_graph,
        )
        draft_translator = self._draft_runner.runner_ids
        draft_plan = iteration_ids.reads_from(
            draft_translator,
            seq_lens=draft_seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            read_extent=block_size,
        )
        draft_plan.bind(forward_batch, draft_translator)
        # -- draft
        sampler = self._sampler
        # Consumed by the in-graph sample; must be staged before the replay.
        sampler.stage_sampling_params(bs=bs, sampling_info=sampling_info)
        with (
            torch.inference_mode(),
            draft_tp_context(self._draft_parallel_layout),
        ):
            draft_out = self._draft_runner.forward(forward_batch)
        if not draft_out.can_run_graph:
            raise RuntimeError(f"K3.1 DFlash runtime: the draft ran eager at bs={bs}")
        draft_next = sampler.out[: bs * (block_size - 1)].view(bs, block_size - 1)
        selector_sample = (
            (sampler.candidate_out[:bs], sampler.q_out[:bs]) if sampled else None
        )
        if not (
            bs == 1
            and self._batched_small_copy
            and try_small_copy(
                [draft_tokens[:, 0], draft_tokens[:, 1:]], [block_ids[:, 0], draft_next]
            )
        ):
            draft_tokens[:, 0].copy_(block_ids[:, 0])
            draft_tokens[:, 1:].copy_(draft_next)
        # -- verify
        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=block_size,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            iteration_ids=iteration_ids,
        )
        batch.out_cache_loc = verify_out_cache_loc
        seq_lens_pre_verify = batch.seq_lens.clone()
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if seq_lens_cpu_backup is not None:
            # Verify host bound = committed prefix + one verify block (matches draft).
            verify_host_seq_lens = seq_lens_cpu_backup + block_size
            batch.seq_lens_cpu = verify_host_seq_lens
            batch.seq_lens_sum = int(verify_host_seq_lens.sum())
        elif draft_input.nxt_kv_lens_cpu is not None:
            batch.seq_lens_cpu = draft_input.nxt_kv_lens_cpu
            batch.seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self._target_worker, defer_metadata_init=False
        )
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup
        target_out = self._target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = target_out.logits_output
        # -- resolve_verify_logits + accept_tokens
        accept_len, commit_lens, bonus, out_tokens, new_seq_lens = self._accept(
            batch,
            sampling_info,
            logits_output=logits_output,
            selector_sample=selector_sample,
            draft_tokens=draft_tokens,
            prefix_lens=prefix_lens,
            bs=bs,
        )
        # -- commit
        if new_seq_lens is None:
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        _require(
            self._megakernel_kda_commit(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            ),
            "the KDA commit declined the target's recurrent state",
        )
        # -- publish
        if on_publish is not None:
            on_publish(new_seq_lens)
        # -- append_aux
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, block_size, -1)
        # Consume in this step: every decode graph size shares one aux output,
        # which the next target forward overwrites (resolve_aux_hidden_states_width).
        cache_loc = iteration_ids.write_ids(draft_translator)
        self._append_target_hidden_megakernel(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            positions=positions,
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc.view(verify_out_cache_loc_2d.shape),
            commit_lens=commit_lens,
        )
        # -- finish
        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        logits_output.hidden_states = None
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            next_draft_input=make_draft_input_v2(
                bonus_tokens=bonus, new_seq_lens=new_seq_lens
            ),
            side_stream_done=None,
            speculative_num_draft_tokens=block_size,
            new_seq_lens=new_seq_lens,
            block_accept_lens=None,
            cap_lens=None,
            routed_experts_output=target_out.routed_experts_output,
            indexer_topk_output=target_out.indexer_topk_output,
            kv_weight_version_record=target_out.kv_weight_version_record,
        )

    def _accept(
        self,
        batch,
        sampling_info,
        *,
        logits_output,
        selector_sample,
        draft_tokens,
        prefix_lens,
        bs,
    ) -> tuple:
        """(accept_len, commit_lens, bonus, out_tokens, new_seq_lens or None) from the
        verify's vocab shard, which stays ungathered."""
        shard = logits_output.next_token_logits
        target_lp = self._target_logits_processor
        if not target_lp.is_deferred_shard(shard):
            raise RuntimeError("K3.1 DFlash runtime: the verify gathered its logits")
        logits_output.next_token_logits = None
        if selector_sample is None:
            # A greedy batch: SGLang's greedy accept on the shards' argmax.
            target_predict = target_lp.deferred_argmax(shard)
            return self._accept_block(
                candidates=draft_tokens,
                next_token_logits=None,
                sampling_info=sampling_info,
                draft_input=batch.spec_info,
                prefix_lens=prefix_lens,
                bs=bs,
                target_predict=target_predict,
                cutoff_verify_lens=None,
            )[:5]
        if not self._megakernel_sampled.sampled_accept_fits(bs, shard):
            raise RuntimeError(f"K3.1 DFlash runtime: a {shard.dtype} verify shard")
        # Samples on the vocab shards; every rank gets rank 0's decision.
        selector_candidate_ids, selector_q_rows = selector_sample
        accept_len, bonus, target_predict = self._megakernel_sampled.sampled_accept(
            logits=shard,
            temperatures=sampling_info.temperatures,
            candidates=draft_tokens,
            candidate_ids=selector_candidate_ids,
            q_rows=selector_q_rows,
        )
        out_tokens, commit_lens = _commit_accept(draft_tokens, accept_len, bonus)
        new_seq_lens = None
        if SIMULATE_ACC_LEN > 0:
            new_seq_lens = apply_dflash_simulated_acceptance(
                candidates=draft_tokens,
                target_predict=target_predict,
                accept_len=accept_len,
                commit_lens=commit_lens,
                bonus=bonus,
                out_tokens=out_tokens,
                simulate_acc_len=SIMULATE_ACC_LEN,
                simulate_acc_method=SIMULATE_ACC_METHOD,
                simulate_acc_token_mode=SIMULATE_ACC_TOKEN_MODE,
                next_token_logits=None,
                generator=self._simulated_acc_len_rng,
                prefix_lens=prefix_lens,
                new_seq_lens=new_seq_lens,
            )
        return accept_len, commit_lens, bonus, out_tokens, new_seq_lens


def _require(ok: bool, what: str) -> None:
    if not ok:
        raise RuntimeError(f"K3.1 DFlash runtime: {what}")
