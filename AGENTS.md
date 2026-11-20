# Agent guide

The rules for coding agents working in this repository, in three parts: the
dev guide (how to write a change), the contribution guide (how to submit it)
and the CI test guide (how to test it). Read this before editing.
[README.md](README.md) describes what each model's megakernel covers and how to
launch it; [CONTRIBUTING.md](CONTRIBUTING.md) is the contributor guide for
people and defines the accuracy and performance comparison protocol in detail.

This repository is an SGLang general plugin. `sgl_nvidia_mega/plugin.py` maps
each HF architecture to its spec (`SPECS`); SGLang imports a spec only when the
server runs with `--enable-nvidia-megakernel`. A spec admits or refuses the
setup, builds the megakernel after the model loads, and takes over the steps it
covers. Every other step runs SGLang's own code.

## 1. Dev guide

### What goes where

The megakernels so far have needed three kinds of change:

1. **The model's forward and its kernels:** the covered step itself, the fused
   kernels it runs and the build that prepares them.
2. **Speculative decoding:** the target-verify and draft rounds of DFlash,
   DSpark and MTP, the accepts and commits around them, and their kernels.
3. **Small optimizations:** improvements to the scheduler, PD disaggregation,
   HiCache, CUDA graph capture and replay, or an isolated fusion or removed
   copy, useful beyond one model's megakernel.

This repository holds the first two. They are large, model-specific and change
often; inside SGLang they would make the shared model and speculative code hard
to read, and every SGLang change would have to keep them working. So they live
here, and SGLang keeps only a binding:

- **Binding:** SGLang defines the whole-phase entry points it calls and the
  names the package may read; the package declares what it uses and the
  contract tests check both sides.
- **Guard:** the package decides from what its build offers whether it serves a
  setup or a step: it refuses an unservable setup at startup, or hands a step
  back to SGLang's own code before any launch and logs why.

Kimi K3.1's DFlash round is the reference. #77, with sgl-project/sglang#51676,
moved it out of SGLang's DFlash worker into `kimi_k31/dflash/`; SGLang kept one
binding module (`kernel_plugins/nvidia/kimi_k31_dflash_binding.py`) and calls
`decode_round`, which runs the whole round or returns `None` so SGLang runs its
own. #78 decides what the round serves from what the build offers.

Small optimizations are usually general, so they go straight to SGLang as
ordinary SGLang PRs and are extended to most setups, megakernel or not, rather
than kept on the megakernel path. Turn them on wherever they apply instead of
behind a switch.

The cost of this split: SGLang cannot see the code behind its bindings, so an
SGLang change can break this package without failing SGLang's own CI. The
contract tests and the E2E runs on SGLang `main` are what catch it; keep both
current with every change. The split is accepted because it keeps SGLang `main`
readable.

### Where code goes

- One package per model, `sgl_nvidia_mega/<model>/`, usually with `spec.py`
  (admission and build), `plan.py` (which kernel serves each row width),
  `runtime/` (build, executor, per-step state), the step packages (`attention/`,
  `moe/`, `boundary/`, ...) and one directory per GPU architecture (`sm103/`, ...)
  for kernels and launchers. Shared helpers go in `common/`. Read the model's
  own package before assuming this layout.
- Architecture directories do not overlap. GB300 (SM103) kernels, launchers and
  tactics live in `sm103/`, Rubin (SM107) ones in `sm107/`. A file in one
  directory never branches on the other's compute capability and never imports
  from it. When SM107 needs a kernel SM103 has, copy it into `sm107/` and change
  it there. The duplication is accepted: each architecture can then be tuned,
  rewritten or removed without touching the other. The step code chooses the
  directory by the device's architecture once, at build.
- Do not patch SGLang internals from this package. Hooks into SGLang's own
  forward go through its `HookRegistry`.

### The SGLang boundary

The package reads SGLang by name, so SGLang refactors break it.
`test/unit/test_contract.py` checks every declaration statically:

- A module that imports from `sglang` lists every name it reads in
  `SGLANG_READS`; an optional extension uses `SGLANG_OPTIONAL_READS`, and its
  tests cover the refusal when the extension is absent.
- Code that mirrors an SGLang function is listed in `SGLANG_MIRRORS`, and code
  moved out of SGLang into the package in `SGLANG_MOVED`, each with the digest
  of the SGLang code. When a digest fails, re-read the SGLang change and update
  the copy before updating the digest.
- A DFlash runtime declares the worker names it binds (`DFLASH_BINDINGS`) and
  any guard values (`DFLASH_GUARDS`); SGLang's binding must provide each.

### Switches and environment variables

A user turns the megakernel on with `--enable-nvidia-megakernel` and sets
nothing else. Do not add environment variables that select features.

- Do not add an environment variable that is always set one way. If every
  supported setup should run the new path, make it the only path and delete
  the old one. If no setup should, delete the new code instead of shipping it
  switched off.
- Remove the debug and A/B switches you added while developing before the PR is
  merged. Tuning values found by a sweep become constants or per-shape tables in
  the code.
- The only environment variables a change may add are test hooks: read by
  test-only code paths, set by the tests that need them, named with the
  `SGLANG_NVIDIA_MEGAKERNEL_` prefix and described in that test.
- An SGLang environment variable the model needs is declared in its spec
  (`Requirements.env`), so startup names it when it is missing. For the flag to
  set it for the user, it must be listed in SGLang's `FLAG_SETS_ENV`; that is an
  SGLang change.
- In the slot-based models (DeepSeek-V4.1-Flash, Qwen3.5, GLM-5.3),
  `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` replaces the shipped slot list for tests and
  bisecting; an empty value runs every step's stock code. It is not a user
  setting. Add a new kernel as a slot that runs by default.

### Where an optimization applies

Most optimizations apply to more setups than the one they were first measured
on. Decide where a path runs from what it needs, in code: the GPU, the TP size,
the row count it was built for, the cache layout it reads. Do not limit it to the
configuration a benchmark used, and do not copy a list of tested settings into a
guard. Widen coverage by validating more setups (CI test guide).

When a setup cannot run the path:

- If the megakernel cannot serve the setup at all, the spec refuses it at
  startup, listing every reason, before the weights load.
- If only part of the work cannot be served, hand that part back to SGLang's
  stock code before any launch, decided the same way on every rank, and log the
  reason once on TP rank 0. Never fall back silently.

### Invariants

- **Built at load.** Buffers, kernels and tactics for every covered width are
  allocated and compiled at build time, before the KV cache is sized. Nothing is
  compiled, allocated or tuned inside a step.
- **Graph-safe decisions.** Whether a step or round runs the megakernel depends
  only on host state every rank shares (forward mode, row count, batch flags),
  so every rank decides the same and a captured CUDA graph replays the decision
  made at capture.
- **Fail-stop waits.** A wait on another GPU has a deadline below SGLang's
  watchdog; on expiry the rank faults instead of returning tokens from a failed
  step.
- **Numerics are declared.** A path is either bitwise with the stock step it
  replaces or changes the arithmetic on purpose; say which in the code and in
  the PR, and test it accordingly (CI test guide).

### Code

- New source files start with the SPDX header used across the package.
- Comments state what a reader cannot see from the code: layouts shared with a
  kernel, units, where a constant came from. History and measurements go in the
  PR, not in comments.

## 2. Contribution guide

### Across SGLang and this repository

- An interface the package needs lands in SGLang through normal SGLang CI. The
  Mega PR's GPU tests run against SGLang `main`, so its integration run is valid
  only once the SGLang side is on `main`. Companion PRs link each other; the
  SGLang PR merges first, or both merge together.
- Keep `main` here working against SGLang `main` at every step. To replace an
  interface, add the new one in SGLang, migrate this package, then remove the
  old one.
- The CPU unit tests run against the SGLang commit pinned as `SGLANG_COMMIT` in
  `.github/workflows/unit-test.yml`. Reads that the pinned SGLang lacks must be
  optional or skipped by name; moving the pin is its own PR.

### Branch, checks and PR

- Work on a feature branch; `pre-commit` refuses commits on `main`. Run
  `pre-commit run --all-files` and the unit tests before pushing.
- PR title: `[Type] scope: Title` or `[Type] [HARDWARE] scope: Title`, with Type
  one of Feat, Fix, Refactor, Perf, Doc, Test, CI, Chore and an uppercase
  hardware tag. The `PR Title` workflow checks it. PRs are squash-merged, so the
  title becomes the commit on `main`.
- Fill in every section of the
  [PR template](.github/pull_request_template.md). For each affected model and
  hardware setup give the commands, the SGLang and Mega SHAs, the hardware and
  topology, every measurement-only setting (for example forced speculative
  acceptance) next to the number it shapes, and links to the runs.
- A new model adds, in the same PR: its entries in `SPECS`, its unit tests, its
  E2E tests and `test/e2e/ci_models.yaml` entry, its row in the README's
  supported-models table, and one line in `.github/CODEOWNERS` naming its owner.
- Update the README wherever coverage, requirements or launch commands change.

### Reporting

- Report only tests that ran, on the hardware they name. Mark skipped and unrun
  cases as such; CPU unit-test success is not GPU qualification.
- Never weaken an assertion, loosen a tolerance or change a baseline to make a
  test pass. Find the cause.

## 3. CI test guide

A change is covered by unit tests and E2E tests only. Do not add tests under
`test/kernels/`: those are legacy standalone kernel tests outside CI. A kernel
is tested through the unit tests of its host logic and its model's E2E tests.

### Unit tests: `test/unit/`, this repository's CI

```bash
python -m pytest test/unit -v -rfE
```

- They run on a CPU-only GitHub runner against the pinned SGLang, without model
  weights. A case that needs CUDA skips cleanly there; mock CUDA where the logic
  under test does not need it. To run them locally, install as the workflow
  does: the pinned SGLang source, CPU torch, then `pip install --no-deps -e .`.
- Cover the spec's admission and each refusal reason, the plan and its widths,
  every hand-back reason, and the SGLang contract. For a runtime that orders
  launches, test the order.

### E2E tests: `test/e2e/`, SGLang's GPU CI

- A maintainer adds the `run-ci-sglang` label to the PR. This repository's
  `run-ci-sglang.yml` dispatches SGLang's `pr-test-megakernel.yml` with the PR's
  head SHA and reports the result as the `sglang-ci/megakernel` status. SGLang
  installs the PR on its current `main` and runs only `test/e2e/`. Record the run
  link and both SHAs in the PR.
- `test/e2e/ci_models.yaml` selects what runs. Each key is a model: `gpu` and
  `gpus` are the GPU type and the count on one host its tests need, and `tests`
  lists its files. The runner has four GB300 GPUs; a model that needs another
  GPU or more GPUs is skipped with a notice, and a `test_*.py` file no model
  lists never runs.
- Each supported model and hardware setup has:
  - an accuracy test against native SGLang with the megakernel off, on the same
    SGLang SHA and evaluation set, with a comparison verdict rather than a fixed
    absolute score;
  - a performance test against the same baseline, with repeated runs and their
    variation;
  - a stock-parity test where the model has stock step code
    (`test_deepseek_v41_switch.py`).

  CONTRIBUTING.md defines the comparison protocol.
- Follow the DeepSeek-V4.1-Flash layout: a helper module
  (`deepseek_v41_server.py`) holds the model path, launch arguments and
  environment; the `test_*.py` files launch servers with
  `sglang.test.test_utils.popen_launch_server`, with `--enable-nvidia-megakernel`
  for the candidate and without it for the baseline.
  `SGLANG_NVIDIA_MEGAKERNEL_E2E_URL` points the same checks at a server already
  running.
- Accuracy runs use real speculative acceptance; forced acceptance is for
  performance runs only and is stated with their numbers.
- A setup the CI runner cannot hold still has its tests in `test/e2e/`. Run them
  by hand on the named hardware and attach the results to the PR.
