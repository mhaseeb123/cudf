# nvCOMP ZSTD Race Investigation Checkpoint

Last updated: 2026-07-15

## Goal

Explain flaky CUDA failures from concurrent `chunked_parquet_reader` use on ZSTD Parquet data with:

- `LIBCUDF_NVCOMP_POLICY=OFF`
- `LIBCUDF_NVCOMP_POLICY=ALWAYS`

Typical failures:

- `cudaErrorIllegalAddress`
- `cudaErrorMisalignedAddress`
- GPU stalls
- Secondary RMM allocation or stream-synchronization errors after CUDA enters error state

Primary workload:

- `/home/coder/data/tpch_sf10`
- 6,002 Parquet files
- 16 concurrent reader threads
- Usually 30 iterations per thread

## Current state

Root cause not identified.

Strongest correlation: old reproducer used libcudf's 32-stream internal global pool for long-lived reader parent streams. `fork_streams()` also selects temporary children from that pool. Dedicated owning parent streams repeatedly suppress failure; old pooled-parent mode reproduces it.

This is topology/timing evidence, not proof of root cause:

- Shared child streams are legal. Child work serializes.
- Captured `A → B` and `B → A` stream-level relationships do not alone prove temporal CUDA cycle. Parent event record occurs before child wait submission.
- CUDA event reuse is valid. Later record does not alter already-enqueued wait.
- No current buffer-specific missing ordering/lifetime edge has been proven.

## Reproducer state

`cpp/examples/nvcomp_zstd_repro/reproducer.cpp`:

- Default: one owning `rmm::cuda_stream` per reader thread.
- `--global-pool-streams`: restores old pooled-parent negative control.
- One parent stream persists through reader's iterations.

Observed:

- Dedicated parents passed `OFF` and `ALWAYS` stress runs.
- Dedicated parents passed installed and RC 5.3.0.16 nvCOMP runs.
- Old pooled-parent mode reproduced `cudaErrorIllegalAddress`.
- 512-stream pool, separate fork pools, per-thread fork pools, default stream, and broad device synchronization also suppressed failure. These alter scheduling; none proves root cause.

## Audited paths

Four source audits found no present direct free/use violation in:

- OFF host-ZSTD decompression route.
- Parquet decompression output → page decode route.
- Decode-local descriptors/output buffer lifetimes.
- Stable shared `cuda_async_memory_resource` use.

Current host decompression has required worker-to-host join:

```cpp
cudf::detail::join_streams(streams, stream);
```

This is valid correctness fix. It does not alone eliminate pooled-parent failures.

Historical Compute Sanitizer UAF in `decode_page_data` predates this ordering change. It names a symptom, not current root cause.

## Latest sanitizer results

Both runs used pooled-parent `OFF` control, 16 threads, 2 iterations:

| Check | Reads | Result | Interpretation |
|---|---:|---|---|
| `synccheck` | 12,004 | `ERROR SUMMARY: 0 errors`, 278 s | No checked device-side barrier misuse |
| `memcheck --track-stream-ordered-races=all` | 12,004 | `ERROR SUMMARY: 0 errors`, 2,459 s | No detected async allocation/free ordering race |

Neither run reproduced application failure. Sanitizer overhead changes timing drastically. Neither result identifies root cause or clears untraced cross-stream lifetime paths.

## Current best lead

Capture address-correlated allocation, free, and consumer records under old pooled-parent mode.

Need correlate:

1. Device address, size, allocation stream, free stream.
2. Buffer owner and reader/thread tag.
3. Kernel/copy consumer and stream.
4. `fork_streams()` / `join_streams()` event and wait relationships.
5. First asynchronous CUDA error.

Target candidate buffers:

- `subpass.decomp_page_data`
- `pass.decomp_dict_data`
- `pass.raw_page_data`
- decompression descriptor/result arrays
- decode-local descriptor arrays
- returned output-table buffers

Use in-memory fixed-size ring buffer. Avoid synchronous logging, events, or extra stream synchronization in hot path.

## Secondary follow-up

Trace actual Velox integration stream ownership:

- Does it use `cudf::detail::global_cuda_stream_pool()` for long-lived reader parent streams?
- If yes, reproducer topology matches production.
- If no, dedicated parent streams may only fix reproducer scheduling.

## Current working tree

Intentional investigation changes:

- `cpp/examples/nvcomp_zstd_repro/reproducer.cpp`
- `nvcomp-zstd-race-hypotheses.md` — canonical experiment ledger
- `nvcomp-zstd-race-checkpoint.md` — current handoff
- `nvcomp-zstd-race-investigation-slack.md` — Slack summary

Temporary libcudf diagnostics removed. `cpp/src/utilities/stream_pool.cpp` clean relative to git.

Other working-tree entries previously observed and not owned by this investigation:

- `.agents/skills/build-test-cudf/SKILL.md`
- `.gitignore`
- Untracked `core`

## Next session

1. Read this checkpoint and `nvcomp-zstd-race-hypotheses.md`.
2. Preserve dedicated-stream default and pooled-parent negative control.
3. Do not claim pooled stream relationships prove cycle.
4. Add lowest-perturbation address/stream tracing.
5. Run failing pooled-parent workload until first CUDA error.
6. Identify actual free/consumer ordering violation before proposing libcudf production fix.
