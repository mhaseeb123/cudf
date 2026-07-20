# nvCOMP ZSTD Race Experiment Ledger

Last updated: 2026-07-17

Use this file before starting new experiments. It records attempted hypotheses, changes, observations, and current verdicts.

Verdicts:

- **Confirmed** — direct evidence supports claim.
- **Supporting** — result fits claim but may only change timing.
- **Rejected** — experiment failed to fix or audit disproved claim.
- **Inconclusive** — timing, incomplete run, or confounding factor prevents conclusion.
- **Open** — still worth isolated follow-up.

## Baseline and nvCOMP runtime

| Hypothesis | Change or test | Result / observation | Verdict |
|---|---|---|---|
| Failure requires nvCOMP | Installed nvCOMP, `LIBCUDF_NVCOMP_POLICY=ALWAYS` | Illegal-address failures and GPU stalls | Rejected as sole cause |
| Failure exists without nvCOMP | `LIBCUDF_NVCOMP_POLICY=OFF` | Repeated `cudaErrorIllegalAddress` failures | Confirmed |
| nvCOMP RC 5.3.0.16 fixes failure | Built libcudf against RC and ran `ALWAYS` | Failure remained | Rejected |
| RC changes clean `OFF` behavior | Preloaded RC while running `OFF` | Misaligned-address/RMM failures | Confounded by preload |
| RC preload interposes bundled RMM symbols | Inspected exports and changed preload order | Conda `librmm.so` before RC nvCOMP removed misleading RMM interposition | Confirmed setup issue |
| RC preload caused all `OFF` failures | Ran `OFF` without preload | Failure still reproduced | Rejected |
| Existing CMake cache respected new `nvcomp_DIR` | Reconfigured existing build | Initial build still used old package | Rejected; explicit reconfigure required |
| RC static package is usable directly | Configured static dependency | RC CMake referenced missing old-path archive | Rejected for this environment |
| Dynamic-only RC view selects RC library | Copied dynamic libraries/CMake metadata into separate prefix | Build configured, but runtime RPATH still favored conda library | Partially successful |
| `LD_PRELOAD` selects RC reliably | Preloaded RC library | RC selected, subject to RMM ordering caveat | Confirmed |

## Stream topology and race condition

| Hypothesis | Change or test | Result / observation | Verdict |
|---|---|---|---|
| Dedicated caller parent streams suppress race | One owning `rmm::cuda_stream` per reader | Repeated full runs passed | Strong mitigation; root cause unproven |
| Dedicated streams work under both policies | Stress `OFF` and `ALWAYS` | Both passed | Confirmed in tested runs |
| Dedicated streams work with nvCOMP RC | Installed and RC runs | Both passed | Confirmed in tested runs |
| Old pooled-parent behavior remains a negative control | Added `--global-pool-streams` | Reproduced `cudaErrorIllegalAddress` | Confirmed |
| 32-stream pool collisions matter | Increased pool size from 32 to 512 | Full run passed | Supporting; lowers collision rate |
| Separate pool for fork children avoids race | Temporary separate `fork_streams()` pool | `OFF`, installed nvCOMP, and RC runs passed | Supporting |
| Separate fork pool is production-ready | Code/test review | Doubled resources; nested-fork isolation incomplete; regression test had issues | Rejected implementation |
| Per-thread fork pools avoid cross-thread collision | Temporary thread-local fork pools | Four full runs passed: 720,240 reads | Strong supporting evidence |
| Excluding only current parent stream is enough | Filtered current parent from fork results | Failed | Rejected; broader cross-reader reuse matters |
| Per-thread default-stream semantics fix race | Enabled per-thread default stream | Failed | Rejected |
| Single default stream suppresses race | `LIBCUDF_USE_DEBUG_STREAM_POOL=1` | Passed | Supporting through serialization |
| Parent-stream wait alone fixes ordering | Added targeted parent-stream waits | Failed | Rejected |
| Direct child-stream waits fix ordering | Synchronized child streams directly | Failed | Rejected |
| Device-wide wait fixes race | Added `cudaDeviceSynchronize()` at selected boundary | Passed | Supporting only; serializes all concurrent work |
| Removing outer host-only decompression fork helps | Temporary host-only fast path | Two full runs passed | Supporting; removes collision opportunity |
| Stream reuse itself is illegal | Static CUDA ordering audit | Multiple host threads may legally enqueue ordered work on same stream | Rejected as stated |
| Reusing one CUDA event breaks prior waits | CUDA API documentation audit | Later records do not affect waits already submitted | Rejected |
| Pooled parents and fork children create opposite stream-level dependencies | Tracked active fork parent/child pairs | Captured simultaneous depth-0 `A → B` and `B → A` on separate reader threads | Confirmed topology |
| Opposite stream-level dependencies prove a temporal cycle | Analyzed CUDA enqueue ordering | Parent event record precedes child wait; opposite edges alone can remain acyclic | Rejected |
| Separate persistent events expose a GPU stall | Persistent event per dependency | Run stalled after iteration 1 at high GPU utilization | Inconclusive; instrumentation changed timing |
| Captured inversion always fails immediately | One traced old-mode iteration | Inversion captured; 6,002 reads still passed | Rejected; race manifestation remains timing-sensitive |

## Fork/join synchronization checkpoints

| Hypothesis | Change or test | Result / observation | Verdict |
|---|---|---|---|
| Missing host-worker join is complete root cause | Added `join_streams(streams, stream)` after worker futures | Early runs passed; later clean `OFF` run failed | Rejected as complete fix |
| Host-worker join is still required | Audited result publication ordering | Parent should depend on worker streams before publishing results | Confirmed correctness change |
| Device sync after Parquet decompression isolates earlier work | `LIBCUDF_SYNC_AFTER_PARQUET_DECOMPRESSION` diagnostic | Failure remained | Did not isolate/fix |
| Device sync after `host_decompress()` suppresses race | `LIBCUDF_SYNC_AFTER_HOST_DECOMPRESSION` | Passed | Supporting; broad timing change |
| Device sync after outer decompression join suppresses race | `LIBCUDF_SYNC_AFTER_DECOMPRESSION_JOIN` | Passed while control failed | Supporting; broad timing change |
| Parent stream sync after outer join is enough | `LIBCUDF_STREAM_SYNC_AFTER_DECOMPRESSION_JOIN` | Failed | Rejected |
| Synchronizing each immediate child is enough | `LIBCUDF_CHILD_SYNC_AFTER_DECOMPRESSION_JOIN` | Failed | Rejected |
| Thread-local fork pool changes result | `LIBCUDF_USE_THREAD_LOCAL_FORK_POOL` diagnostic | Multiple full passes | Supporting; diagnostic removed |

## Host ZSTD decompression and pinned memory

| Hypothesis | Change or test | Result / observation | Verdict |
|---|---|---|---|
| CPU reads compressed data before D2H completes | Audited `h_in` and following `h_out` allocation | `h_out` pinned allocation synchronizes worker stream before CPU decompression | Rejected |
| `h_out` is destroyed before H2D completes | Added explicit worker-stream sync | Existing `cudf::detail::cuda_memcpy` already synchronizes | Rejected; redundant sync removed |
| Pinned pool is unsafe across worker threads | Audited RMM pinned pool | Pool is thread-safe and stream-ordered | Rejected |
| Fallback pinned free races pending stream work | Synchronized stream before `pinned_host_memory_resource::deallocate` | Default-pool sync run still failed at iterations 8–10; with a 256-byte pool both sync and no-sync controls passed 180,060 reads because fallback overhead slowed each run to ~1,540 s | Rejected; synchronization showed no benefit |
| H2D copy contents are corrupted | Added D2H verification after host decompression copy | No mismatch before later failure | Rejected |
| Host-only path avoids all problematic concurrency | Temporary host fast path | Passed limited full runs, but only removes scheduling overlap | Supporting, not final fix |
| Host output/result descriptors outlive publication | Audited futures, join, and synchronous result copy | Lifetimes valid after retained host join | Confirmed for current path |

## Parquet descriptors, copies, and page decode

| Hypothesis | Change or test | Result / observation | Verdict |
|---|---|---|---|
| `comp_in`, `comp_out`, `copy_in`, `copy_out` are used before pinned allocation completes | Audited allocator and initialization | Allocator synchronizes before host initialization | Rejected |
| Descriptor arrays die before H2D/consumer work completes | Audited lifetime through copies and validation | Descriptors remain alive | Rejected |
| Uninitialized `copy_in`/`copy_out` tail is submitted | Found vectors sized to `num_comp_pages` but populated to `curr_copy_page` | Confirmed separate bug; fixed by passing `curr_copy_page` in commit `a9a50ead54` | Confirmed separate bug |
| `cudaMemcpyBatchAsync` causes corruption | Disabled batched memcpy | Later failure still occurred around iterations 7–8 | Rejected |
| Parquet file-read/H2D futures are incomplete | Traced `parquet_io_utils.cpp` ordering | Reads and H2D copies are awaited before decompression | Rejected currently |
| One specific bad file triggers failure | Repeated one lineitem file across 16 threads | 1,600 reads passed | Rejected as simple file-specific bug |
| File diversity and allocation churn increase race probability | Compared one-file and full 6,002-file workloads | Full workload fails much more reliably | Supporting |
| `lineitem` subset alone can reproduce | 1,000-file lineitem stress | Failed around iterations 54–57 | Confirmed but very flaky |
| Failure affects all codecs | Full Snappy `OFF` stress | 180,060 reads passed | Rejected; ZSTD/path-specific signal |

## Device allocation and memory reuse

| Hypothesis | Change or test | Result / observation | Verdict |
|---|---|---|---|
| Async device allocation/reuse is necessary | `--synchronous-mr` using `cudaMalloc`/`cudaFree` | Reached iteration 10 without failure; stopped due runtime | Inconclusive |
| Conservative cudaMallocAsync reuse fixes failure | Disabled opportunistic/internal reuse | Did not establish reliable fix | Inconclusive; option removed |
| Shared async MR alone causes race | Compared synchronous and async resources | Synchronous run changed timing heavily | Inconclusive |
| Failure is device use-after-free | Earlier Compute Sanitizer run | Reported UAF in Parquet `decode_page_data` | Real historical evidence, stale for current code |
| Historical sanitizer stack identifies current race | Compared timestamps/code | Run predates host join and current stream fix | Rejected as current proof |

## nvCOMP-specific execution

| Hypothesis | Change or test | Result / observation | Verdict |
|---|---|---|---|
| Concurrent nvCOMP sequence causes `ALWAYS` race | Serialized all nvCOMP calls through completion | Passed 180,060 reads; severe slowdown | Supporting but timing-heavy |
| Temp-size query races | Serialized only `GetTempSizeSync` | Failed around iterations 4–6 | Rejected |
| Decompression execution overlap matters | Serialized decompression through completion | Passed 180,060 reads; severe slowdown | Supporting for `ALWAYS` |
| Launch call itself races | Serialized launch only | Failed or stalled | Rejected/inconclusive |
| Asynchronous event-chain serialization is sufficient | Shared ZSTD completion event | Failed around iterations 26–30 | Rejected |
| nvCOMP ZSTD alignment requirements are violated | Audited documented 8-byte alignment against buffers | Possible violation found | Open, independent `ALWAYS` lead |
| Alignment explains `OFF` failures | Compared policy paths | `OFF` bypasses nvCOMP | Rejected |
| nvCOMP RC alone fixes concurrency | RC stress under old topology | Failed | Rejected |

## Debugger and sanitizer experiments

| Hypothesis | Change or test | Result / observation | Verdict |
|---|---|---|---|
| Compute Sanitizer will pinpoint current access | Full sanitizer attempt | Too slow; timing changed substantially | Inconclusive |
| One sanitizer iteration is enough | One-iteration runs with current variants | Completed without errors | Inconclusive |
| Sync-after-read preserves sanitizer failure | Added stream/device sync after each chunk read | No current error captured; timing changed | Inconclusive; options removed |
| `CUDA_LAUNCH_BLOCKING=1` identifies kernel | Launch-blocking run | Too slow/aborted without useful location | Inconclusive |
| Short memcheck run detects stream-ordered allocation race | `memcheck --track-stream-ordered-races=all`, pooled-parent `OFF`, 2 iterations | Failure did not reproduce: 12,004 reads completed; `ERROR SUMMARY: 0 errors` | Inconclusive; extreme timing perturbation |
| Device-side synchronization misuse causes failure | `synccheck`, pooled-parent `OFF`, 2 iterations | 12,004 reads, 278 s, `ERROR SUMMARY: 0 errors` | Rejected for checked barrier primitives |
| Error location reported by RMM is root location | Examined stacks | RMM stream sync often reports prior asynchronous CUDA failure | Rejected |

## Removed diagnostic controls

These were temporary and should not be reintroduced without a new reason:

- `LIBCUDF_NVCOMP_SERIALIZE`
  - Modes: `ALL`, `TEMP`, `DECOMP`, `LAUNCH`
- `LIBCUDF_HOST_FAST_PATH`
- `LIBCUDF_VERIFY_HOST_DECOMPRESSION_COPY`
- `LIBCUDF_DISABLE_CUDA_MEMCPY_BATCH`
- `LIBCUDF_SYNC_AFTER_PARQUET_DECOMPRESSION`
- `LIBCUDF_SYNC_AFTER_HOST_DECOMPRESSION`
- `LIBCUDF_SYNC_AFTER_DECOMPRESSION_JOIN`
- `LIBCUDF_STREAM_SYNC_AFTER_DECOMPRESSION_JOIN`
- `LIBCUDF_CHILD_SYNC_AFTER_DECOMPRESSION_JOIN`
- `LIBCUDF_USE_THREAD_LOCAL_FORK_POOL`
- `LIBCUDF_USE_SEPARATE_STREAM_EVENTS`
- `LIBCUDF_TRACE_STREAM_CYCLES`
- `--synchronous-mr`
- `--conservative-async-reuse`
- `--sync-after-read`
- `--device-sync-after-read`

Retained negative-control option:

- `--global-pool-streams`

## Current best leads

### Lead 1 — caller stream ownership

Confirmed in reproducer: long-lived reader parents from libcudf internal pool permit opposite depth-0 stream relationships. Dedicated owning streams suppress tested failure. Neither fact identifies the missing temporal ordering or lifetime edge.

Next question: verify actual Velox integration. Does it use libcudf's internal global pool for reader parent streams, or does reproducer only approximate that behavior?

### Lead 2 — production-level guard or documentation

Choose between:

- Caller contract: parent streams must not come from libcudf internal fork pool.
- API redesign: owning/leased stream abstraction.
- Internal active-parent tracking, only if a real temporal cycle is later demonstrated.

Do not revive simple separate-pool patch without handling nested forks, device count, resource cost, and tests.

### Lead 3 — stale sanitizer UAF

Existing UAF report predates current fixes. Use it only as a lead for targeted allocation/consumer tracing; it does not prove current root cause.

## Do not repeat without new evidence

- Full Compute Sanitizer stress run.
- Launch-blocking full workload.
- Temp-size-only nvCOMP serialization.
- Launch-only nvCOMP serialization.
- `cudaMemcpyBatchAsync` disable test.
- Pinned-vector allocation/lifetime audit.
- `h_out` post-copy synchronization.
- Per-thread default-stream test.
- Current-parent-only child exclusion.
- Immediate parent/child stream synchronization.
- RC `OFF` run with RC nvCOMP preloaded ahead of conda RMM.
