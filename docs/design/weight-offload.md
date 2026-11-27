# Host-resident weights

Flyx has two independent limits on CUDA weight residency. `--gpu-experts N` retains up to N experts per routed layer. `--gpu-layers N` retains the first N target attention or GDN layers and their MLPs. Both accept zero. With elastic loading, the engine asks the backend for byte estimates and chooses the largest resident partition that leaves room for runtime and the device reserve. It checks spare host memory for that partition. An explicit cap takes precedence. The public backend ABI carries these fields in version 53.

## Routed experts

The Inkling-2 binder emits routed MoE blocks. A repeated visit through a loop shares the same physical gate/up and down matrices. Each pair initially keeps logical IDs below the cap on the GPU. The suffix lives in pageable host memory with its per-expert scales. The loader retains the CPU row representation and, when streaming is enabled, a second copy of the cold suffix in the CUDA projection layout. The elastic planner charges both copies to host memory. Its device estimate includes the two staging banks, workspace and route maps. It uses the device reserve already calculated for the platform, including the smaller native Windows reserve introduced in PR #71. If the allocator refuses a partition before session sizing can measure a shortfall, the engine first releases one unit and doubles the reduction on repeated refusals. A measured shortfall releases enough weight bytes to cover it. A partial upload can leave the read-only checkpoint in the process working set. Before refusing a host allocation, the loader stops read-ahead, releases the clean pages of its own source mappings and measures available physical RAM again. The file stays mapped for the retry. [Windows working-set behavior](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualunlock)

**Prompt passes.** The compute stream groups routed rows by expert. At or above `expert_prefill_min`, the loader's packed cold banks feed the same grouped GPU projections as the resident matrices. A copy stream uploads the next bank through one of two reusable pinned bounce buffers. An upload event protects the host buffer from premature refill, and a consumption event protects the GPU slot from overwrite. All gate/up rows finish before the down pass reuses shared scratch. The reduction visits slots in their original router order. Small prompt tails and verification rows use the CPU path.

**Decode.** The compute stream copies router IDs and normalized activations into pinned buffers and records an event. A stream query flushes the submission on WDDM. The host queues resident projections before waiting for the routing event. The submitting thread and the persistent pool share cold gate/up and down work across disjoint output rows. The workers briefly spin for another job and park when idle. Only cold result ranges return over PCIe. The GPU combines them with its resident results and applies the residual once. CPU and GPU accumulation order can differ. The backend therefore measures cross-placement logits against full residency.

**CPU arithmetic.** Each gate/up pair uses the same prepared input row. Supported IQ formats with Q8 inputs quantize each group of 32 activations once and use integer dot products on AVX2 or AVX-512 VNNI. A16H, BF16 and FP32 inputs retain the activation precision of their projection. AVX-512 also accelerates the Q2_0, NVFP4, MXFP4 and IQ4_NL down projections. Runtime dispatch checks processor and OS vector-state support. The scalar path remains available. The `expert_threads`, `expert_simd` and `expert_integer` options support controlled CPU comparisons.

**Placement.** A GPU histogram counts logical IDs during prefill. When a new session opens after every previous session has closed, the backend halves each physical pair's prior score, adds the observed counts and exchanges up to one eighth of its experts. It favors the current resident on equal scores and uses logical ID to break other ties. A swap moves weights and scales together and reconstructs the evicted expert's CPU representation from the packed data. Placement stays fixed across verification, checkpoint restore and overlapping sessions. Load-time plan calibration can train the first user session. Verbose output gives the hit rate over routed verification slots and the aggregate for each completed group of concurrent sessions. Setting `expert_prefill_min=0` omits the packed archive used by exchanges and runs CPU experts for every prompt size. `expert_adapt=0` fixes the initial partition, and `expert_global=1` shares scores across physical layers with the same expert count.

The quant in a GGUF filename describes the bundle. The two local Inkling files use these tensor types for the 24 physical gate/up and down matrices. Each matrix contains 128 experts.

| Checkpoint | Gate/up matrices | Down matrices |
| --- | --- | --- |
| Inkling-2-Mini-UD-Q2_K_XL | 11 IQ2_S, 6 IQ1_S, 5 IQ2_XXS, 2 IQ2_XS | 12 Q2_0, 11 NVFP4, 1 MXFP4 |
| Inkling-2-Mini-UD-IQ3_XXS | 13 IQ3_XXS, 9 IQ2_S, 1 IQ2_XXS, 1 IQ2_XS | 20 NVFP4, 4 IQ4_NL |

The dense Qwen3.8-27B-UD-IQ3_XXS GGUF uses the Qwen3.5 architecture. Its 64 target layers mix gated delta net and attention blocks. The planner counts the target mixers separately from the MTP companion. The transfer unit uses the physical CUDA bytes of each projection after binding. For example, a stack of different packed types retains one typed run per group of adjacent matrices with the same type.

## Dense layer streaming

The loader preserves the CUDA projection layout, including packed codes, scales, column dimensions and row permutations. Streamed projections are assembled at load, archived in page-locked RAM and released from VRAM before the next block. Each attention or GDN mixer and its MLP form separate transfer units. Two VRAM slots are sized for the largest unit assigned to each slot. Norms, small vectors, embeddings, output head and recurrent state follow the regular backend placement policy. The copy stream waits on a slot's last compute event, copies the next weight image and records readiness. The compute stream waits for readiness and returns the slot after its last use. This permits copy of the next block while the current block computes. Every prefill row or verification row in a pass reuses that transfer.

The implementation uses eager execution on active offload paths. Model-owned staging has a submission lock and stream dependencies across sessions. Full residency allocates neither workspace and uses the existing capture and batching paths. Pinned dense weights consume physical host memory throughout inference. A smaller staged copy from pageable storage would reduce this pressure at the cost of a second memory copy.

## Architectural references

Basalt's Inkling-2 documentation describes profile-guided placement, dynamic changes as KV grows, and staging experts on GPU for prefill. Its distributed v0.5.7 profile ranks 24 physical layers of 128 experts. Its pool groups routed rows by expert and selects AVX2 or AVX-512 row kernels by format and CPU capability. The GPU checks a per-visit completion ticket. The local runtime log reported seven workers and 1,354 of 3,072 experts in RAM for that pack. [Basalt Inkling-2 placement](https://github.com/jesdga95/basalt/blob/v0.5.7/docs/models/inkling2.md) [CPU row kernels](https://github.com/jesdga95/basalt/blob/v0.5.7/src/kernels/cpu/native_expert.cpp)

Strata's Inkling hybrid dispatcher uses host-visible doorbells, CPU SIMD kernels and a persistent worker pool. It tracks decaying expert hit counts for promotion and can switch prefill to double-buffered DMA feeding GPU GEMMs. On Windows it can store weights in pageable RAM and copy through a small pinned buffer when pinning the arena would exceed its budget. Flyx initially uses CUDA events so completion and buffer ownership follow CUDA stream ordering. A stream-wide synchronization would also wait for later work enqueued before the call. A mapped doorbell would require a system-visible publication protocol and can consume a polling core. Profiling on WDDM should determine whether it earns that cost. [Strata hybrid dispatcher](https://github.com/Niko1221/Strata/blob/c234774038d1c9249cb33637d27b895d47b39786/engine/src/inkling2/cuda/ink_hybrid.cuh) [CUDA host programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html)

## Measurements

`cuda_bench h2d` measures the pinned H2D ceiling with 4, 32 and 256 MiB transfers. Verbose streaming runs report DMA bandwidth during sampled copies, separately for prefill and rounds. Device timestamps exclude the wait for a free slot. For plain decode, multiply the logged streamed bytes per full pass by steady-state passes per second to obtain effective streaming bandwidth. For multirow prefill use the actual number of passes, including the final short pass. Compare on the same prompt, context, GPU, driver, clocks and build. Verbose hybrid runs report the wall time from enqueuing the route copies to observing their event per layer visit, including GPU work ahead of the copies.

### Inkling-2 on RTX 5070 Ti

Measured on native Windows 11 build 26300, GeForce RTX 5070 Ti 16,303 MiB, driver 610.88, Ryzen 7 9800X3D (8 cores, 16 logical processors) and 32 GB DDR5-6000. Both Release builds targeted `120-real`. The baseline was PR #70 rebased onto `ab4a4b2b`, with elapsed-time instrumentation. The intermediate optimized build `e7c724e` used a 16-row streaming threshold. The final SKU crossover measurement appears below. The [PR benchmark table](https://github.com/enwove/flyx/pull/70) uses the final default and includes the elastic 8 GiB runs. The six questions in `bench/prompts/moe_offload.jsonl` include p512 and the town-record prompt used in the Strata and Basalt comparisons. They tokenize to 511 and 2,897 tokens with this template. The competitor clients counted 2,899.

| Q2_K_XL residency | v1 prefill 511 / 2,897 | v2 prefill 511 / 2,897, chunk 128 | v2 prefill 511 / 2,897, chunk 1,024 | v1 / v2 decode | v2 cache hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/128 | 1,694.74 / 1,402.55 | 1,699.09 / 1,399.51 | 2,398.33 / 2,235.46 | 127.45 / 127.39 | 100% resident |
| 96/128 | 38.94 / 41.44 | 693.50 / 667.66 | 1,794.71 / 1,991.58 | 20.40 / 38.20 | 90.47% |
| 64/128 | 21.99 / 20.65 | 346.50 / 342.59 | 1,183.52 / 1,624.28 | 13.25 / 27.88 | 74.85% |
| 32/128 | 11.82 / 14.77 | 232.61 / 233.05 | 878.45 / 1,375.73 | 9.05 / 21.94 | 52.44% |

All throughput values are tokens/s. Decode pools elapsed time and tokens after the first generated token over prompts two through six. Hit rate includes load-time calibration and all six prompts. With 64 resident experts, hits were 73.89% on the first user prompt, 84.85% on p512 and 74.51% on the town-record prompt. Pooling scores across physical layers yielded 57.99% overall, versus 74.85% for per-layer scoring. With 96 experts the corresponding rates were 80.29% and 90.47%.

| Q2_K_XL decode stage, chunk 128 | 64/128 tok/s | 96/128 tok/s |
| --- | ---: | ---: |
| GPU-streamed prompt, original CPU decode | 12.40 | 19.17 |
| Early submission and persistent pool, float AVX2, static placement | 12.16 | 19.55 |
| Integer SIMD and AVX-512 down, static placement | 20.49 | 28.44 |
| Cross-layer scores | 22.15 | 30.79 |
| Per-layer scores | 27.88 | 38.20 |

| Per-layer decode diagnostic, mean µs/visit | Route to host | CPU work | Resident GPU | Cold result H2D | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 64/128 | 89.64 | 1,435.26 | 103.78 | 55.41 | 31.59 |
| v2 64/128 | 102.42 | 390.07 | 83.89 | 44.46 | 26.91 |
| v1 96/128 | 98.09 | 611.94 | 93.77 | 43.00 | 26.11 |
| v2 96/128 | 103.70 | 196.42 | 85.54 | 29.02 | 19.92 |

The per-layer run uses the first prompt, 12 generated tokens and `--backend-opt profile=1`. Event timing and a synchronization following the reduction change queue depth. In the regular six-prompt run the wall time from copy submission through observing readiness was 176.89 µs at 64/128 and 187.52 µs at 96/128. Cold CPU work read 16.06 and 13.86 GB/s of compressed rows respectively, including activation preparation and pool handoffs in elapsed time. A secondary-stream CUDA host callback took 382.21 µs to hand work to the CPU in a 64/128 probe and reduced five-prompt decode throughput from 26.9 to 24.6 tok/s. The event path remains in use.

Reproduce a throughput run by substituting the checkpoint path and resident count. Repeat with both chunk sizes. For the ablations use `expert_adapt=0`, `expert_adapt=0,expert_integer=0,expert_simd=256` or `expert_global=1` as a single backend option. For the DSpark comparison replace `--no-spec --width 0` with `--drafter dspark --drafter-file DRAFTER --width 3` or width 7.

```sh
flyx bench MODEL --backend cuda --kv-method exact --no-lookup --ctx 4096 --prefill-chunk 1024 -v --prompts bench/prompts/moe_offload.jsonl -n 16 --ignore-eos --json result.json --no-spec --width 0 --gpu-experts N
kld MODEL --text bench/prompts/p512.txt --tokens 64 --start 16 --ctx 1024 --kv-method exact --prefill-chunk 128 --save resident.bin
kld MODEL --text bench/prompts/p512.txt --tokens 64 --start 16 --ctx 1024 --kv-method exact --prefill-chunk 128 --ref resident.bin --backend-options gpu_experts=N
```

A separate crossover test used three 97-token prompts and two generated tokens with static residency. The CPU and staged paths were forced with `expert_prefill_min=0` and `expert_prefill_min=2`. At 96/128, CPU / staged prefill was 82.3 / 43.3 tok/s at eight rows, 91.2 / 86.8 at 16, 89.6 / 168.2 at 32 and 88.9 / 219.9 at 64. At 64/128 the staged path also led at 32 rows (79.8 versus 38.9) and 64 rows (106.3 versus 44.8). The 5070 Ti table therefore selects 32 rows. Other devices use a portable starting point of one routed slot per expert on average. The table setting applies to prompt passes only.

A host-visible CUDA stream-write ticket also ran slower than the event path. At 64/128 its five-prompt decode and handoff were 26.85 tok/s and 211.55 microseconds, versus 28.21 tok/s and 176.44 microseconds with events. At 96/128 the corresponding figures were 36.19 / 221.75 and 38.99 / 185.50. A VNNI gate/up variant sharing each vector load gave 26.83 versus 26.72 tok/s at 64/128 and 38.20 versus 39.31 at 96/128. The retained CPU path shares activation preparation and scheduling and keeps independent row products. At 64/128, four, seven, eight, twelve and sixteen total CPU threads yielded 15.28, 21.03, 21.99, 27.51 and 27.97 tok/s.
