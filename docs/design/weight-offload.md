# Host-resident weights

Flyx has two independent limits on CUDA weight residency. `--gpu-experts N` retains up to N experts per routed layer. `--gpu-layers N` retains the first N target attention or GDN layers and their MLPs. Both accept zero. With elastic loading, the engine asks the backend for byte estimates and chooses the largest resident partition that leaves room for runtime and the device reserve. It checks spare host memory for that partition. An explicit cap takes precedence. The public backend ABI carries these fields in version 53.

## Routed experts

The GGUF tensor axes are expert, output row and input column after model binding. The split follows expert ID in each physical tensor. Repeated visits through a loop reuse the same stored matrix. GPU memory holds only the resident prefix. A model-owned pageable buffer holds the suffix in the original block quantization or the BF16 representation selected by the loader. A per-expert scale follows the same split. Expert IDs can later map through a frequency-ranked permutation without changing the router or the row kernels.

The GPU router computes all scores and top-k choices. For a bounded chunk of up to eight input rows the compute stream copies IDs and normalized input into pinned staging and records an event. The submitting thread waits for that event, queues hot work, and joins the persistent CPU pool to compute cold gate/up and down rows. The GPU keeps router weights. The CPU returns the selected slots in FP32. The existing GPU reduction consumes all slots in router order and applies the residual once. Only activations, IDs and results cross PCIe during inference.

Each worker owns a disjoint range of output rows. Its reduction order is independent of token count and worker scheduling. A runtime AVX2 and FMA check selects vector dot products for the expert types in the two Inkling checkpoints. Other supported block formats use the scalar reference decoder. CPU activation preparation follows the matrix's Q8, A16H, BF16 or FP32 input contract. GPU and CPU accumulation trees differ, so comparisons across placement use logit error and token agreement. Prefill uses the same arithmetic in chunks. Long prompts revisit more weights, making grouped CPU matrix multiplication or staging active experts to GPU worth measuring.

## Dense layer streaming

The loader preserves the CUDA projection layout, including packed codes, scales, column dimensions and row permutations. Streamed projections are assembled at load, archived in page-locked RAM and released from VRAM before the next block. Each attention or GDN mixer and its MLP form separate transfer units. Two VRAM slots are sized for the largest unit assigned to each slot. Norms, small vectors, embeddings, output head and recurrent state follow the regular backend placement policy. The copy stream waits on a slot's last compute event, copies the next weight image and records readiness. The compute stream waits for readiness and returns the slot after its last use. This permits copy of the next block while the current block computes. Every prefill row or verification row in a pass reuses that transfer.

The implementation uses eager execution on active offload paths. Model-owned staging has a submission lock and stream dependencies across sessions. Full residency allocates neither workspace and uses the existing capture and batching paths. Pinned dense weights consume physical host memory throughout inference. A smaller staged copy from pageable storage would reduce this pressure at the cost of a second memory copy.

## Architectural references

Basalt's Inkling-2 documentation describes profile-guided placement, dynamic changes as KV grows, and staging experts on GPU for prefill. Its distributed v0.5.7 profile ranks 24 physical layers of 128 experts. The local runtime log reports CPU worker count and routing hits. These data describe that run and its pack. [Basalt](https://github.com/jesdga95/basalt)

Strata's Inkling hybrid dispatcher uses host-visible doorbells, a persistent worker pool and buffered DMA for prefill. Its implementation also accounts for mapped and staged host access on Windows. Flyx initially uses CUDA events so completion and buffer ownership follow CUDA stream ordering. A stream-wide synchronization would also wait for later work enqueued before the call. A mapped doorbell would require a system-visible publication protocol and can consume a polling core. Profiling on WDDM should determine whether it earns that cost. [Strata hybrid dispatcher](https://github.com/Niko1221/Strata/blob/c234774038d1c9249cb33637d27b895d47b39786/engine/src/inkling2/cuda/ink_hybrid.cuh) [CUDA host programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html)

## Measurements

`cuda_bench h2d` measures the pinned H2D ceiling with 4, 32 and 256 MiB transfers. For plain decode, multiply the logged streamed bytes per full pass by steady-state passes per second to obtain effective streaming bandwidth. For multirow prefill use the actual number of passes, including the final short pass. Compare on the same prompt, context, GPU, driver, clocks and build. Verbose hybrid runs report the wall time from enqueuing the route copies to observing their event per layer visit, including GPU work ahead of the copies.
