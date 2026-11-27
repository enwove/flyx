# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
# DV's tests (a drafter that drafts alone; not in production), included by tests/CMakeLists.txt with FLYX_TEST_DV alone
# (the *-full configure presets: the same build tree, DV's code is built either way).  Every test here is labelled dv,
# and only the tests here: flyx_finish_test_labels refuses a test named as DV's, or needing DV's assets, anywhere else.
# DV's assets are make_tiny.py's group dv (make_dv_assets, fixture dv_assets), made from the others (assets).
get_property(flyx_before_dv DIRECTORY PROPERTY TESTS)

add_test(NAME make_dv_assets
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/e2e/make_tiny.py ${ASSETS} --group dv)
set_tests_properties(make_dv_assets PROPERTIES FIXTURES_REQUIRED assets FIXTURES_SETUP dv_assets)

# DV's float64 references against the Python reference's cases (make_tiny.py's dv_cases.json): trees on 1000 random
# tables of each kind and edge cases, cov's term, selector tables, the canvas's input, statistics and slide.
add_test(NAME dv_trees COMMAND dv_tree_test ${ASSETS})
add_test(NAME dv_canvas COMMAND canvas_test ${ASSETS})
# The Python reference's DV drafting on the tiny Llama emits its plain decoding's tokens (both trees, a V-LoRA); its
# canvas rows are transformers' Llama with the canvas adapter merged in, and its trees grow again from those canvases,
# also when it commits another run's counts.
flyx_add_py_test(reference_dv ${CMAKE_SOURCE_DIR}/tests/python/reference/dv_test.py ARGS ${ASSETS}
                 FIXTURES assets dv_assets)
# DV's description from the tiny Llama's DV file (models/dv.h): its blocks, canvas and programs, the drafter the
# options choose, sections that cannot be drafted refused.
add_test(NAME dv_models COMMAND dv_models_test ${ASSETS})
# llama_models with tiny_llama_dv_lora.flyx too: a V-LoRA beside DV, and DV's canvas layers with their own adapters.
add_test(NAME llama_models_dv COMMAND llama_models_test ${ASSETS})
set_tests_properties(dv_trees dv_canvas dv_models llama_models_dv PROPERTIES FIXTURES_REQUIRED "assets;dv_assets")
# The tiny DV files' tensors are export_dv.py's, and the reference loads them to their checkpoints' and transformers'
# weights: the beam tree, the pool tree (a causal canvas) and the canvas of 32 slots.
flyx_add_py_test(dv_convert ${CMAKE_SOURCE_DIR}/tests/python/convert/check_dv_convert.py PY_FLAGS -I
                 ARGS ${ASSETS}/tiny_llama_dv.flyx ${ASSETS}/dv_tiny/export --canvas ${ASSETS}/dv_tiny/canvas.pt
                      --selector ${ASSETS}/dv_tiny/selector.pt --hf ${ASSETS}/hf_llama
                 FIXTURES assets dv_assets)
flyx_add_py_test(dv_convert_pool ${CMAKE_SOURCE_DIR}/tests/python/convert/check_dv_convert.py PY_FLAGS -I
                 ARGS ${ASSETS}/tiny_llama_dv_pool.flyx ${ASSETS}/dv_tiny/export_pool
                      --canvas ${ASSETS}/dv_tiny/canvas_causal.pt --selector ${ASSETS}/dv_tiny/selector.pt
                      --hf ${ASSETS}/hf_llama
                 FIXTURES assets dv_assets)
flyx_add_py_test(dv_convert_32 ${CMAKE_SOURCE_DIR}/tests/python/convert/check_dv_convert.py PY_FLAGS -I
                 ARGS ${ASSETS}/tiny_llama_dv32.flyx ${ASSETS}/dv_tiny/export_32 --canvas ${ASSETS}/dv_tiny/canvas.pt
                      --selector ${ASSETS}/dv_tiny/selector.pt --hf ${ASSETS}/hf_llama
                 FIXTURES assets dv_assets)
# The same with a V-LoRA beside the DV (its adapters are bifill-dv's checkpoint's, byte for byte, as the reference
# loads them).
flyx_add_py_test(dv_convert_lora ${CMAKE_SOURCE_DIR}/tests/python/convert/check_dv_convert.py PY_FLAGS -I
                 ARGS ${ASSETS}/tiny_llama_dv_lora.flyx ${ASSETS}/dv_tiny/export_dv_lora
                      --canvas ${ASSETS}/dv_tiny/canvas.pt --selector ${ASSETS}/dv_tiny/selector.pt
                      --vlora ${ASSETS}/dv_tiny/vlora32.pt --hf ${ASSETS}/hf_llama
                 FIXTURES assets dv_assets)
# The DV runner's results file goes on only with the settings and thread count that made it, past a row a stopped run
# cut off (no model is loaded).
flyx_add_py_test(reference_dv_gsm8k ${CMAKE_SOURCE_DIR}/tests/python/reference/dv_gsm8k_test.py)
# The DV reference forced to a bench record's answers (and replaying its rounds): what it is forced to, the rows and
# settings it keeps, its rounds against a free run's where the answers agree (a stand-in for the model); whole runs on
# the tiny Llama's DV: a .flyx file and its checkpoints run the same rounds, the options that choose questions,
# threads, a V-LoRA, a stopped run going on.
flyx_add_py_test(reference_dv_forced ${CMAKE_SOURCE_DIR}/tests/python/reference/dv_forced_test.py ARGS ${ASSETS}
                 FIXTURES assets dv_assets)
# A drafter's answers and tokens per forward counted as bifill-dv counts them, GSM8K scored as it scores them, its
# rounds against the teacher-forced reference's, and what the checks refuse (tests/e2e/check_dv.py, dv_ref_json.py;
# no model).
flyx_add_py_test(dv_counts ${CMAKE_SOURCE_DIR}/tests/e2e/check_dv_test.py)
# DV's rounds chosen, replayed on the kept traces (bench/replay/dv_choice): the sets made from them,
# DV pinned and plain decoding at the traces' speeds, a constant set on the command line taking effect, and what is
# refused.
flyx_add_py_test(dv_choice_replay ${CMAKE_SOURCE_DIR}/tests/bench/replay/dv_choice_test.py
                 ARGS $<TARGET_FILE:dv_choice>)
# The tiny Llama with its DV drafter in the file: as loaded by default (cpu_ref has no DV rounds) and with --drafter
# none; with its beam tree and with a pool tree (whose tree fields are not zeros); beside a V-LoRA (decoded as without
# DV here).
flyx_add_golden_test(golden_cpu_ref_llama_dv ARGS cpu_ref llama ${ASSETS}/tiny_llama_dv.flyx FIXTURES assets dv_assets)
flyx_add_golden_test(golden_cpu_ref_llama_dv_pool ARGS cpu_ref llama ${ASSETS}/tiny_llama_dv_pool.flyx
                     FIXTURES assets dv_assets)
flyx_add_golden_test(golden_cpu_ref_llama_dv_lora ARGS cpu_ref llama_dv_lora FIXTURES assets dv_assets)

if(TARGET flyx-cuda)
  get_property(flyx_cpu_dv_tests DIRECTORY PROPERTY TESTS)
  # DV's rounds in three requests at once: every round its warm-up (rounds of several sessions all warm-ups), a cycle
  # of warm-ups, trees and plain rounds, and its rounds chosen: plain decoding's text, greedy, sampled, penalized.
  add_test(NAME scheduler_cuda_dv COMMAND scheduler_test ${ASSETS}/tiny_llama_dv31.flyx cuda dv)
  set_tests_properties(scheduler_cuda_dv PROPERTIES FIXTURES_REQUIRED "assets;dv_assets")
  flyx_add_golden_test(golden_cuda_llama_dv ARGS cuda llama ${ASSETS}/tiny_llama_dv.flyx FIXTURES assets dv_assets)
  flyx_add_golden_test(golden_cuda_llama_dv_pool ARGS cuda llama ${ASSETS}/tiny_llama_dv_pool.flyx
                       FIXTURES assets dv_assets)
  flyx_add_golden_test(golden_cuda_llama_dv32 ARGS cuda llama ${ASSETS}/tiny_llama_dv32.flyx FIXTURES assets dv_assets)
  # DV beside a V-LoRA: the verify rows with the target's adapters, the canvas with the base weights and its own.
  flyx_add_golden_test(golden_cuda_llama_dv_lora ARGS cuda llama_dv_lora FIXTURES assets dv_assets)
  # DV's rounds (gate 4, its drafts against the reference) with the generic regions and with the verify rows'
  # attention on tensor cores
  foreach(dv dv dv_pool dv_lora)
    set(dv_file ${ASSETS}/tiny_llama_${dv}.flyx)
    set(dv_golden llama)
    if(dv STREQUAL "dv_lora")
      set(dv_golden llama_dv_lora)
    endif()
    flyx_add_golden_test(golden_cuda_llama_${dv}_unfused ARGS cuda ${dv_golden} ${dv_file}
                         ENV FLYX_BACKEND_OPTIONS=fusion=0 FIXTURES assets dv_assets)
    flyx_add_golden_test(golden_cuda_llama_${dv}_rows_mma ARGS cuda ${dv_golden} ${dv_file}
                         ENV FLYX_BACKEND_OPTIONS=rows_attention_from=0 FIXTURES assets dv_assets)
  endforeach()
  # DV's rounds of 31 drafts (32 verify rows: the wide rows of a file that holds DV) on 32 slots, MiniCPM5-1B's DV
  # files' trees: a beam of 8 with coverage, a pool of 7 ranks; the beam's also with the verify rows' attention on
  # tensor cores.
  flyx_add_golden_test(golden_cuda_llama_dv31 ARGS cuda llama ${ASSETS}/tiny_llama_dv31.flyx FIXTURES assets dv_assets)
  flyx_add_golden_test(golden_cuda_llama_dv31_pool ARGS cuda llama ${ASSETS}/tiny_llama_dv31_pool.flyx
                       FIXTURES assets dv_assets)
  flyx_add_golden_test(golden_cuda_llama_dv31_rows_mma ARGS cuda llama ${ASSETS}/tiny_llama_dv31.flyx
                       ENV FLYX_BACKEND_OPTIONS=rows_attention_from=0 FIXTURES assets dv_assets)
  # The beam's rounds of 31 drafts beside a V-LoRA of rank 32 (MiniCPM5-1B's g_b2h line): the target's adapters on
  # 32 verify rows (their GEMMs one k_gemm_skinny_mt launch each) next to the canvas's grouped GEMMs and its pass the
  # round's graph leaves out after a no-op round; also with the generic regions (the canvas's GEMMs one by one).
  flyx_add_golden_test(golden_cuda_llama_dv31_lora ARGS cuda llama_dv_lora ${ASSETS}/tiny_llama_dv31_lora.flyx
                       FIXTURES assets dv_assets)
  flyx_add_golden_test(golden_cuda_llama_dv31_lora_unfused ARGS cuda llama_dv_lora ${ASSETS}/tiny_llama_dv31_lora.flyx
                       ENV FLYX_BACKEND_OPTIONS=fusion=0 FIXTURES assets dv_assets)
  # The same two trees on 16 slots from a flatter selector: trees that take the beam's 8 nodes a depth and the
  # pool's 7th rank, held whole (the canvas keeps float32's sums up to 16 rows).
  flyx_add_golden_test(golden_cuda_llama_dv31_wide ARGS cuda llama ${ASSETS}/tiny_llama_dv31_wide.flyx
                       FIXTURES assets dv_assets)
  flyx_add_golden_test(golden_cuda_llama_dv31_wide_pool ARGS cuda llama ${ASSETS}/tiny_llama_dv31_wide_pool.flyx
                       FIXTURES assets dv_assets)
  # The device memory checks for DV's rounds (its trees of 15 and 31, its warm-ups, batched too) and beside a V-LoRA
  # (the adapters' GEMMs in every round's graph), the tiny Llama's DV files (rooms: its graphs are smaller than the
  # prices are tight for); and DV's round graphs (dvgraphs) with the canvas pass the body of a conditional node and
  # inline (dv_conditional=0): within what they are counted at and under twice what they took, the conditional nodes
  # and their bodies on their own too.
  flyx_add_golden_test(cuda_device_memory_dv ARGS cuda llama ${ASSETS}/tiny_llama_dv31.flyx
                       --device-memory=rooms SERIAL FIXTURES assets dv_assets)
  flyx_add_golden_test(cuda_device_memory_dv_lora ARGS cuda llama_dv_lora ${ASSETS}/tiny_llama_dv31_lora.flyx
                       --device-memory=rooms SERIAL FIXTURES assets dv_assets)
  flyx_add_golden_test(cuda_device_memory_dv_graphs ARGS cuda llama ${ASSETS}/tiny_llama_dv31.flyx
                       --device-memory=dvgraphs SERIAL FIXTURES assets dv_assets)
  flyx_add_golden_test(cuda_device_memory_dv_lora_graphs ARGS cuda llama_dv_lora ${ASSETS}/tiny_llama_dv31_lora.flyx
                       --device-memory=dvgraphs SERIAL FIXTURES assets dv_assets)
  get_property(flyx_gpu_dv_tests DIRECTORY PROPERTY TESTS)
  list(REMOVE_ITEM flyx_gpu_dv_tests ${flyx_cpu_dv_tests})
  set_property(TEST ${flyx_gpu_dv_tests} APPEND PROPERTY LABELS gpu)

  # Known red in a tree on Linux: its fixtures are made by Linux's torch, whose random draws differ from Windows' in
  # their last bits (a bf16 weight or two of the tiny Llama's DV moves, and with it the float64 reference's trees), and
  # these tests' draft agreement falls under its floor while every token stays plain decoding's.  Seen on Linux: an
  # RTX 3090 at v0.2.1 (torch 2.15.0) failed the first five, other GPUs (torch 2.13.0) dv31_wide_pool as well; a tree
  # on Windows (RTX 5070 Ti) passes all six.  Disabled, not left out: every run lists them (Not Run, Disabled) until the
  # fixtures are the same bytes on both systems.  -DFLYX_TEST_DV_KNOWN_RED=ON runs them anyway.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT FLYX_TEST_DV_KNOWN_RED)
    set_tests_properties(golden_cuda_llama_dv_pool golden_cuda_llama_dv_pool_unfused golden_cuda_llama_dv_pool_rows_mma
                         golden_cuda_llama_dv_lora_rows_mma golden_cuda_llama_dv31_wide golden_cuda_llama_dv31_wide_pool
                         PROPERTIES DISABLED TRUE)
  endif()
endif()

get_property(flyx_dv_tests DIRECTORY PROPERTY TESTS)
list(REMOVE_ITEM flyx_dv_tests ${flyx_before_dv})
set_property(TEST ${flyx_dv_tests} APPEND PROPERTY LABELS dv)
