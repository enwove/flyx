# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
# Tests, laid out like the sources they check:
#   core/tokenizer/      tokenizer vs HuggingFace on recorded cases; its BPE vs the rule on vocabularies made there
#   core/modelfile/      the GGUF reader (the container); make_gguf.py writes the GGUF fixtures with the gguf package;
#                        the .flyx reader on files that are wrong (write_flyx.h writes them)
#   ops/dtype/           GGML block weights as float64 vs the gguf package's decoder
#   ops/                 the float64 references' sums side by side: each output the bits of its sum alone; DV's
#                        trees, selector table (ops/draft/) and canvas (ops/canvas/) vs the Python reference
#   models/              a model's description from a GGUF file means the one from its .flyx file; DV's description
#                        (blocks, state, programs, the drafter chosen) from the tiny Llama's DV file
#   models/drafters/     the DFlash2 drafter the architectures share: its config, the dense targets that take one
#                        from a GGUF drafter file, drafter files that are refused
#   core/exec/           telemetry (what the server reports to monitors), the request pipeline, a reply's text,
#                        the scheduler (requests taking turns), the C API
#   core/render/         chat templates vs transformers
#   core/speculation/    which round a session launches next (the drafter's widths, lookup rounds)
#   core/plan/           the memory budget (a context shrinks to what fits); the round plans of a drafter that drafts
#                        alone (DV)
#   models/formats/      chat formats: reading replies back; renderers written for a template vs the Jinja engine;
#                        reply grammars
#   backends/cuda/       every CUDA kernel vs the float64 ops (ops/); a DFlash2 drafter's context ring (its slots,
#                        the positions it still holds for rewinds, restores and snapshots); the rings checkpoints
#                        copy (sliding-window layers', the MTP layer's: their slots, what they hold for rewinds)
#   e2e/                 the whole engine vs the Python reference (golden; make_tiny.py builds the tiny
#                        models and their golden outputs), and a real model vs transformers; a drafter on a
#                        prompt set vs plain decoding and bifill-dv's references (check_dv.py: its counting and
#                        scoring, and its rounds against the teacher-forced reference's, tested in check_dv_test.py)
#   apps/flyx-server/    the stored responses and the HTTP API (run against a live server)
#   apps/flyx-cli/       the command line's file paths and flags, flyx bench on a prompt set (the tiny model)
#   apps/common/         numbers given on the command line, the drafters fetched beside a model
#   bench/ab/            two builds' records of one prompt set compared answer for answer, round for round
#                        (same_runs.py)
#   python/reference/    the Python reference vs transformers; its DV drafting emits plain decoding's tokens, its
#                        canvas is transformers', the DV runners' results files keep their settings (free and
#                        teacher-forced)
#   python/convert/      the converter: configs and tokenizer pipelines it refuses, a post-processor's prefix; DV's
#                        export and what is refused of it; a DV file's tensors are its sources' (check_dv_convert.py)
#
# Labels (docs/testing.md; the test presets in CMakePresets.json select by them).  Every test has one tier, what it
# needs and so about what it costs:
#   unit    no fixture, no model file: seconds        tiny    the tiny models (make_assets): seconds
#   golden  a whole golden_test or gguf_e2e_test run  tools   a bench or tools script's own test (not the engine's)
# and any of these aspects: gpu (a CUDA device: every test of the CUDA block below, and the ones above it that take
# the CUDA backend too), gpu_serial (the device's free memory measured: runs alone), smoke (a golden or GPU test the
# quick preset runs anyway), dv (DV's, not in production: tests/dv.cmake's, registered with FLYX_TEST_DV alone), server
# (starts flyx serve).  The registering functions below label what they know; flyx_finish_test_labels() (each
# directory's end) gives a test without a tier one (tiny with a fixture, else unit), adds quick (unit and tiny tests
# that need neither the GPU nor DV, and the smoke ones), unsets the shell's settings the test presets clear, and
# refuses the configure where a test breaks these rules.
# The labels it cannot infer are at this file's end.
# Labels are only ever appended (set_property(TEST ... APPEND PROPERTY LABELS ...)): one set elsewhere stays.

# The variables the test presets clear: test-base's environment in CMakePresets.json (the engine's and server's
# settings of core/os/env.h, PYTHONOPTIMIZE), read from there so that a run without a preset clears the same ones.
function(flyx_test_cleared_env out)
  set(file ${CMAKE_SOURCE_DIR}/CMakePresets.json)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${file})
  file(READ ${file} presets)
  string(JSON n LENGTH "${presets}" testPresets)
  math(EXPR last "${n} - 1")
  foreach(i RANGE ${last})
    string(JSON name GET "${presets}" testPresets ${i} name)
    if(name STREQUAL "test-base")
      string(JSON n LENGTH "${presets}" testPresets ${i} environment)
      math(EXPR last "${n} - 1")
      set(names "")
      foreach(j RANGE ${last})
        string(JSON var MEMBER "${presets}" testPresets ${i} environment ${j})
        list(APPEND names ${var})
      endforeach()
      set(${out} ${names} PARENT_SCOPE)
      return()
    endif()
  endforeach()
  message(FATAL_ERROR "${file}: no test preset test-base (the environment the tests run without)")
endfunction()

# After a directory's last test: the tier where none was given, quick, the GPU tests' slot and preflight, the shell's
# settings unset, the rules.
function(flyx_finish_test_labels)
  set(tier_labels unit tiny golden tools)
  set(aspect_labels gpu gpu_serial smoke dv server)
  flyx_test_cleared_env(cleared_env)
  get_property(tests DIRECTORY PROPERTY TESTS)
  set(errors "")
  foreach(t IN LISTS tests)
    # A plain ctest as a preset's: the shell's FLYX_BACKEND_DIR, FLYX_BACKEND_OPTIONS, ... unset, but for what the
    # test sets itself (its ENVIRONMENT is applied before ENVIRONMENT_MODIFICATION, which would unset that too).
    get_test_property(${t} ENVIRONMENT own_env)
    set(unset "")
    foreach(var IN LISTS cleared_env)
      if(NOT own_env MATCHES "(^|;)${var}=")
        list(APPEND unset "${var}=unset:")
      endif()
    endforeach()
    set_property(TEST ${t} APPEND PROPERTY ENVIRONMENT_MODIFICATION ${unset})
    get_test_property(${t} LABELS labels)
    if(NOT labels)
      set(labels "")
    endif()
    list(REMOVE_DUPLICATES labels)
    get_test_property(${t} FIXTURES_REQUIRED fixtures)
    get_test_property(${t} FIXTURES_SETUP setups)
    get_test_property(${t} RUN_SERIAL serial)
    # DV's tests are tests/dv.cmake's (labelled dv there): one named as DV's (dv, _dv, dv31_, ...) or needing DV's
    # assets elsewhere would run without FLYX_TEST_DV and in every preset.
    if(NOT "dv" IN_LIST labels AND (t MATCHES "(^|_)dv([0-9_]|$)" OR "dv_assets" IN_LIST fixtures))
      string(APPEND errors "  ${t}: a DV test (by its name or DV's assets) outside tests/dv.cmake\n")
    endif()
    set(tier ${labels})
    list(FILTER tier INCLUDE REGEX "^(unit|tiny|golden|tools)$")
    list(LENGTH tier n)
    if(n EQUAL 0)
      if(fixtures)
        set(tier tiny)
      else()
        set(tier unit)
      endif()
      set_property(TEST ${t} APPEND PROPERTY LABELS ${tier})
      list(APPEND labels ${tier})
    elseif(n GREATER 1)
      string(APPEND errors "  ${t}: tiers ${tier} (one only)\n")
    endif()
    foreach(l IN LISTS labels)
      if(NOT l IN_LIST tier_labels AND NOT l IN_LIST aspect_labels)
        string(APPEND errors "  ${t}: unknown label ${l} (the tiers: ${tier_labels}; the aspects: ${aspect_labels})\n")
      endif()
    endforeach()
    if(serial AND NOT "gpu_serial" IN_LIST labels)
      string(APPEND errors "  ${t}: RUN_SERIAL is for the tests measuring the device (flyx_add_golden_test SERIAL)\n")
    endif()
    if("gpu_serial" IN_LIST labels AND NOT "gpu" IN_LIST labels)
      string(APPEND errors "  ${t}: gpu_serial outside a GPU test\n")
    endif()
    if("smoke" IN_LIST labels AND "dv" IN_LIST labels)
      string(APPEND errors "  ${t}: smoke on a DV test (quick runs no DV)\n")
    endif()
    # A slot of the GPU (tests/gpu_resources.json, the presets' resourceSpecFile): CPU tests run beside the GPU's at
    # any -j, the GPU's bounded on their own.  Without a resource spec (no preset) ctest ignores it.  And
    # gpu_preflight before it (the GPU not already in use by another program), which takes no slot itself.
    if("gpu" IN_LIST labels AND NOT "gpu_preflight" IN_LIST setups)
      set_property(TEST ${t} PROPERTY RESOURCE_GROUPS "gpus:1")
      set_property(TEST ${t} APPEND PROPERTY FIXTURES_REQUIRED gpu_preflight)
    endif()
    if("smoke" IN_LIST labels OR (tier MATCHES "^(unit|tiny)$" AND NOT "gpu" IN_LIST labels
                                  AND NOT "dv" IN_LIST labels))
      set_property(TEST ${t} APPEND PROPERTY LABELS quick)
    endif()
  endforeach()
  if(errors)
    message(FATAL_ERROR "test labels (tests/CMakeLists.txt, flyx_finish_test_labels):\n${errors}")
  endif()
endfunction()

function(flyx_register_test name target)
  add_test(NAME ${name} COMMAND ${target} ${ARGN})
endfunction()

function(flyx_add_unit_test target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE flyx_core ${ARGN})
endfunction()

function(flyx_add_py_test name script)
  cmake_parse_arguments(ARG "" "TIMEOUT;SKIP_RETURN_CODE" "ARGS;FIXTURES;ENVIRONMENT;PY_FLAGS;LABELS" ${ARGN})
  add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE} ${ARG_PY_FLAGS} ${script} ${ARG_ARGS})
  if(ARG_FIXTURES)
    set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED "${ARG_FIXTURES}")
  endif()
  if(ARG_TIMEOUT)
    set_tests_properties(${name} PROPERTIES TIMEOUT ${ARG_TIMEOUT})
  endif()
  if(ARG_SKIP_RETURN_CODE)
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE ${ARG_SKIP_RETURN_CODE})
  endif()
  if(ARG_ENVIRONMENT)
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "${ARG_ENVIRONMENT}")
  endif()
  set(labels ${ARG_LABELS})
  if(script MATCHES "/bench/")
    list(APPEND labels tools)  # a bench script's own test
  endif()
  if(script MATCHES "/apps/flyx-server/")
    list(APPEND labels server)
  endif()
  if(labels)
    set_property(TEST ${name} APPEND PROPERTY LABELS ${labels})
  endif()
endfunction()

function(flyx_add_golden_test name)
  cmake_parse_arguments(ARG "SERIAL" "ENV" "ARGS;FIXTURES;LABELS" ${ARGN})
  add_test(NAME ${name} COMMAND golden_test ${ASSETS} ${ARG_ARGS})
  set(fixtures ${ARG_FIXTURES})
  if(NOT fixtures)
    set(fixtures assets)
  endif()
  set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED "${fixtures}")
  if(ARG_ENV)
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "${ARG_ENV}")
  endif()
  set(labels golden ${ARG_LABELS})
  if(ARG_SERIAL)
    set_tests_properties(${name} PROPERTIES RUN_SERIAL TRUE)
    list(APPEND labels gpu_serial)
  endif()
  set_property(TEST ${name} APPEND PROPERTY LABELS ${labels})
endfunction()

flyx_add_unit_test(tokenizer_test core/tokenizer/tokenizer_test.cpp)
flyx_add_unit_test(merge_test core/tokenizer/merge_test.cpp)
flyx_add_unit_test(flyx_format_test core/modelfile/flyx_test.cpp)
flyx_add_unit_test(drafter_dir_test core/modelfile/drafter_dir_test.cpp)
flyx_add_unit_test(peft_dir_test core/modelfile/peft_dir_test.cpp)
flyx_add_unit_test(hub_test core/hub/hub_test.cpp Threads::Threads)
flyx_add_unit_test(chat_test core/render/chat_test.cpp)
flyx_add_unit_test(tags_test models/formats/tags_test.cpp)
flyx_add_unit_test(qwen35_format_test models/formats/qwen35_test.cpp)
flyx_add_unit_test(grammar_test models/formats/grammar_test.cpp)
flyx_add_unit_test(xgrammar_local_changes_test third_party/xgrammar/local_changes_test.cpp flyx_xgrammar)
flyx_add_unit_test(telemetry_test core/exec/telemetry_test.cpp)
flyx_add_unit_test(snapshot_store_test core/session/store_test.cpp)
flyx_add_unit_test(parked_edit_test core/session/parked_edit_test.cpp)
flyx_add_unit_test(request_test core/exec/request_test.cpp)
flyx_add_unit_test(reply_text_test core/exec/reply_text_test.cpp)
flyx_add_unit_test(chooser_test core/speculation/chooser_test.cpp)
flyx_add_unit_test(calibrate_test core/speculation/calibrate_test.cpp)
flyx_add_unit_test(budget_test core/plan/budget_test.cpp)
flyx_add_unit_test(golden_test e2e/golden_test.cpp)
flyx_add_unit_test(gguf_test core/modelfile/gguf_test.cpp)
flyx_add_unit_test(ggml_test ops/dtype/ggml_test.cpp)
flyx_add_unit_test(sums_test ops/sums_test.cpp)
flyx_add_unit_test(dv_tree_test ops/draft/dv_tree_test.cpp)
flyx_add_unit_test(canvas_test ops/canvas/canvas_test.cpp)
flyx_add_unit_test(gguf_models_test models/gguf_test.cpp)
flyx_add_unit_test(dflash2_models_test models/drafters/dflash2_test.cpp)
flyx_add_unit_test(dv_models_test models/dv_test.cpp)
flyx_add_unit_test(llama_models_test models/llama/llama_test.cpp)
flyx_add_unit_test(gguf_e2e_test e2e/gguf_test.cpp)
flyx_add_unit_test(vision_test e2e/vision_test.cpp)
add_dependencies(vision_test flyx-cpu_ref)
if(TARGET flyx-cuda)
  add_dependencies(vision_test flyx-cuda)
endif()
flyx_add_unit_test(inkling2_test e2e/inkling2_test.cpp)
add_dependencies(inkling2_test flyx-cpu_ref)
# Each run writes the models it reads: a directory of its own, so the two can run at once (ctest -j).
flyx_register_test(inkling2_cpu_ref inkling2_test ${CMAKE_BINARY_DIR}/inkling2_assets/cpu_ref cpu_ref)
set_tests_properties(inkling2_cpu_ref PROPERTIES FIXTURES_SETUP inkling2_models)

add_executable(numbers_test apps/common/numbers_test.cpp ${CMAKE_SOURCE_DIR}/apps/common/numbers.cpp)
target_include_directories(numbers_test PRIVATE ${CMAKE_SOURCE_DIR})
add_executable(model_source_test apps/common/model_source_test.cpp ${CMAKE_SOURCE_DIR}/apps/common/model_source.cpp)
target_link_libraries(model_source_test PRIVATE flyx_core)
add_executable(known_drafters_test apps/common/known_drafters_test.cpp ${CMAKE_SOURCE_DIR}/apps/common/model_source.cpp)
target_link_libraries(known_drafters_test PRIVATE flyx_core)
add_executable(drafter_speeds_test apps/flyx-server/drafter_speeds_test.cpp ${CMAKE_SOURCE_DIR}/apps/flyx-server/drafter_speeds.cpp)
target_link_libraries(drafter_speeds_test PRIVATE flyx_core)
add_executable(strata_snapshot_test backends/cuda/strata_snapshot_test.cpp)
target_include_directories(strata_snapshot_test PRIVATE ${CMAKE_SOURCE_DIR})
add_executable(draft_ring_test backends/cuda/draft_ring_test.cpp)
target_include_directories(draft_ring_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)
add_executable(window_ring_test backends/cuda/window_ring_test.cpp)
target_include_directories(window_ring_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)
add_executable(response_store_test apps/flyx-server/response_store_test.cpp)
target_include_directories(response_store_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(response_store_test PRIVATE Threads::Threads)
add_executable(request_log_test apps/flyx-server/request_log_test.cpp ${CMAKE_SOURCE_DIR}/apps/flyx-server/request_log.cpp)
target_include_directories(request_log_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(request_log_test PRIVATE flyx_core Threads::Threads)
add_executable(agent_launch_test apps/flyx-server/agent_launch_test.cpp ${CMAKE_SOURCE_DIR}/apps/flyx-server/agent_launch.cpp)
target_include_directories(agent_launch_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/third_party)
add_executable(gpu_apps_test apps/flyx-server/gpu_apps_test.cpp ${CMAKE_SOURCE_DIR}/apps/flyx-server/gpu_apps.cpp)
target_include_directories(gpu_apps_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(gpu_apps_test PRIVATE flyx_core)
add_executable(http_util_test apps/flyx-server/http_util_test.cpp
                              ${CMAKE_SOURCE_DIR}/apps/flyx-server/http_util.cpp
                              ${CMAKE_SOURCE_DIR}/apps/flyx-server/model_host.cpp
                              ${CMAKE_SOURCE_DIR}/apps/flyx-server/request_log.cpp
                              ${CMAKE_SOURCE_DIR}/apps/flyx-server/drafter_speeds.cpp
                              ${CMAKE_SOURCE_DIR}/apps/common/model_source.cpp
                              ${CMAKE_SOURCE_DIR}/apps/common/numbers.cpp)
target_include_directories(http_util_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(http_util_test PRIVATE flyx_core Threads::Threads)

add_library(flyx-test-host-kv SHARED core/session/backend_adapter.cpp)
target_include_directories(flyx-test-host-kv PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(flyx-test-host-kv PROPERTIES PREFIX "lib")
flyx_add_unit_test(backend_errors_test core/exec/backend_errors_test.cpp flyx-test-host-kv)
flyx_register_test(backend_errors backend_errors_test)
flyx_add_unit_test(session_budget_test core/session/budget_test.cpp flyx-test-host-kv)
add_dependencies(session_budget_test flyx-cpu_ref)
flyx_add_unit_test(generation_test core/exec/generation_test.cpp flyx-test-host-kv)
add_dependencies(generation_test flyx-cpu_ref)
flyx_add_unit_test(elastic_load_test core/exec/elastic_load_test.cpp flyx-test-host-kv)
add_dependencies(elastic_load_test flyx-cpu_ref)
flyx_add_unit_test(plans_test core/plan/plans_test.cpp)
flyx_add_unit_test(scheduler_test core/exec/scheduler_test.cpp flyx-test-host-kv)
# A model switch in flyx serve's host: the server's model host and model sources over the tiny model.
flyx_add_unit_test(model_host_test apps/flyx-server/model_host_test.cpp flyx-test-host-kv)
target_sources(model_host_test PRIVATE ${CMAKE_SOURCE_DIR}/apps/flyx-server/model_host.cpp
                                       ${CMAKE_SOURCE_DIR}/apps/common/model_source.cpp
                                       ${CMAKE_SOURCE_DIR}/apps/common/numbers.cpp
                                       ${CMAKE_SOURCE_DIR}/apps/flyx-server/drafter_speeds.cpp)
add_dependencies(model_host_test flyx-cpu_ref)

add_executable(capi_test core/exec/capi_test.cpp ${CMAKE_SOURCE_DIR}/core/exec/capi.cpp)
target_compile_definitions(capi_test PRIVATE FLYX_BUILDING)
target_link_libraries(capi_test PRIVATE flyx_core)

# Tiny random model (BF16 and an NVFP4 copy) + golden outputs from the Python
# reference (needs torch, transformers, tokenizers, safetensors).  Generated
# once per build tree.  DV's (make_tiny.py --group dv) are tests/dv.cmake's.
set(ASSETS ${CMAKE_BINARY_DIR}/assets)
add_test(NAME make_assets
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/e2e/make_tiny.py ${ASSETS} --group core)
set_tests_properties(make_assets PROPERTIES FIXTURES_SETUP assets)
# The tiny models again as GGUF files, and random blocks of every GGML block
# type with what the gguf package (a test dependency) decodes them to.
add_test(NAME make_gguf_assets
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/core/modelfile/make_gguf.py ${ASSETS})
set_tests_properties(make_gguf_assets PROPERTIES FIXTURES_REQUIRED assets FIXTURES_SETUP gguf_assets)

add_test(NAME gguf_reader COMMAND gguf_test ${ASSETS})
add_test(NAME ggml_blocks COMMAND ggml_test ${ASSETS})
add_test(NAME gguf_models COMMAND gguf_models_test ${ASSETS})
# The DFlash2 drafter the architectures share (models/drafters/): its config checked before a tensor is bound, the
# dense targets that take one from llama.cpp's GGUF layout, and what is refused of a drafter file.
add_test(NAME dflash2_models COMMAND dflash2_models_test ${ASSETS})
# Qwen2's pre-tokenizer split (the published Qwen3 checkpoints), from GGUF vocabularies, against tokenizers.
add_test(NAME gguf_qwen2_nfc COMMAND tokenizer_test ${ASSETS}/qwen2_nfc.gguf ${ASSETS}/qwen2_nfc.json)
add_test(NAME gguf_qwen2_none COMMAND tokenizer_test ${ASSETS}/qwen2_none.gguf ${ASSETS}/qwen2_none.json)
# The tiny models from their GGUF files: the golden outputs, drafting, sampling, reuse.
add_test(NAME gguf_cpu_ref COMMAND gguf_e2e_test ${ASSETS} cpu_ref)
flyx_add_golden_test(golden_cpu_ref_qwen3_gguf ARGS cpu_ref qwen3 ${ASSETS}/tiny_qwen3_gguf.gguf FIXTURES gguf_assets)
# The tiny Qwen3.5 of GGUF (each head's gate rows after its q rows, value heads tiled) with a PEFT LoRA adapter in
# Hugging Face's layout beside it (--lora), against transformers with it merged.
flyx_add_golden_test(golden_cpu_ref_qwen35_gguf_lora ARGS cpu_ref qwen35_lora ${ASSETS}/tiny_gguf.gguf
                     ${ASSETS}/peft_qwen35 FIXTURES gguf_assets)
# The same GGUF file with a PEFT LoRA adapter beside it (--lora), against transformers with it merged.
flyx_add_golden_test(golden_cpu_ref_qwen3_gguf_lora ARGS cpu_ref qwen3_lora ${ASSETS}/tiny_qwen3_gguf.gguf
                     ${ASSETS}/peft_qwen3 FIXTURES gguf_assets)
set_tests_properties(gguf_reader ggml_blocks gguf_models dflash2_models gguf_qwen2_nfc gguf_qwen2_none gguf_cpu_ref
                     PROPERTIES FIXTURES_REQUIRED gguf_assets)

flyx_add_py_test(dspine_reference ${CMAKE_SOURCE_DIR}/tests/python/reference/dspine_test.py ARGS ${ASSETS} FIXTURES assets)
flyx_add_py_test(draft_reference ${CMAKE_SOURCE_DIR}/tests/python/reference/draft_test.py ARGS ${ASSETS} FIXTURES assets)
flyx_add_py_test(reference_vs_transformers ${CMAKE_SOURCE_DIR}/tests/python/reference/check_vs_hf.py
                 ARGS ${ASSETS}/hf_target ${ASSETS}/tiny.flyx FIXTURES assets)
add_test(NAME tokenizer COMMAND tokenizer_test ${ASSETS}/tiny.flyx ${ASSETS}/tokenizer_cases.json)
# MiniCPM5's pre-tokenizer (numbers in groups of three first), on the tiny Llama's tokenizer, against tokenizers.
add_test(NAME tokenizer_minicpm5
         COMMAND tokenizer_test ${ASSETS}/tiny_llama.flyx ${ASSETS}/tokenizer_cases_minicpm5.json)
add_test(NAME relative_backend_dir COMMAND flyx-cli run ${ASSETS}/tiny.flyx --backend cpu_ref -p hello -n 1)
set_tests_properties(relative_backend_dir PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:flyx-cli>"
                     ENVIRONMENT "FLYX_BACKEND_DIR=." FIXTURES_REQUIRED assets)

flyx_add_py_test(cli_utf8_paths ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-cli/utf8_paths.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets)
flyx_add_py_test(kv_method_flags ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-cli/kv_method.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets)
flyx_add_py_test(cli_usage ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-cli/usage.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets)
flyx_add_py_test(server_ephemeral_port ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/ephemeral_port.py
                 ARGS $<TARGET_FILE:flyx-cli> TIMEOUT 45)
flyx_add_py_test(remembered_model ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/remembered_model.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets)
flyx_add_py_test(server_lan_access ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/lan_access.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets)
flyx_add_py_test(lora_load ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/lora_load.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny_qwen3_gguf.gguf ${ASSETS}/peft_qwen3 FIXTURES gguf_assets
                 TIMEOUT 120)
flyx_add_py_test(queued_request_ttft ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/queued_request_ttft.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 90)
flyx_add_py_test(streaming_tool_calls ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/streaming_tool_calls.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 90)
flyx_add_py_test(api_compat ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/api_compat.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 90)
flyx_add_py_test(server_hardening ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/hardening.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 120)
flyx_add_py_test(queued_request_log ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/queued_request_log.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 90)
flyx_add_py_test(client_conversations ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/client_conversations.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 90)
flyx_add_py_test(codex_threads ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/codex_threads.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 90)
flyx_add_py_test(model_timestamps ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/model_timestamps.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets)

add_test(NAME chat COMMAND chat_test ${ASSETS}/tiny.flyx ${ASSETS}/chat_cases.json
                                     ${CMAKE_SOURCE_DIR}/tests/core/render/templates)
# The tokenizer's BPE against the rule a pass per merge, and the .flyx reader on wrong files: on files the tests
# write themselves (no model, no Python).
add_test(NAME bpe_merges COMMAND merge_test ${CMAKE_BINARY_DIR})
add_test(NAME flyx_reader COMMAND flyx_format_test ${CMAKE_BINARY_DIR})
# Models named as the Hugging Face Hub names them: fetching from a fake Hub on this machine, and a
# DFlash2 drafter read from its authors' directory as python/convert writes it.
add_test(NAME hub COMMAND hub_test)
set_tests_properties(hub PROPERTIES SKIP_RETURN_CODE 77)
flyx_add_py_test(hub_concurrent ${CMAKE_CURRENT_SOURCE_DIR}/core/hub/check_concurrent_fetch.py
                 ARGS $<TARGET_FILE:hub_test> SKIP_RETURN_CODE 77 TIMEOUT 90)
add_test(NAME drafter_dir COMMAND drafter_dir_test ${ASSETS})
set_tests_properties(drafter_dir PROPERTIES FIXTURES_REQUIRED gguf_assets)
add_test(NAME peft_dir COMMAND peft_dir_test)

# Numbers on the command line: all of the text a number, inside what the option takes.
add_test(NAME flag_numbers COMMAND numbers_test)
# The DSpark drafter published beside a model, taken by default: beside a GGUF on disk, and in its repository (a
# Hugging Face cache read offline).
flyx_register_test(model_source model_source_test)
add_test(NAME known_drafters COMMAND known_drafters_test)
set_tests_properties(known_drafters PROPERTIES ENVIRONMENT "HF_HUB_OFFLINE=1")  # no network
add_test(NAME drafter_speeds COMMAND drafter_speeds_test)
add_test(NAME chat_replies COMMAND tags_test)
# Qwen3.5's own renderer against the Jinja engine on generated conversations, and its calls read back.
add_test(NAME chat_qwen35 COMMAND qwen35_format_test ${ASSETS}/tiny.flyx
                                  ${CMAKE_SOURCE_DIR}/tests/core/render/templates/qwen3.5.jinja)
# Reply grammars (tool_choice, response_format) in both call languages, on the tiny model's tokenizer.
add_test(NAME reply_grammars COMMAND grammar_test ${ASSETS}/tiny.flyx)
flyx_register_test(xgrammar_local_changes xgrammar_local_changes_test)
add_test(NAME telemetry COMMAND telemetry_test)
add_test(NAME snapshot_store COMMAND snapshot_store_test)
add_test(NAME strata_snapshot COMMAND strata_snapshot_test)
flyx_register_test(draft_ring draft_ring_test)
flyx_register_test(window_ring window_ring_test)
add_test(NAME parked_edit COMMAND parked_edit_test ${ASSETS}/tiny.flyx)
set_tests_properties(parked_edit PROPERTIES FIXTURES_REQUIRED assets)
add_test(NAME response_store COMMAND response_store_test)
add_test(NAME request_log COMMAND request_log_test)
add_test(NAME agent_launch COMMAND agent_launch_test)
add_test(NAME gpu_apps COMMAND gpu_apps_test)
add_test(NAME http_util COMMAND http_util_test)
# Which round next: the drafter's widths by acceptance, lookup rounds, the rest after a poor one (no model).
add_test(NAME round_chooser COMMAND chooser_test)
# Round costs timed at load at two contexts: what a device takes, not what another program's work made them (no model).
add_test(NAME round_costs_plausible COMMAND calibrate_test)
# The memory budget: a context the caller did not ask for shrinks to what fits (no model, no GPU).
add_test(NAME memory_budget COMMAND budget_test)
# The round plans: a drafter whose state follows its own rounds (DV) beside plain decoding's and lookup's (pinned:
# alone) where the backend runs them, else as without it.
add_test(NAME round_plans COMMAND plans_test)
# A reply's text: characters shown whole, stop strings, which tokens count as shown (the tiny model's tokenizer).
add_test(NAME reply_text COMMAND reply_text_test ${ASSETS}/tiny.flyx)
# The request pipeline every entry point shares, on the tiny model (float64 backend) with Qwen3.5's template; a raw
# prompt on the tiny Llama, whose tokenizer puts <s> before a text (MiniCPM5's post-processor).
add_test(NAME request COMMAND request_test ${ASSETS}/tiny.flyx
                                 ${CMAKE_SOURCE_DIR}/tests/core/render/templates/qwen3.5.jinja
                                 ${ASSETS}/tiny_llama.flyx)
add_test(NAME ref_sums COMMAND sums_test)
add_test(NAME vision COMMAND vision_test)
set_tests_properties(vision PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vision_test>")
# The target's own adapters (a V-LoRA, ABI 43) from the tiny Llama's V-LoRA files (models/llama): on every linear of
# every layer, their rank and scale; headers and tensors that do not fit refused.  nodv: without tiny_llama_dv_lora.flyx
# (DV's assets: llama_models_dv in tests/dv.cmake runs the rest).
add_test(NAME llama_models COMMAND llama_models_test ${ASSETS} nodv)
# A V-LoRA alone (scale 0.5): its tensors are export_dv.py --vlora's (bifill-dv's checkpoint's, byte for byte), as the
# reference loads them.
flyx_add_py_test(lora_convert ${CMAKE_SOURCE_DIR}/tests/python/convert/check_dv_convert.py PY_FLAGS -I
                 ARGS ${ASSETS}/tiny_llama_lora.flyx ${ASSETS}/dv_tiny/export_vlora --vlora ${ASSETS}/dv_tiny/vlora16.pt
                      --hf ${ASSETS}/hf_llama
                 FIXTURES assets)
# The bf16 prefill residual's decision on each backend (-v): taken by an all-W4A4 stack, never with adapters.
# --fp4-kernels: the build has the FP4 tensor cores' kernels (sm_120a), so a device of compute capability 12.0 must
# compute NVFP4 inputs, not refuse them.
if(TARGET flyx-cuda)
  set(FLYX_RESIDUAL_BACKENDS cpu_ref cuda)
  if(TARGET flyx_cuda_sm120a)
    list(PREPEND FLYX_RESIDUAL_BACKENDS --fp4-kernels)
  endif()
else()
  set(FLYX_RESIDUAL_BACKENDS cpu_ref)
endif()
flyx_add_py_test(residual_decision ${CMAKE_SOURCE_DIR}/tests/e2e/residual_decision.py PY_FLAGS -I
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS} ${FLYX_RESIDUAL_BACKENDS} FIXTURES assets)
# What the converter refuses (a scaled rotary in either key's form) and the tokens a post-processor puts before a text.
flyx_add_py_test(converter ${CMAKE_SOURCE_DIR}/tests/python/convert/convert_test.py)
# flyx bench --prompts: each line's prompt, the generated tokens and rounds in the record (an end token accepted as a
# draft marked), drafted answers plain's, bad prompts and output paths refused before anything runs.
flyx_add_py_test(bench_prompts ${CMAKE_SOURCE_DIR}/tests/apps/flyx-cli/bench_prompts_test.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS} ${CMAKE_SOURCE_DIR}/tests/core/render/templates/qwen3.5.jinja
                 FIXTURES assets)
# The same where a chat without thinking carries a reply grammar (the tiny Inkling-2 model inkling2_cpu_ref writes):
# every prompt of the set keeps its grammar until it runs, and its reply is held to it.
flyx_add_py_test(bench_prompts_grammar ${CMAKE_SOURCE_DIR}/tests/apps/flyx-cli/bench_prompts_grammar_test.py
                 ARGS $<TARGET_FILE:flyx-cli> ${CMAKE_BINARY_DIR}/inkling2_assets/cpu_ref/tiny_inkling2_bf16.gguf
                 FIXTURES inkling2_models)
# Two builds' flyx bench --prompts records of one command: the same answers and rounds over the prompts both hold, and
# what fails (bench/ab/same_runs.py; records made up, no model).
flyx_add_py_test(same_runs ${CMAKE_SOURCE_DIR}/tests/bench/ab/same_runs_test.py)
flyx_add_py_test(layering ${CMAKE_SOURCE_DIR}/tools/check_layering.py ARGS ${CMAKE_SOURCE_DIR})
flyx_add_py_test(numerics ${CMAKE_SOURCE_DIR}/tools/check_numerics.py ARGS ${CMAKE_SOURCE_DIR})
flyx_add_py_test(docs ${CMAKE_SOURCE_DIR}/tools/check_docs.py ARGS ${CMAKE_SOURCE_DIR})
flyx_add_py_test(check_glibc ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_glibc_test.py LABELS tools smoke)
find_program(FLYX_POWERSHELL NAMES powershell pwsh)
if(FLYX_POWERSHELL)
  flyx_add_py_test(release_zip ${CMAKE_CURRENT_SOURCE_DIR}/tools/release_zip_test.py ARGS ${FLYX_POWERSHELL} LABELS tools)
endif()
flyx_add_py_test(agent_replay ${CMAKE_CURRENT_SOURCE_DIR}/bench/replay/agent_replay_test.py)
flyx_add_py_test(trace_layer_roles ${CMAKE_CURRENT_SOURCE_DIR}/bench/trace/layer_roles_test.py)
flyx_add_py_test(two_at_once ${CMAKE_CURRENT_SOURCE_DIR}/bench/replay/two_at_once_test.py TIMEOUT 15)
if(UNIX)
  flyx_add_py_test(drafter_captures ${CMAKE_CURRENT_SOURCE_DIR}/bench/drafter_captures.py)
  flyx_add_py_test(external_comparisons ${CMAKE_CURRENT_SOURCE_DIR}/bench/external_comparisons.py)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  flyx_add_py_test(quickstart_venv ${CMAKE_CURRENT_SOURCE_DIR}/tools/quickstart_venv.py)
  flyx_add_py_test(cpu_session_allocation ${CMAKE_CURRENT_SOURCE_DIR}/backends/cpu_ref/allocation_test.py
                   ARGS ${ASSETS}/tiny.flyx $<TARGET_FILE:flyx> FIXTURES assets)
endif()
# Tuning tables: a GPU's own rules for any family, inherited ones, and the tables refused.
flyx_add_py_test(gemm_tables ${CMAKE_SOURCE_DIR}/tests/tools/codegen/gen_gemm_tables_test.py ARGS ${CMAKE_SOURCE_DIR})

# The scheduler: requests at once take turns and give what they give alone; a busy session is waited for.
add_test(NAME scheduler_cpu_ref COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref)
add_test(NAME scheduler_queue COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref queue)
add_test(NAME scheduler_prewarmed_history COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref history)
# Raw prompts of different clients (the same text, or one extending another's) each run to their own end; a shared
# conversation id replaces.
flyx_register_test(scheduler_raw_clients scheduler_test ${ASSETS}/tiny.flyx cpu_ref raw-clients)
add_test(NAME scheduler_interrupted_prewarm COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref interrupted-prewarm)
add_test(NAME scheduler_reassigned_prewarm COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref reassigned-prewarm)
add_test(NAME scheduler_repeated_prewarm COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref repeated-prewarm)
add_test(NAME scheduler_marked_system COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref marked-system)
add_test(NAME session_host_budget COMMAND session_budget_test ${ASSETS}/tiny.flyx)
# A model switch loads the new model once the old one is closed, not once nobody holds it: the last request lets go
# of it on its own thread while the backend's close is held up there, and the backend holds one model at a time.
flyx_register_test(model_host model_host_test ${ASSETS}/tiny.flyx)
add_test(NAME generation_edges COMMAND generation_test ${ASSETS}/tiny.flyx)
flyx_register_test(elastic_load elastic_load_test ${ASSETS}/tiny.flyx)
# The C API: a chat request with the file's template, and with the one the load options give.
add_test(NAME capi COMMAND capi_test ${ASSETS}/tiny.flyx cpu_ref)
flyx_add_py_test(python_binding ${CMAKE_CURRENT_SOURCE_DIR}/python/flyx/binding_test.py
                 ARGS ${ASSETS}/tiny.flyx $<TARGET_FILE:flyx> FIXTURES assets)
flyx_add_py_test(convert_utf8 ${CMAKE_CURRENT_SOURCE_DIR}/python/convert/utf8_test.py
                 PY_FLAGS -X utf8=0 ARGS ${ASSETS} $<TARGET_FILE:flyx>
                 FIXTURES assets ENVIRONMENT "LC_ALL=C" "PYTHONCOERCECLOCALE=0")
flyx_add_py_test(convert_write ${CMAKE_CURRENT_SOURCE_DIR}/python/convert/write_test.py)
flyx_add_py_test(convert_cli ${CMAKE_CURRENT_SOURCE_DIR}/python/convert/cli_test.py
                 PY_FLAGS -X utf8=1 ARGS ${ASSETS} FIXTURES assets)

flyx_add_golden_test(golden_cpu_ref ARGS cpu_ref)
flyx_add_golden_test(golden_cpu_ref_nvfp4 ARGS cpu_ref nvfp4)
# A second architecture (Qwen3) through models/, against transformers: no backend knows it by name.
flyx_add_golden_test(golden_cpu_ref_qwen3 ARGS cpu_ref qwen3)
# The tiny Qwen3 with a LoRA adapter as Hugging Face PEFT saves one (--lora: on some linears of some layers, float32),
# against transformers with it merged.
flyx_add_golden_test(golden_cpu_ref_qwen3_lora ARGS cpu_ref qwen3_lora ${ASSETS}/tiny_qwen3.flyx ${ASSETS}/peft_qwen3)
# The tiny Qwen3.5 (gated attention, gated delta nets, its drafters) with a PEFT LoRA adapter on attention, GDN and
# MLP linears (--lora), against transformers with it merged; drafting must give the same tokens.
flyx_add_golden_test(golden_cpu_ref_qwen35_lora ARGS cpu_ref qwen35_lora ${ASSETS}/tiny.flyx ${ASSETS}/peft_qwen35)
# Llama's attention without q/k norms (MiniCPM5's layers), against transformers.
flyx_add_golden_test(golden_cpu_ref_llama ARGS cpu_ref llama)
# The tiny Llama with a V-LoRA of its own (the target's adapters, ABI 43) against transformers with it merged.
flyx_add_golden_test(golden_cpu_ref_llama_lora ARGS cpu_ref llama_lora)
# GDN heads of 128: the CUDA prefill takes the chunked recurrence.
flyx_add_golden_test(golden_cpu_ref_gdn128 ARGS cpu_ref gdn128)
# A DFlash2 drafter whose layers attend a sliding window (z-lab's 27B drafter's), shorter than its block here.
flyx_add_golden_test(golden_cpu_ref_window ARGS cpu_ref window)
# A DSpine drafter in place of DFlash2's (the whole battery: its rounds, oracle runs, rollback, sampling, ...), and one
# whose layers attend a causal sliding window of 8 (the 27B's drafter's), shorter than its block.
flyx_add_golden_test(golden_cpu_ref_dspine ARGS cpu_ref dspine)
flyx_add_golden_test(golden_cpu_ref_dspine1 ARGS cpu_ref dspine1)  # a drafter of one layer
flyx_add_golden_test(golden_cpu_ref_dspine_window ARGS cpu_ref dspine_window)
flyx_add_golden_test(golden_cpu_ref_dspine_nvfp4 ARGS cpu_ref dspine_nvfp4)  # beside the NVFP4 copy
flyx_add_golden_test(golden_cpu_ref_dspine8 ARGS cpu_ref dspine8)  # a block of 8
# The tiny Qwen3 with a DSpine drafter (taps and context in a second architecture's programs): transformers' tokens.
flyx_add_golden_test(golden_cpu_ref_qwen3_dspine ARGS cpu_ref qwen3 ${ASSETS}/tiny_qwen3_dspine.flyx - dspine)
flyx_add_golden_test(golden_cpu_ref_qwen3_dspine_lora ARGS cpu_ref qwen3_lora ${ASSETS}/tiny_qwen3_dspine.flyx
                     ${ASSETS}/peft_qwen3 dspine)
# DFlash (z-lab's DFlashDraftModel): DFlash2's block without convolutions or selector, in its place
flyx_add_golden_test(golden_cpu_ref_dflash ARGS cpu_ref dflash)
flyx_add_golden_test(golden_cpu_ref_dflash8 ARGS cpu_ref dflash8)  # a block of 8
flyx_add_golden_test(golden_cpu_ref_qwen3_dflash ARGS cpu_ref qwen3 ${ASSETS}/tiny_qwen3_dflash.flyx - dflash)

set_tests_properties(tokenizer tokenizer_minicpm5 chat chat_qwen35 reply_grammars request reply_text
                     scheduler_cpu_ref scheduler_queue scheduler_prewarmed_history scheduler_raw_clients
                     scheduler_interrupted_prewarm
                     scheduler_reassigned_prewarm scheduler_repeated_prewarm scheduler_marked_system session_host_budget generation_edges elastic_load capi
                     model_host
                     llama_models PROPERTIES FIXTURES_REQUIRED assets)

# The tests that need the GPU (label gpu): the ones below, and those above that take the CUDA backend too.
get_property(flyx_cpu_tests DIRECTORY PROPERTY TESTS)
if(TARGET flyx-cuda)
  # Before every GPU test (flyx_finish_test_labels: FIXTURES_REQUIRED gpu_preflight): fails, naming who holds it, when
  # another program has more than FLYX_TEST_GPU_MAX_USED_MIB (2560) of the GPU; skipped without nvidia-smi.
  add_test(NAME gpu_preflight COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gpu_preflight.py)
  set_tests_properties(gpu_preflight PROPERTIES FIXTURES_SETUP gpu_preflight SKIP_RETURN_CODE 77)
  add_executable(cuda_loader_test backends/cuda/loader_test.cpp ${CMAKE_SOURCE_DIR}/backends/cuda/backend.cpp)
  target_compile_definitions(cuda_loader_test PRIVATE FLYX_TEST_DEVICE_ALLOC)
  target_link_libraries(cuda_loader_test PRIVATE flyx_cuda_host)
  flyx_register_test(cuda_loader cuda_loader_test)
  flyx_register_test(elastic_load_cuda elastic_load_test ${ASSETS}/tiny.flyx cuda)
  set_tests_properties(elastic_load_cuda PROPERTIES FIXTURES_REQUIRED assets)

  # Packed block weights against the float64 backend on the same files; the generic regions too.  Programmatic
  # dependent launch (each kernel starts while the one before finishes) on and off: the same tokens.  Whatever the
  # GPU's table makes the default is one of the two, so neither these nor the golden tests below run it a third time.
  add_test(NAME gguf_cuda_unfused COMMAND gguf_e2e_test ${ASSETS} cuda fusion=0)
  add_test(NAME gguf_cuda_pdl COMMAND gguf_e2e_test ${ASSETS} cuda pdl=1)
  add_test(NAME gguf_cuda_no_pdl COMMAND gguf_e2e_test ${ASSETS} cuda pdl=0)
  flyx_add_golden_test(golden_cuda_qwen3_gguf ARGS cuda qwen3 ${ASSETS}/tiny_qwen3_gguf.gguf FIXTURES gguf_assets)
  flyx_add_golden_test(golden_cuda_qwen3_gguf_lora ARGS cuda qwen3_lora ${ASSETS}/tiny_qwen3_gguf.gguf
                       ${ASSETS}/peft_qwen3 FIXTURES gguf_assets)
  # A tiny Qwen3 of GGML block weights (tiny_qwen3_gguf.gguf is float32, read as bf16), held to its own plain
  # decoding (golden_quant.json): tree rounds of more than 16 rows on packed weights, a round's GEMMs and its packed
  # head 16 rows at a time; Q8_0, and Q4_K for the K types' round kernels.
  flyx_add_golden_test(golden_cuda_quant_q8_0 ARGS cuda quant ${ASSETS}/quant_Q8_0.gguf FIXTURES gguf_assets)
  flyx_add_golden_test(golden_cuda_quant_q4_k ARGS cuda quant ${ASSETS}/quant_Q4_K.gguf FIXTURES gguf_assets)
  set_tests_properties(gguf_cuda_unfused gguf_cuda_pdl gguf_cuda_no_pdl PROPERTIES FIXTURES_REQUIRED gguf_assets)

  # A family of kernels a file (op_test_*.cu), helpers in op_test.cuh, main() in op_test.cu.
  add_executable(cuda_op_test backends/cuda/op_test.cu backends/cuda/op_test_gemm.cu backends/cuda/op_test_blocks.cu
                 backends/cuda/op_test_attention.cu backends/cuda/op_test_gdn.cu backends/cuda/op_test_nvfp4.cu
                 backends/cuda/op_test_head.cu backends/cuda/op_test_sampling.cu backends/cuda/op_test_layers.cu
                 backends/cuda/op_test_dv.cu backends/cuda/op_test_vision.cu
                 backends/cuda/op_test_inkling2.cu backends/cuda/op_test_dspine.cu)
  if(MSVC)
    # The MSVC INFINITY macro triggers nvcc's overflow warning at every use, as in flyx_cuda_kernels.
    target_compile_options(cuda_op_test PRIVATE --diag-suppress=221)
  endif()
  target_link_libraries(cuda_op_test PRIVATE flyx_cuda_kernels)
  add_dependencies(cuda_op_test flyx_gemm_tables)  # op_test_gemm.cu reads the generated tables
  set_target_properties(cuda_op_test PROPERTIES CUDA_ARCHITECTURES "${FLYX_CUDA_ARCHS}")
  add_test(NAME cuda_ops COMMAND cuda_op_test)
  add_dependencies(inkling2_test flyx-cuda)
  flyx_register_test(inkling2_cuda inkling2_test ${CMAKE_BINARY_DIR}/inkling2_assets/cuda cuda)

  add_test(NAME scheduler_cuda COMMAND scheduler_test ${ASSETS}/tiny.flyx cuda)
  set_tests_properties(scheduler_cuda PROPERTIES FIXTURES_REQUIRED assets)

  flyx_add_golden_test(golden_cuda_nvfp4 ARGS cuda nvfp4)
  # the generic implementation of every region (no fused overrides) must pass as well
  flyx_add_golden_test(golden_cuda_unfused ARGS cuda nvfp4 ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_pdl ARGS cuda ENV FLYX_BACKEND_OPTIONS=pdl=1)
  flyx_add_golden_test(golden_cuda_no_pdl ARGS cuda ENV FLYX_BACKEND_OPTIONS=pdl=0)
  flyx_add_golden_test(golden_cuda_qwen3 ARGS cuda qwen3 ENV FLYX_BACKEND_OPTIONS=embed=host)
  flyx_add_golden_test(golden_cuda_qwen3_unfused ARGS cuda qwen3 ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_qwen3_lora ARGS cuda qwen3_lora ${ASSETS}/tiny_qwen3.flyx ${ASSETS}/peft_qwen3)
  flyx_add_golden_test(golden_cuda_qwen35_lora ARGS cuda qwen35_lora ${ASSETS}/tiny.flyx ${ASSETS}/peft_qwen35)
  flyx_add_golden_test(golden_cuda_qwen35_lora_unfused ARGS cuda qwen35_lora ${ASSETS}/tiny.flyx ${ASSETS}/peft_qwen35
                       ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_qwen35_gguf_lora ARGS cuda qwen35_lora ${ASSETS}/tiny_gguf.gguf
                       ${ASSETS}/peft_qwen35 FIXTURES gguf_assets)
  flyx_add_golden_test(golden_cuda_qwen3_lora_unfused ARGS cuda qwen3_lora ${ASSETS}/tiny_qwen3.flyx
                       ${ASSETS}/peft_qwen3 ENV FLYX_BACKEND_OPTIONS=fusion=0)
  # Its adapters' rank (6, padded to 8) one lane an output: the SwiGLU takes gate's and up's products by default, here
  # lora_up (the same bits).
  flyx_add_golden_test(golden_cuda_qwen3_lora_no_glu ARGS cuda qwen3_lora ${ASSETS}/tiny_qwen3.flyx
                       ${ASSETS}/peft_qwen3 ENV FLYX_BACKEND_OPTIONS=lora_glu=0)
  # The tiny Qwen3's head is not its embedding: the table in host memory by default, here on the GPU.
  flyx_add_golden_test(golden_cuda_qwen3_embed_gpu ARGS cuda qwen3 ENV FLYX_BACKEND_OPTIONS=embed=gpu)
  # Llama without q/k norms: the generic regions too, and its 16 query heads on 2 KV heads on tensor cores for every
  # row.
  flyx_add_golden_test(golden_cuda_llama ARGS cuda llama)
  flyx_add_golden_test(golden_cuda_llama_unfused ARGS cuda llama ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_llama_rows_mma ARGS cuda llama ENV FLYX_BACKEND_OPTIONS=rows_attention_from=0)
  # The tiny Llama's NVFP4 copy, held to its own plain decoding: tree rounds past 16 rows on gemm_fp4_rows.
  flyx_add_golden_test(golden_cuda_llama_nvfp4 ARGS cuda llama_nvfp4)
  flyx_add_golden_test(golden_cuda_llama_nvfp4_unfused ARGS cuda llama_nvfp4 ENV FLYX_BACKEND_OPTIONS=fusion=0)
  # A V-LoRA (the target's own adapters, ABI 43) on every pass: against transformers with it merged, with the generic
  # regions (gate 2) and the verify rows' attention on tensor cores; tree rounds of 31 to 63 drafts give plain
  # decoding's tokens (gate 4); beside NVFP4 linears, held to its own plain decoding.
  flyx_add_golden_test(golden_cuda_llama_lora ARGS cuda llama_lora)
  flyx_add_golden_test(golden_cuda_llama_lora_unfused ARGS cuda llama_lora ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_llama_lora_rows_mma ARGS cuda llama_lora
                       ENV FLYX_BACKEND_OPTIONS=rows_attention_from=0)
  flyx_add_golden_test(golden_cuda_llama_nvfp4_lora ARGS cuda llama_nvfp4_lora)
  flyx_add_golden_test(golden_cuda_llama_nvfp4_lora_unfused ARGS cuda llama_nvfp4_lora
                       ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_gdn128 ARGS cuda gdn128)
  flyx_add_golden_test(golden_cuda_gdn128_unfused ARGS cuda gdn128 ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_window ARGS cuda window)
  flyx_add_golden_test(golden_cuda_dspine ARGS cuda dspine)
  flyx_add_golden_test(golden_cuda_dspine_unfused ARGS cuda dspine ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_dspine_window ARGS cuda dspine_window)
  flyx_add_golden_test(golden_cuda_dspine_nvfp4 ARGS cuda dspine_nvfp4)  # its layers' copies beside NVFP4 weights
  flyx_add_golden_test(golden_cuda_dspine8 ARGS cuda dspine8)  # a block of 8
  flyx_add_golden_test(golden_cuda_dspine1 ARGS cuda dspine1)  # one layer: its cache binary16 beside fp8 ones
  flyx_add_golden_test(golden_cuda_qwen3_dspine ARGS cuda qwen3 ${ASSETS}/tiny_qwen3_dspine.flyx - dspine)
  flyx_add_golden_test(golden_cuda_qwen3_dspine_lora ARGS cuda qwen3_lora ${ASSETS}/tiny_qwen3_dspine.flyx
                       ${ASSETS}/peft_qwen3 dspine)
  flyx_add_golden_test(golden_cuda_dflash ARGS cuda dflash)
  flyx_add_golden_test(golden_cuda_dflash_unfused ARGS cuda dflash ENV FLYX_BACKEND_OPTIONS=fusion=0)
  # a block of 8: batched rounds of two sequences, each drafted alone (no grouped buffers), each its solo rounds
  flyx_add_golden_test(golden_cuda_dflash8 ARGS cuda dflash8)
  flyx_add_golden_test(golden_cuda_qwen3_dflash ARGS cuda qwen3 ${ASSETS}/tiny_qwen3_dflash.flyx - dflash)
  # decode and verification attention on tensor cores for every row (by default only past rows_attention_from keys)
  flyx_add_golden_test(golden_cuda_rows_mma ARGS cuda nvfp4 ENV FLYX_BACKEND_OPTIONS=rows_attention_from=0)
  flyx_add_golden_test(golden_cuda_qwen3_rows_mma ARGS cuda qwen3 ENV FLYX_BACKEND_OPTIONS=rows_attention_from=0)
  # greedy choices from the whole head, not the certified head
  flyx_add_golden_test(golden_cuda_full_head ARGS cuda nvfp4 ENV FLYX_BACKEND_OPTIONS=head_cert=0)
  # Sessions and engines give their device memory back, measured as the device's free memory: that is the whole
  # GPU's, so these run alone (RUN_SERIAL).  Every other GPU test runs beside the others and the CPU tests (ctest
  # -j, as the presets ask): their outputs do not depend on what else runs.
  flyx_add_golden_test(cuda_device_memory ARGS cuda --device-memory SERIAL)
  flyx_add_golden_test(cuda_device_memory_dspine ARGS cuda dspine --device-memory SERIAL)
  flyx_add_golden_test(cuda_device_memory_nvfp4 ARGS cuda nvfp4 --device-memory SERIAL)
  # The sessions' round graphs inside what they lease with the K/V and the low-bit copies in host
  # memory, across auto's switch: a process of its own (the driver keeps the graphs' memory of the one before).
  flyx_add_golden_test(cuda_device_memory_tiers ARGS cuda --device-memory=tiers SERIAL)
  # tree_bench times every width and shape from the same context, the prompt prefilled again before each timing: it
  # exits 1 when a timing's first round starts elsewhere (every one after the first did, each where the last ended).
  add_test(NAME tree_bench_context COMMAND tree_bench ${ASSETS}/tiny_llama.flyx --ctx-at 40 --widths 1,3,7 --rounds 4)
  set_tests_properties(tree_bench_context PROPERTIES FIXTURES_REQUIRED assets)
  get_property(flyx_gpu_tests DIRECTORY PROPERTY TESTS)
  list(REMOVE_ITEM flyx_gpu_tests ${flyx_cpu_tests})
  set_property(TEST ${flyx_gpu_tests} residual_decision vision APPEND PROPERTY LABELS gpu)
endif()

# On Windows these programs read their files by UTF-8 paths as flyx does (a build tree under a user folder named in
# Chinese): the same manifest (flyx_program).
get_property(flyx_dir_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(t IN LISTS flyx_dir_targets)
  get_target_property(flyx_target_type ${t} TYPE)
  if(flyx_target_type STREQUAL "EXECUTABLE")
    flyx_program(${t})
  endif()
endforeach()

# DV (not in production): its tests and assets only with FLYX_TEST_DV (the *-full configure presets).  Without it,
# dv_not_configured stands in for them: skipped, but failed where a test preset asks for DV (FLYX_TEST_REQUIRE_DV: the
# full presets), so that a full run in a tree configured without DV fails instead of passing short.
if(FLYX_TEST_DV)
  include(${CMAKE_CURRENT_SOURCE_DIR}/dv.cmake)
else()
  string(CONCAT flyx_dv_off "import os, sys; print('DV tests are not registered in this tree: configure it with "
                "-DFLYX_TEST_DV=ON (cmake --preset dev-full, cpu-full, ...-windows-full)'); "
                "sys.exit(1 if os.environ.get('FLYX_TEST_REQUIRE_DV') else 77)")
  add_test(NAME dv_not_configured COMMAND ${Python3_EXECUTABLE} -c "${flyx_dv_off}")
  set_tests_properties(dv_not_configured PROPERTIES SKIP_RETURN_CODE 77)
  set_property(TEST dv_not_configured APPEND PROPERTY LABELS unit dv)
endif()

# Labels flyx_finish_test_labels cannot infer (the registering functions give golden, gpu_serial, server and tools;
# the CUDA block gpu).  smoke: the golden and GPU tests quick runs too.  quick takes as long as its longest tests,
# cuda_ops (about 35 s on an RTX 5070 Ti) and golden_cuda_pdl (24 to 30 s): the others a few seconds each, so that
# they run beside those two (the gguf_cuda runs, as long as them or longer, would set quick's time instead).  The
# second architecture against transformers (gate 3, from its .flyx and its GGUF file), and on the GPU every kernel
# against float64 (gate 1), the tiny Qwen3.5 (MTP, DFlash2, sampling: gates 3 and 4), Qwen3 against transformers
# fused and generic (gates 2 and 3), Llama against transformers and its NVFP4 copy (NVFP4 on CUDA: gate 3), packed
# GGML blocks in tree rounds, Inkling-2 (fused == generic, DSpark == plain: gates 2 and 4) and the loader.
set_property(TEST make_assets APPEND PROPERTY LABELS tiny)          # makes the tiny models (no fixture before it)
set_property(TEST gguf_cpu_ref APPEND PROPERTY LABELS golden)       # gguf_e2e_test: the golden outputs from GGUF
set_property(TEST golden_cpu_ref_qwen3 golden_cpu_ref_qwen3_gguf APPEND PROPERTY LABELS smoke)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_property(TEST quickstart_venv APPEND PROPERTY LABELS tools)   # tools/quickstart.sh's environment
endif()
if(TARGET flyx-cuda)
  set_property(TEST gguf_cuda_unfused gguf_cuda_pdl gguf_cuda_no_pdl APPEND PROPERTY LABELS golden)
  set_property(TEST tree_bench_context APPEND PROPERTY LABELS tools)
  set_property(TEST cuda_ops golden_cuda_pdl golden_cuda_qwen3 golden_cuda_qwen3_unfused golden_cuda_llama
                    golden_cuda_llama_nvfp4 golden_cuda_quant_q4_k inkling2_cuda cuda_loader elastic_load_cuda vision
                    residual_decision APPEND PROPERTY LABELS smoke)
endif()
flyx_finish_test_labels()
