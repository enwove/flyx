# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
# bench/
#   microbench/  kernel timings (cuda_bench; the numbers behind the tuning tables)
#   ab/          A/B runs of two builds or two option sets, alternating
#   external/    other engines run the same way (llama.cpp)
#   replay/      multi-turn agent sessions against an OpenAI-compatible server; the speculation controller's choices
#                replayed on recorded rounds (dv_choice)
#   trace/       kernel timelines of unchanged programs (CUPTI), and where a round's time goes
#   tree/        what a round verifying a tree of drafts costs, by width and shape (tree_bench)
add_executable(tree_bench tree/tree_bench.cpp)
target_link_libraries(tree_bench PRIVATE flyx_core)
add_executable(wide_rounds tree/wide_rounds.cpp)  # lookup rounds of more rows replayed on a reply
target_link_libraries(wide_rounds PRIVATE flyx_core)
add_executable(kld quality/kld.cpp)  # next-token distributions against a saved reference
target_link_libraries(kld PRIVATE flyx_core)

# DV's rounds chosen by SpecController, replayed on pinned runs' rounds: core's controller, built with
# its public constants as variables the replay sets (replay/tunable_controller.cmake writes that copy).  Another
# revision's rule: FLYX_DV_CHOICE_SRC a folder holding its core/speculation/controller.{h,cpp}.
set(FLYX_DV_CHOICE_SRC ${PROJECT_SOURCE_DIR} CACHE PATH "the tree whose core/speculation/controller is replayed")
set(dv_choice_dir ${CMAKE_CURRENT_BINARY_DIR}/dv_choice)
add_custom_command(
  OUTPUT ${dv_choice_dir}/core/speculation/controller.h ${dv_choice_dir}/core/speculation/controller.cpp
         ${dv_choice_dir}/dv_choice_knobs.inc
  COMMAND ${CMAKE_COMMAND} -DSRC=${FLYX_DV_CHOICE_SRC} -DOUT=${dv_choice_dir}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/replay/tunable_controller.cmake
  DEPENDS ${FLYX_DV_CHOICE_SRC}/core/speculation/controller.h ${FLYX_DV_CHOICE_SRC}/core/speculation/controller.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/replay/tunable_controller.cmake
  COMMENT "SpecController with its constants as variables, for dv_choice")
add_executable(dv_choice replay/dv_choice.cpp ${dv_choice_dir}/core/speculation/controller.cpp
                         ${dv_choice_dir}/dv_choice_knobs.inc)
target_include_directories(dv_choice BEFORE PRIVATE ${dv_choice_dir})
target_include_directories(dv_choice PRIVATE ${PROJECT_SOURCE_DIR}/include)

if(TARGET flyx_cuda_kernels)
  add_executable(cuda_bench microbench/cuda_bench.cu microbench/mlp_persistent.cu microbench/tune.cu
                            microbench/tune_files.cpp microbench/cold_dot.cpp microbench/graphs.cu microbench/paging.cu)
  # flyx_core: the model file reader, for the matrices `cuda_bench tune FILE...` weighs its rules by
  target_link_libraries(cuda_bench PRIVATE flyx_cuda_host flyx_cuda_kernels flyx_core)
  set_target_properties(cuda_bench PROPERTIES CUDA_ARCHITECTURES "${FLYX_CUDA_ARCHS}")
  add_test(NAME cuda_bench_round_finite COMMAND cuda_bench round IQ3_XXS 1024x5120 4)
  set_property(TEST cuda_bench_round_finite APPEND PROPERTY LABELS tools gpu)

  # CUPTI ships in extras/CUPTI of a toolkit install, next to the toolkit's own
  # headers and libraries in the PyPI one (and only as libcupti.so.13 there).
  find_library(FLYX_CUPTI NAMES cupti libcupti.so.13
               HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/lib64
                     ${CUDAToolkit_TARGET_DIR}/extras/CUPTI/lib64
                     ${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/lib/x64 ${CUDAToolkit_TARGET_DIR}/extras/CUPTI/lib/x64)
  find_path(FLYX_CUPTI_INCLUDE cupti.h
            HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/include
                  ${CUDAToolkit_TARGET_DIR}/extras/CUPTI/include)
  if(FLYX_CUPTI AND FLYX_CUPTI_INCLUDE)
    add_library(flyx-trace MODULE trace/kernel_trace.cpp)
    target_include_directories(flyx-trace PRIVATE ${FLYX_CUPTI_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(flyx-trace PRIVATE ${FLYX_CUPTI})
    if(WIN32)
      # CUDA's demangler understands the device's Itanium names. CUPTI alone still records the original names.
      find_library(FLYX_CUFILT NAMES cufilt HINTS ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES lib/x64)
      find_path(FLYX_CUFILT_INCLUDE nv_decode.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
      if(FLYX_CUFILT AND FLYX_CUFILT_INCLUDE)
        target_link_libraries(flyx-trace PRIVATE ${FLYX_CUFILT})
        target_include_directories(flyx-trace PRIVATE ${FLYX_CUFILT_INCLUDE})
        target_compile_definitions(flyx-trace PRIVATE FLYX_TRACE_CUFILT)
        if(MSVC)
          set_property(TARGET flyx-trace PROPERTY MSVC_RUNTIME_LIBRARY MultiThreaded)  # cufilt.lib uses the release CRT
        endif()
      endif()
    endif()
    set_target_properties(flyx-trace PROPERTIES PREFIX "lib")
  endif()
endif()

flyx_finish_test_labels()  # tests/CMakeLists.txt: the tiers, quick, the GPU's resource group

# On Windows these programs read their files by UTF-8 paths as flyx does (a build tree under a user folder named in
# Chinese): the same manifest (flyx_program).
get_property(flyx_dir_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(t IN LISTS flyx_dir_targets)
  get_target_property(flyx_target_type ${t} TYPE)
  if(flyx_target_type STREQUAL "EXECUTABLE")
    flyx_program(${t})
  endif()
endforeach()
