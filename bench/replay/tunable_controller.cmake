# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
# core/speculation/controller.{h,cpp} with SpecController's public constants as variables, for bench/replay/dv_choice:
# the same code, constants a replay may set (the rule's tuning), and the list of them (dv_choice_knobs.inc,
# FLYX_KNOB(type, name) a line).
#
#   cmake -DSRC=<repository> -DOUT=<dir> -P tunable_controller.cmake
file(READ "${SRC}/core/speculation/controller.h" header)
string(FIND "${header}" "class SpecController {" begin)
string(FIND "${header}" "  void reset(" end)
if(begin LESS 0 OR end LESS begin)
  message(FATAL_ERROR "tunable_controller: SpecController's constants (before reset()) not found in controller.h")
endif()
math(EXPR length "${end} - ${begin}")
string(SUBSTRING "${header}" 0 ${begin} before)
string(SUBSTRING "${header}" ${begin} ${length} constants)
string(SUBSTRING "${header}" ${end} -1 after)
# Each `static constexpr int|double kA = 1, kB = 2;` (the ';' left out: a CMake list's separator).
string(REGEX MATCHALL "static constexpr (int|double) [^;]*" declarations "${constants}")
set(knobs "")
foreach(declaration IN LISTS declarations)
  string(REGEX MATCH "^static constexpr (int|double) (.*)$" parts "${declaration}")
  set(type "${CMAKE_MATCH_1}")
  string(REPLACE "," ";" names "${CMAKE_MATCH_2}")
  foreach(name IN LISTS names)
    string(REGEX MATCH "k[A-Za-z0-9_]+" name "${name}")
    string(APPEND knobs "FLYX_KNOB(${type}, ${name})\n")
  endforeach()
endforeach()
if(knobs STREQUAL "")
  message(FATAL_ERROR "tunable_controller: no SpecController constants found")
endif()
string(REPLACE "static constexpr int " "static inline int " constants "${constants}")
string(REPLACE "static constexpr double " "static inline double " constants "${constants}")
file(WRITE "${OUT}/core/speculation/controller.h" "${before}${constants}${after}")
file(WRITE "${OUT}/dv_choice_knobs.inc" "${knobs}")
configure_file("${SRC}/core/speculation/controller.cpp" "${OUT}/core/speculation/controller.cpp" COPYONLY)
