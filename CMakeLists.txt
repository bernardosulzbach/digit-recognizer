cmake_minimum_required(VERSION 3.10)
project(digit-recognizer)

# Just add our own flags if using GCC or Clang.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunreachable-code -Wreturn-type -Wall -Wextra -Wpedantic -Werror")
endif ()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(SOURCES svm.cpp svm.h)

add_executable(recognize recognize.cpp ${SOURCES})
