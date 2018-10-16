cmake_minimum_required(VERSION 3.10)
project(digit-recognizer)

# Just add our own flags if using GCC or Clang.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunreachable-code -Wreturn-type -Wall -Wextra -Wpedantic -Werror")
endif ()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

find_package(OpenCV 3 REQUIRED)
find_package(Boost 1.66 REQUIRED COMPONENTS filesystem system)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

set(SOURCES src/ProcessedImage.cpp src/ProcessedImage.hpp src/Types.hpp src/Duration.hpp src/Clock.hpp src/Timer.cpp src/Timer.hpp src/SVM.cpp src/SVM.h src/String.cpp src/String.hpp src/Filesystem.cpp src/Filesystem.hpp src/Image.cpp src/Image.hpp src/EdgeCounters.cpp src/EdgeCounters.hpp src/BoundingBox.cpp src/BoundingBox.hpp src/UnionFind.cpp src/UnionFind.hpp)
add_executable(recognize src/Recognize.cpp ${SOURCES})
target_link_libraries(recognize ${OpenCV_LIBS})
target_link_libraries(recognize ${Boost_LIBRARIES})
