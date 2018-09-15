#pragma once

#include <chrono>

#include "Duration.hpp"

class Clock {
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

 public:
  inline void restart() { start = std::chrono::steady_clock::now(); }
  inline Duration getElapsed() const {
    const auto duration = std::chrono::steady_clock::now() - start;
    const auto delta = std::chrono::duration_cast<std::chrono::duration<U64, std::nano>>(duration);
    return Duration{delta.count()};
  }
};
