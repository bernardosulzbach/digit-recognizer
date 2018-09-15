#pragma once

#include <stdexcept>

#include "Clock.hpp"
#include "Duration.hpp"

class Timer {
  Duration elapsed{0};
  Clock clock;
  bool running = false;

 public:
  void restart() {
    elapsed = Duration{0};
    start();
  }
  void start() {
    if (running) {
      throw std::logic_error("Starting a Timer which is already running.");
    }
    running = true;
    clock.restart();
  }
  void stop() {
    if (!running) {
      throw std::logic_error("Stopping a Timer which is already stopped.");
    }
    running = false;
    elapsed += clock.getElapsed();
  }
  Duration getElapsed() const { return elapsed; }
};
