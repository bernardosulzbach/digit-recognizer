#pragma once

#include "String.hpp"
#include "Types.hpp"

typedef double Seconds;

class Duration {
  U64 nanoseconds;

 public:
  inline explicit Duration(U64 nanoseconds) : nanoseconds(nanoseconds) {}
  inline U64 getNanoseconds() const { return nanoseconds; }
  inline Duration operator+=(const Duration &rhs) {
    nanoseconds += rhs.nanoseconds;
    return *this;
  }
  inline bool operator<(const Duration &rhs) const { return nanoseconds < rhs.nanoseconds; }
  inline bool operator>(const Duration &rhs) const { return rhs < *this; }
  inline bool operator<=(const Duration &rhs) const { return !(rhs < *this); }
  inline bool operator>=(const Duration &rhs) const { return !(*this < rhs); }
  inline bool operator==(const Duration &rhs) const { return nanoseconds == rhs.nanoseconds; }
  inline Seconds toSeconds() const { return nanoseconds / 1000.0 / 1000.0 / 1000.0; }
  inline std::string toSecondsString() const { return toString(toSeconds(), 3) + " seconds"; }
  static Duration fromSeconds(Seconds seconds) { return Duration{static_cast<U64>(1000.0 * 1000.0 * 1000.0 * seconds)}; }
};
