#pragma once

#include <sstream>
#include <string>

template <typename T>
T stringToIntegral(const std::string &string) {
  std::stringstream ss(string);
  T integer;
  ss >> integer;
  return integer;
}

std::string toString(double value, int digits);

std::string padString(std::string string, size_t digits);