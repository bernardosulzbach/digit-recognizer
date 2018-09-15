#pragma once

#include <iomanip>
#include <sstream>
#include <string>

inline int stringToInteger(const std::string &string) {
  std::stringstream ss(string);
  int integer;
  ss >> integer;
  return integer;
}

inline std::string toString(double value, int digits) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(digits) << value;
  return ss.str();
}