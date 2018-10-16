#include "String.hpp"

#include <iomanip>

std::string toString(double value, int digits) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(digits) << value;
  return ss.str();
}

std::string padString(std::string string, size_t digits) {
  if (string.size() >= digits) return string;
  std::string result;
  size_t required = digits - string.size();
  for (size_t i = 0; i < required; i++) result += ' ';
  result += string;
  return result;
}
