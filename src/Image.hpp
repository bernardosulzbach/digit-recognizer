#pragma once

#include "BoundingBox.hpp"

#include <boost/numeric/conversion/cast.hpp>

#include <opencv2/opencv.hpp>

constexpr size_t imageSide = 28;
constexpr size_t imageSize = imageSide * imageSide;

class Image {
 public:
  std::array<uint8_t, imageSize> data{};

  void applyThreshold(uint8_t threshold);

  void dump(const std::string &string) const;

  size_t neighbors(ssize_t i, ssize_t j);

  void removeIslands();

  float linearInterpolate(float s, float e, float t);

  float bilinearInterpolate(float c00, float c10, float c01, float c11, float tx, float ty);

  int getIndex(int x, int y);

  uint8_t get(int x, int y);

  void set(int x, int y, uint8_t value);

  Image bilinearScaleToFitVertically(BoundingBox box);

  inline uint8_t operator[](size_t i) const { return data[i]; }

  inline uint8_t &operator[](size_t i) { return data[i]; }
};
