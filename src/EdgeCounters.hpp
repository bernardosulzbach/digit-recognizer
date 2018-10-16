#pragma once

#include "BoundingBox.hpp"
#include "Image.hpp"

#include <boost/numeric/conversion/cast.hpp>

#include <cstdio>

class EdgeCounters {
 public:
  std::vector<size_t> row;
  std::vector<size_t> column;

  size_t firstRow{};
  size_t firstColumn{};
  size_t lastRow{};
  size_t lastColumn{};

  explicit EdgeCounters(const Image &image);

  inline BoundingBox getBoundingBox() const {
    const auto x1 = boost::numeric_cast<int>(firstColumn);
    const auto y1 = boost::numeric_cast<int>(firstRow);
    const auto x2 = boost::numeric_cast<int>(lastColumn);
    const auto y2 = boost::numeric_cast<int>(lastRow);
    return BoundingBox{x1, y1, x2, y2};
  }

  inline size_t rowSpan() const { return lastRow - firstRow; }

  inline size_t columnSpan() const { return lastColumn - firstColumn; }
};