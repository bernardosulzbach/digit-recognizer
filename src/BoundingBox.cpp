#include "BoundingBox.hpp"

#include <stdexcept>

BoundingBox::BoundingBox(int x1, int y1, int x2, int y2) : x1(x1), y1(y1), x2(x2), y2(y2) {
  if (x2 < x1) throw std::domain_error("Not a valid bounding box.");
  if (y2 < y1) throw std::domain_error("Not a valid bounding box.");
}
