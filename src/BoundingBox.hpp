#pragma once

class BoundingBox {
 public:
  int x1{};
  int y1{};
  int x2{};
  int y2{};

  BoundingBox() = default;
  BoundingBox(int x1, int y1, int x2, int y2);
};
