#pragma once

#include <cstdint>
#include <limits>
#include <vector>

using UnionFindInteger = uint64_t;

class UnionFind {
 public:
  const UnionFindInteger noComponent = std::numeric_limits<UnionFindInteger>::max();
  std::vector<UnionFindInteger> component;

  explicit UnionFind(UnionFindInteger size);

  bool is_connected(UnionFindInteger a, UnionFindInteger b) const;

  UnionFindInteger getComponent(UnionFindInteger a) const;

  void connect(UnionFindInteger a, UnionFindInteger b);
};
