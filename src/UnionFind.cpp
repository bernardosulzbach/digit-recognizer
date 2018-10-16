#include "UnionFind.hpp"

UnionFind::UnionFind(const UnionFindInteger size) { component.resize(size, noComponent); }

bool UnionFind::is_connected(UnionFindInteger a, UnionFindInteger b) const { return getComponent(a) == getComponent(b); }

UnionFindInteger UnionFind::getComponent(UnionFindInteger a) const {
  while (component[a] != noComponent) {
    a = component[a];
  }
  return a;
}

void UnionFind::connect(UnionFindInteger a, UnionFindInteger b) {
  if (!is_connected(a, b)) {
    component[getComponent(a)] = getComponent(b);
  }
}
