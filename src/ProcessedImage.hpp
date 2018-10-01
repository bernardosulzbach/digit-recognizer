#pragma once

#include <algorithm>
#include <array>
#include <vector>

#include "EdgeCounters.hpp"

#include "SVM.h"

class ProcessedImage {
 public:
  Image image;
  EdgeCounters counters;

  explicit ProcessedImage(const Image &source);

  std::vector<svm_node> svmNodesFromValues();

  std::vector<svm_node> svmNodesFromEdgeCounters();
};