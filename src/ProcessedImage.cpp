#include "ProcessedImage.hpp"

ProcessedImage::ProcessedImage(const Image &source) : image(source), counters(image) {}

std::vector<svm_node> ProcessedImage::svmNodesFromValues() {
  std::vector<svm_node> nodes;
  for (size_t i = 0; i < imageSize; i++) {
    if (image[i] != 0) nodes.emplace_back(static_cast<int>(1 + i), image[i]);
  }
  nodes.emplace_back(-1, 0);
  return nodes;
}

std::vector<svm_node> ProcessedImage::svmNodesFromEdgeCounters() {
  std::vector<svm_node> nodes;
  for (size_t i = 0; i < imageSide; i++) {
    if (counters.row[i] != 0) nodes.emplace_back(static_cast<int>(1 + i), counters.row[i]);
  }
  for (size_t j = 0; j < imageSide; j++) {
    if (counters.column[j] != 0) nodes.emplace_back(static_cast<int>(1 + imageSide + j), counters.row[j]);
  }
  nodes.emplace_back(-1, 0);
  return nodes;
}
