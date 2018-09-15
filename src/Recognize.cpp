#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>

#include "SVM.h"

#include "String.hpp"
#include "Timer.hpp"

constexpr double svmEps = 0.001;

constexpr double cacheSize = 1024;

constexpr size_t imageSide = 28;
constexpr size_t imageSize = imageSide * imageSide;
constexpr size_t threshold = 1;

using Label = uint8_t;

class Image {
 public:
  std::array<uint8_t, imageSize> data{};
  void applyThreshold() {
    for (auto &pixel : data) {
      if (pixel < threshold) {
        pixel = 0;
      } else {
        pixel = 1;
      }
    }
  }
  uint8_t operator[](size_t i) const { return data[i]; }
  uint8_t &operator[](size_t i) { return data[i]; }
};

class LabeledImage {
 public:
  std::optional<Label> label;
  Image image;

  LabeledImage(std::optional<Label> label, const Image &image) : label(label), image(image) {}
};

std::string padString(std::string string, size_t digits) {
  if (string.size() >= digits) return string;
  std::string result;
  size_t required = digits - string.size();
  for (size_t i = 0; i < required; i++) result += ' ';
  result += string;
  return result;
}

std::vector<svm_node> simpleNodesFromImage(const Image &image) {
  std::vector<svm_node> nodes;
  for (size_t i = 0; i < imageSize; i++) {
    if (image[i] != 0) {
      nodes.push_back(svm_node{});
      nodes.back().index = static_cast<int>(i + 1);
      nodes.back().value = image[i];
    }
  }
  nodes.push_back(svm_node{});
  nodes.back().index = -1;
  nodes.back().value = 0;
  return nodes;
}

std::vector<svm_node> edgeCountersFromImage(const Image &image) {
  std::vector<size_t> edges(2 * imageSide);
  for (size_t i = 1; i < imageSide; i++) {
    for (size_t j = 1; j < imageSide; j++) {
      if (image[i * imageSide + j] != image[(i - 1) * imageSide + j]) edges[i]++;
      if (image[i * imageSide + j] != image[i * imageSide + (j - 1)]) edges[imageSide + j]++;
    }
  }
  std::vector<svm_node> nodes;
  for (size_t i = 0; i < 2 * imageSide; i++) {
    if (edges[i] != 0) {
      nodes.push_back(svm_node{});
      nodes.back().index = static_cast<int>(i + 1);
      nodes.back().value = edges[i];
    }
  }
  nodes.push_back(svm_node{});
  nodes.back().index = -1;
  nodes.back().value = 0;
  return nodes;
}

std::vector<LabeledImage> readImagesFromFile(const std::string &filename, bool labeled) {
  std::vector<LabeledImage> labeledImages;
  std::ifstream ifs(filename);
  ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  uint16_t value;
  while (ifs >> value) {
    std::optional<Label> optionalLabel;
    if (labeled) optionalLabel = static_cast<Label>(value);
    Image image;
    if (!labeled) image[0] = static_cast<uint8_t>(value);
    for (size_t j = labeled ? 0 : 1; j < imageSize; j++) {
      char comma;
      ifs >> comma >> value;
      image[j] = static_cast<uint8_t>(value);
    }
    labeledImages.emplace_back(optionalLabel, image);
  }
  return labeledImages;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " [TRAINING FILE] [N] [M] (TESTING FILE)" << '\n';
    return 1;
  }
  std::string trainingFile = argv[1];
  int n = stringToInteger(argv[2]);
  int m = stringToInteger(argv[3]);
  std::string testingFile;
  if (argc >= 5) testingFile = argv[4];
  Timer timer;
  timer.start();
  std::cout << "Reading images...";
  std::cout.flush();
  std::vector<LabeledImage> trainingImages = readImagesFromFile(trainingFile, true);
  if (static_cast<unsigned>(n + m) > trainingImages.size()) throw std::runtime_error("Not enough training images.");
  std::vector<LabeledImage> testingImages;
  if (!testingFile.empty()) readImagesFromFile(testingFile, false);
  for (auto &labeledImage : trainingImages) labeledImage.image.applyThreshold();
  for (auto &labeledImage : testingImages) labeledImage.image.applyThreshold();
  timer.stop();
  std::cout << " took " << timer.getElapsed().toSecondsString() << "." << '\n';
  svm_problem problem{};
  problem.l = n;
  std::vector<double> ys(n);
  for (int i = 0; i < n; i++) ys[i] = trainingImages[i].label.value();
  problem.y = ys.data();
  std::vector<std::vector<svm_node>> xs;
  for (int i = 0; i < n; i++) xs.push_back(edgeCountersFromImage(trainingImages[i].image));
  std::vector<svm_node *> pointersToXs;
  for (int i = 0; i < n; i++) pointersToXs.push_back(xs[i].data());
  problem.x = pointersToXs.data();
  svm_parameter parameter{};
  parameter.svm_type = C_SVC;
  parameter.kernel_type = LINEAR;
  parameter.cache_size = cacheSize;
  parameter.C = 1.0;
  parameter.eps = svmEps;
  const auto error_message = svm_check_parameter(&problem, &parameter);
  if (error_message) throw std::runtime_error(error_message);
  std::cout << "Training model...";
  std::cout.flush();
  timer.restart();
  const auto model = svm_train(&problem, &parameter);
  timer.stop();
  std::cout << " took " << timer.getElapsed().toSecondsString() << "." << '\n';
  std::vector<std::vector<uint32_t>> results(10, std::vector<uint32_t>(10));
  std::cout << "Evaluating model...";
  std::cout.flush();
  timer.restart();
  for (int i = 0; i < m; i++) {
    auto nodes = edgeCountersFromImage(trainingImages[n + i].image);
    const auto prediction = svm_predict(model, nodes.data());
    results[trainingImages[n + i].label.value()][prediction]++;
  }
  timer.stop();
  std::cout << " took " << timer.getElapsed().toSecondsString() << "." << '\n';
  size_t right = 0;
  size_t wrong = 0;
  for (size_t r = 0; r < 10; r++) {
    const auto &row = results[r];
    std::vector<std::pair<int, int>> predictions;
    for (size_t i = 0; i < 10; i++) {
      predictions.emplace_back(row[i], i);
      if (i == r) {
        right += row[i];
      } else {
        wrong += row[i];
      }
    }
    std::sort(rbegin(predictions), rend(predictions));
    for (int i = 0; i < 10; i++) {
      if (predictions[i].first) {
        std::cout << r << " >> " << predictions[i].second << ": " << padString(std::to_string(predictions[i].first), 10) << "\n";
      }
    }
  }
  std::cout << "Got " << right << " of " << (right + wrong) << "." << ' ';
  std::cout << "Rate is " << right / (double)(right + wrong) << "." << '\n';
  return 0;
}
