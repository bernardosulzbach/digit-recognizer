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

#include "svm.h"

inline int stringToInteger(const std::string &string) {
  std::stringstream ss(string);
  int integer;
  ss >> integer;
  return integer;
}

constexpr double svmEps = 0.001;

constexpr double cacheSize = 1024;

constexpr size_t imageSide = 28;
constexpr size_t imageSize = imageSide * imageSide;
constexpr size_t threshold = 1;

using Label = uint8_t;
using Image = std::array<uint8_t, imageSize>;

std::string padString(std::string string, size_t digits) {
  if (string.size() >= digits) return string;
  std::string result;
  size_t required = digits - string.size();
  for (size_t i = 0; i < required; i++) result += ' ';
  result += string;
  return result;
}

std::array<svm_node, imageSize + 1> nodesFromImage(const Image &image) {
  std::array<svm_node, imageSize + 1> array{};
  for (size_t j = 0; j < imageSize; j++) {
    array[j].index = static_cast<int>(j + 1);
    array[j].value = image[j];
  }
  array[imageSize].index = -1;
  array[imageSize].value = 0;
  return array;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " [TRAIN] [N] (TEST)" << '\n';
  }
  std::string trainingFile = argv[1];
  int n = stringToInteger(argv[2]);
  std::string testFile;
  if (argc >= 4) testFile = argv[3];
  std::vector<Label> labels(n);
  std::vector<Image> images(n);
  {
    std::ifstream ifs(trainingFile);
    ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    for (int i = 0; i < n; i++) {
      uint16_t value;
      ifs >> value;
      labels[i] = static_cast<uint8_t>(value);
      for (size_t j = 0; j < imageSize; j++) {
        char comma;
        ifs >> comma >> value;
        images[i][j] = static_cast<uint8_t>(value);
      }
    }
  }
  for (int i = 0; i < n; i++) {
    for (auto &image : images) {
      for (auto &pixel : image) {
        if (pixel < threshold) {
          pixel = 0;
        } else {
          pixel = 1;
        }
      }
    }
  }
  svm_problem problem{};
  problem.l = n;
  std::vector<double> ys(n);
  for (int i = 0; i < n; i++) {
    ys[i] = labels[i];
  }
  problem.y = ys.data();
  // Naive.
  std::vector<std::array<svm_node, imageSize + 1>> xs(n);
  for (int i = 0; i < n; i++) {
    xs[i] = nodesFromImage(images[i]);
  }
  std::vector<svm_node *> xPointers;
  for (int i = 0; i < n; i++) {
    xPointers.push_back(xs[i].data());
  }
  problem.x = xPointers.data();
  svm_parameter parameter{};
  parameter.svm_type = C_SVC;
  parameter.kernel_type = LINEAR;
  parameter.cache_size = cacheSize;
  parameter.C = 1.0;
  parameter.eps = svmEps;
  const auto error_message = svm_check_parameter(&problem, &parameter);
  if (error_message) {
    throw std::runtime_error(error_message);
  }
  const auto model = svm_train(&problem, &parameter);
  std::vector<std::vector<uint32_t>> results(10, std::vector<uint32_t>(10));
  svm_train(&problem, &parameter);
  {
    std::ifstream ifs(trainingFile);
    ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    while (true) {
      uint16_t value;
      if (ifs >> value) {
        uint8_t label = static_cast<uint8_t>(value);
        Image image{};
        for (size_t j = 0; j < imageSize; j++) {
          char comma;
          ifs >> comma >> value;
          if (value < threshold) {
            value = 0;
          } else {
            value = 1;
          }
          image[j] = value;
        }
        auto nodes = nodesFromImage(image);
        const auto prediction = svm_predict(model, nodes.data());
        // std::cout << static_cast<uint32_t>(label) << ' ' << prediction << '\n';
        results[label][prediction]++;
      } else {
        break;
      }
    }
  }
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
  std::cout << "Got " << right << " of " << (right + wrong) << "." << '\n';
  std::cout << "Rate is " << right / (double)(right + wrong) << "." << '\n';
  return 0;
}
