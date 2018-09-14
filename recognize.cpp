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

std::array<svm_node, imageSize + 1> naiveNodesFromImage(const Image &image) {
  std::array<svm_node, imageSize + 1> array{};
  for (size_t j = 0; j < imageSize; j++) {
    array[j].index = static_cast<int>(j + 1);
    array[j].value = image[j];
  }
  array[imageSize].index = -1;
  array[imageSize].value = 0;
  return array;
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
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " [TRAINING FILE] [N] (TESTING FILE)" << '\n';
  }
  std::string trainingFile = argv[1];
  int n = stringToInteger(argv[2]);
  std::string testingFile;
  if (argc >= 4) testingFile = argv[3];
  std::vector<LabeledImage> trainingImages = readImagesFromFile(trainingFile, true);
  std::vector<LabeledImage> testingImages;
  if (!testingFile.empty()) readImagesFromFile(testingFile, false);
  for (auto &labeledImage : trainingImages) labeledImage.image.applyThreshold();
  for (auto &labeledImage : testingImages) labeledImage.image.applyThreshold();
  svm_problem problem{};
  problem.l = n;
  std::vector<double> ys(n);
  for (int i = 0; i < n; i++) ys[i] = trainingImages[i].label.value();
  problem.y = ys.data();
  // Naive.
  std::vector<std::array<svm_node, imageSize + 1>> xs(n);
  for (auto &labeledImage : trainingImages) xs.push_back(naiveNodesFromImage(labeledImage.image));
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
        auto nodes = naiveNodesFromImage(image);
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
