#include <utility>

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

#include <boost/filesystem.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <opencv2/opencv.hpp>

#include "SVM.h"

#include "EdgeCounters.hpp"
#include "Filesystem.hpp"
#include "Image.hpp"
#include "ProcessedImage.hpp"
#include "String.hpp"
#include "Timer.hpp"

constexpr double cacheSize = 1024;

using Label = uint16_t;

const auto separator = boost::filesystem::path::preferred_separator;

class Configuration {
 public:
  std::string trainingFile;
  std::string testingFile;
  uint32_t trainingSamples = 35000;
  uint32_t testingSamples = 7000;
  bool verticallyFittingImages = true;
  uint8_t threshold = 64;
  bool removeIslands = false;
  double svmEpsilon = 0.001;
  bool dumpMistakes = false;
};

std::ostream &operator<<(std::ostream &os, const Configuration &configuration) {
  os << configuration.trainingFile << ",";
  os << configuration.testingFile << ",";
  os << configuration.trainingSamples << ",";
  os << configuration.testingSamples << ",";
  os << configuration.verticallyFittingImages << ",";
  os << static_cast<uint16_t>(configuration.threshold) << ",";
  os << configuration.removeIslands << ",";
  os << configuration.svmEpsilon << ",";
  os << configuration.dumpMistakes;
  return os;
}

std::vector<Image> readImagesFromFile(const std::string &filename, bool labeled) {
  std::vector<Image> labeledImages;
  std::ifstream ifs(filename);
  ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  uint16_t value;
  while (ifs >> value) {
    Image image;
    if (!labeled) image[0] = static_cast<uint8_t>(value);
    for (size_t j = labeled ? 0 : 1; j < imageSize; j++) {
      char comma;
      ifs >> comma >> value;
      image[j] = static_cast<uint8_t>(value);
    }
    labeledImages.emplace_back(image);
  }
  return labeledImages;
}

std::vector<Label> readLabelsFromFile(const std::string &filename) {
  std::vector<Label> labels;
  std::ifstream ifs(filename);
  ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  uint16_t value;
  while (ifs >> value) {
    labels.push_back(static_cast<uint8_t>(value));
    ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return labels;
}

template <typename T>
std::vector<T> getPermutationVector(size_t size) {
  std::vector<T> vector(size);
  std::iota(begin(vector), end(vector), 0);
  std::random_shuffle(begin(vector), end(vector));
  return vector;
}

double test(const Configuration &configuration, const std::vector<Label> &trainingLabels, const std::vector<Image> &trainingImages, const std::vector<Image> &testingImages) {
  const std::string mistakesDirectory = "mistakes";
  const std::string imageExtension = ".png";
  const std::string textFileExtension = ".txt";
  const auto trainingFile = configuration.trainingFile;
  const auto testingFile = configuration.testingFile;
  int n = configuration.trainingSamples;
  int m = configuration.testingSamples;
  std::cout << "Training model...";
  Timer timer;
  timer.start();
  std::vector<ProcessedImage> processedTrainingImages;
  std::vector<ProcessedImage> processedTestingImages;
  for (const auto &image : trainingImages) processedTrainingImages.emplace_back(image);
  for (const auto &image : testingImages) processedTestingImages.emplace_back(image);
  if (configuration.verticallyFittingImages) {
    for (auto &processedImage : processedTrainingImages) {
      processedImage.image = processedImage.image.bilinearScaleToFitVertically(processedImage.counters.getBoundingBox());
    }
    for (auto &processedImage : processedTestingImages) {
      processedImage.image = processedImage.image.bilinearScaleToFitVertically(processedImage.counters.getBoundingBox());
    }
  }
  if (configuration.threshold != 0) {
    for (auto &processedImage : processedTrainingImages) processedImage.image.applyThreshold(configuration.threshold);
    for (auto &processedImage : processedTestingImages) processedImage.image.applyThreshold(configuration.threshold);
  }
  if (configuration.removeIslands) {
    for (auto &processedImage : processedTrainingImages) processedImage.image.removeIslands();
    for (auto &processedImage : processedTestingImages) processedImage.image.removeIslands();
  }
  svm_problem problem{};
  problem.l = n;
  std::vector<double> ys(boost::numeric_cast<unsigned long>(n));
  for (int i = 0; i < n; i++) ys[i] = trainingLabels[i];
  problem.y = ys.data();
  std::vector<std::vector<svm_node>> xs;
  size_t totalNodes = 0;
  for (int i = 0; i < n; i++) {
    const auto nodes = processedTrainingImages[i].svmNodesFromValues();
    xs.push_back(nodes);
    totalNodes += nodes.size();
  }
  std::vector<svm_node *> pointersToXs;
  for (int i = 0; i < n; i++) pointersToXs.push_back(xs[i].data());
  problem.x = pointersToXs.data();
  svm_parameter parameter{};
  parameter.svm_type = C_SVC;
  parameter.kernel_type = LINEAR;
  parameter.cache_size = cacheSize;
  parameter.C = 1.0;
  parameter.eps = configuration.svmEpsilon;
  const auto error_message = svm_check_parameter(&problem, &parameter);
  if (error_message) throw std::runtime_error(error_message);
  std::cout.flush();
  const auto model = svm_train(&problem, &parameter);
  timer.stop();
  std::cout << " took " << timer.getElapsed().toSecondsString() << "." << '\n';
  std::vector<std::vector<uint32_t>> results(10, std::vector<uint32_t>(10));
  std::cout << "Evaluating model...";
  std::cout.flush();
  timer.restart();
  if (configuration.dumpMistakes) {
    removeTree(mistakesDirectory);
    ensurePathExists(mistakesDirectory);
  }
  for (int i = 0; i < m; i++) {
    auto &trainingImage = processedTrainingImages[n + i];
    auto nodes = trainingImage.svmNodesFromValues();
    totalNodes += nodes.size();
    const auto prediction = static_cast<Label>(svm_predict(model, nodes.data()));
    results[trainingLabels[n + i]][prediction]++;
    if (configuration.dumpMistakes) {
      if (prediction != trainingLabels[n + i]) {
        const auto predictionString = std::to_string(prediction);
        const auto fullPath = mistakesDirectory + separator + std::to_string(trainingLabels[n + i]) + "-as-" + predictionString;
        ensurePathExists(fullPath);
        trainingImage.image.dump(fullPath + separator + std::to_string(n + i) + imageExtension);
        std::ofstream rayCounters(fullPath + separator + std::to_string(n + i) + textFileExtension);
        for (auto svm_node : nodes) rayCounters << svm_node.index << ": " << svm_node.value << '\n';
      }
    }
  }
  timer.stop();
  std::cout << " took " << timer.getElapsed().toSecondsString() << "." << '\n';
  std::cout << "On average, there are " << toString(totalNodes / static_cast<double>(n + m), 3) << " nodes per image." << '\n';
  class MistakeClass {
   public:
    Label correct;
    Label answer;
    uint32_t count;

    MistakeClass(Label correct, Label answer, uint32_t count) : correct(correct), answer(answer), count(count) {}

    bool operator<(MistakeClass &rhs) const {
      if (count < rhs.count) return true;
      if (count == rhs.count && correct < rhs.correct) return true;
      return count == rhs.count && correct == rhs.correct && answer < rhs.answer;
    }
  };
  size_t right{};
  size_t wrong{};
  std::vector<MistakeClass> mistakeClasses;
  for (int r = 0; r < 10; r++) {
    for (int i = 0; i < 10; i++) {
      if (i == r) {
        right += results[r][i];
      } else {
        wrong += results[r][i];
        mistakeClasses.emplace_back(r, i, results[r][i]);
      }
    }
  }
  if (configuration.dumpMistakes) {
    std::sort(rbegin(mistakeClasses), rend(mistakeClasses));
    std::ofstream mistakesFile(mistakesDirectory + separator + "summary.txt");
    for (const auto mistakeClass : mistakeClasses) {
      mistakesFile << padString(std::to_string(mistakeClass.count), 5) << " | " << mistakeClass.answer << " >> " << mistakeClass.correct << "\n";
    }
  }
  std::cout << "Got " << right << " of " << (right + wrong) << "." << ' ';
  const auto accuracy = right / (double)(right + wrong);
  std::cout << "Accuracy is " << accuracy << "." << '\n';
  return accuracy;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cout << "Usage: " << argv[0] << " [TRAINING FILE] [TESTING FILE] [TRAINING SET SIZE] [TESTING SET SIZE]" << '\n';
    return 1;
  }
  std::string trainingFile = argv[1];
  std::string testingFile = argv[2];
  const auto trainingSamples = stringToIntegral<uint32_t>(argv[3]);
  const auto testingSamples = stringToIntegral<uint32_t>(argv[4]);
  Timer timer;
  timer.start();
  std::cout << "Reading images...";
  std::cout.flush();
  const auto trainingLabels = readLabelsFromFile(trainingFile);
  std::vector<Image> trainingImages = readImagesFromFile(trainingFile, true);
  uint32_t requiredSamples = trainingSamples + testingSamples;
  if (static_cast<unsigned>(requiredSamples) > trainingImages.size()) {
    const auto requiredSamplesString = std::to_string(requiredSamples);
    const auto has = std::to_string(trainingImages.size());
    throw std::runtime_error("Not enough training images. Required " + requiredSamplesString + ", but has only " + has + ".");
  }
  std::vector<Image> testingImages;
  if (!testingFile.empty()) readImagesFromFile(testingFile, false);
  timer.stop();
  std::cout << " took " << timer.getElapsed().toSecondsString() << "." << '\n';
  std::vector<uint8_t> thresholds{1u, 5u, 15u, 45u, 135u, 250u};
  for (bool verticallyFittingImages : {false, true}) {
    for (const auto threshold : thresholds) {
      for (bool removeIslands : {false, true}) {
        for (double svmEpsilon : {1e-3, 3e-3, 9e-3, 27e-3}) {
          Configuration configuration;
          configuration.trainingFile = trainingFile;
          configuration.testingFile = testingFile;
          configuration.trainingSamples = trainingSamples;
          configuration.testingSamples = testingSamples;
          configuration.verticallyFittingImages = verticallyFittingImages;
          configuration.threshold = threshold;
          configuration.removeIslands = removeIslands;
          configuration.svmEpsilon = svmEpsilon;
          configuration.dumpMistakes = false;
          timer.restart();
          const auto accuracy = test(configuration, trainingLabels, trainingImages, testingImages);
          timer.stop();
          std::ofstream output("results.txt", std::ios_base::app);
          output << configuration << ',' << timer.getElapsed().getNanoseconds() << ',' << accuracy << '\n';
        }
      }
    }
  }
  return 0;
}
