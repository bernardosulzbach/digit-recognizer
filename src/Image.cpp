#include "Image.hpp"

#include "UnionFind.hpp"

void Image::applyThreshold(uint8_t threshold) {
  if (threshold == 0) throw std::runtime_error("Cannot have threshold of 0.");
  for (auto &pixel : data) {
    if (pixel < threshold) {
      pixel = 0;
    } else {
      pixel = 1;
    }
  }
}

void Image::dump(const std::string &string) const {
  cv::Mat matrix(imageSide, imageSide, CV_8U);
  for (size_t i = 0; i < imageSide; i++) {
    for (size_t j = 0; j < imageSide; j++) {
      matrix.at<uint8_t>(i, j) = boost::numeric_cast<uint8_t>(data[i * imageSide + j]);
    }
  }
  imwrite(string, matrix);
}

size_t Image::neighbors(ssize_t i, ssize_t j) {
  size_t counter = 0;
  for (ssize_t a = i - 1; a <= i + 1; a++) {
    for (ssize_t b = j - 1; b <= j + 1; b++) {
      if (a >= 0 && a < static_cast<ssize_t>(imageSide)) {
        if (b >= 0 && b < static_cast<ssize_t>(imageSide)) {
          if (a != i || b != j) {
            if (data[a * imageSide + b] > 0) {
              counter++;
            }
          }
        }
      }
    }
  }
  return counter;
}

void Image::assertIsDiscrete() const {
  for (size_t i = 0; i < imageSize; i++) {
    if (data[i] != 0 && data[i] != 1) throw std::runtime_error("Image should only have 0 and 1.");
  }
}

void Image::removeIslands() {
  assertIsDiscrete();
  UnionFind unionFind(imageSide * imageSide);
  for (size_t i = 0; i < imageSide; i++) {
    for (size_t j = 0; j < imageSide; j++) {
      size_t x1 = (i + 0) * imageSide + (j + 0);
      if (data[x1]) {
        if (i + 1 < imageSide) {
          size_t x2 = (i + 1) * imageSide + (j + 0);
          if (data[x2] == 1) unionFind.connect(x1, x2);
        }
        if (j + 1 < imageSide) {
          size_t x2 = (i + 0) * imageSide + (j + 1);
          if (data[x2] == 1) unionFind.connect(x1, x2);
        }
      }
    }
  }
  std::vector<uint32_t> componentSize(imageSize);
  for (size_t i = 0; i < imageSize; i++) componentSize.at(unionFind.getComponent(i))++;
  const auto maximumSize = std::max_element(begin(componentSize), end(componentSize));
  const auto distance = std::distance(begin(componentSize), maximumSize);
  const auto largestComponent = boost::numeric_cast<UnionFindInteger>(distance);
  for (size_t i = 0; i < imageSize; i++) {
    if (unionFind.getComponent(i) != largestComponent) data[i] = 0;
  }
}

float Image::linearInterpolate(float s, float e, float t) { return s + (e - s) * t; }

float Image::bilinearInterpolate(float c00, float c10, float c01, float c11, float tx, float ty) { return linearInterpolate(linearInterpolate(c00, c10, tx), linearInterpolate(c01, c11, tx), ty); }

int Image::getIndex(int x, int y) {
  const auto imageSideAsInt = static_cast<int>(imageSide);
  const auto index = y * imageSideAsInt + x;
  if (index >= imageSideAsInt * imageSideAsInt) return -1;
  return index;
}

uint8_t Image::get(int x, int y) {
  const auto index = getIndex(x, y);
  if (index == -1) return 0;
  return data[index];
}

void Image::set(int x, int y, uint8_t value) {
  const auto index = getIndex(x, y);
  if (index == -1) return;
  data[getIndex(x, y)] = value;
}

Image Image::bilinearScaleToFitVertically(BoundingBox box) {
  const auto x1 = box.x1;
  const auto y1 = box.y1;
  const auto x2 = box.x2;
  const auto y2 = box.y2;
  if (x1 >= x2) throw std::runtime_error("Invalid scaling factor.");
  if (y1 >= y2) throw std::runtime_error("Invalid scaling factor.");
  const auto imageSideAsInt = static_cast<int>(imageSide);
  const auto scaleFactor = static_cast<float>(y2 - y1) / imageSide;
  Image result;
  for (int x = 0; x < imageSideAsInt; x++) {
    for (int y = 0; y < imageSideAsInt; y++) {
      const auto gx = x1 + x * scaleFactor;
      const auto gy = y1 + y * scaleFactor;
      const auto gxi = static_cast<int>(gx);
      const auto gyi = static_cast<int>(gy);
      const auto c00 = get(gxi + 0, gyi + 0);
      const auto c10 = get(gxi + 1, gyi + 0);
      const auto c01 = get(gxi + 0, gyi + 1);
      const auto c11 = get(gxi + 1, gyi + 1);
      result.set(x, y, static_cast<unsigned char>(bilinearInterpolate(c00, c10, c01, c11, gx - gxi, gy - gyi)));
    }
  }
  return result;
}
