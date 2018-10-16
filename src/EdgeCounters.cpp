#include "EdgeCounters.hpp"

EdgeCounters::EdgeCounters(const Image &image) : row(imageSide), column(imageSide) {
  for (size_t i = 1; i < imageSide; i++) {
    for (size_t j = 1; j < imageSide; j++) {
      if (image[i * imageSide + j] != image[i * imageSide + (j - 1)]) row[i]++;
      if (image[i * imageSide + j] != image[(i - 1) * imageSide + j]) column[j]++;
    }
  }
  for (size_t i = 0; i < imageSide; i++) {
    if (row[i] > 0) {
      firstRow = i;
      break;
    }
  }
  for (size_t i = 0; i < imageSide; i++) {
    if (column[i] > 0) {
      firstColumn = i;
      break;
    }
  }
  for (size_t i = 0; i < imageSide; i++) {
    if (row[i] > 0) lastRow = i + 1;
  }
  for (size_t i = 0; i < imageSide; i++) {
    if (column[i] > 0) lastColumn = i + 1;
  }
}
