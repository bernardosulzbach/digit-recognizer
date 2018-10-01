#include "Filesystem.hpp"

#include <boost/filesystem.hpp>

void removeTree(const std::string &path) { boost::filesystem::remove_all(path); }

void ensurePathExists(const std::string &path) {
  if (!boost::filesystem::exists(path)) {
    boost::filesystem::create_directory(path);
  }
}