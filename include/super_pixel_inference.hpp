#pragma once
#include <feature_matches_interface.hpp>

class SuperPixel : public FeatureMatchesInterface {
public:
  SuperPixel();

  std::tuple<cv::Mat, std::vector<KeyPoint>> &getScore(const cv::Mat &colour, const cv::Mat &depth);
};
