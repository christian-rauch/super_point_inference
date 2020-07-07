#pragma once
#include <feature_matches_interface.hpp>

struct SuperPixelImpl;

class SuperPixel : public FeatureMatchesInterface
{
public:
  SuperPixel(const std::string &model_path);

  ~SuperPixel();

  std::tuple<cv::Mat, Eigen::MatrixX2d, Eigen::MatrixXd> getFeatures(const cv::Mat &image) const;

private:
  SuperPixelImpl *impl;
  const bool valid;
};
