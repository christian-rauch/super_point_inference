#pragma once
#include <feature_matches_interface.hpp>

struct SuperPointImpl;

class SuperPoint : public FeatureMatchesInterface
{
public:
  SuperPoint(const std::string &model_path);

  ~SuperPoint();

  std::tuple<cv::Mat, Eigen::MatrixX2d, Eigen::MatrixXd> getFeatures(const cv::Mat &image) const;

private:
  SuperPointImpl *impl;
  const bool valid;
};
