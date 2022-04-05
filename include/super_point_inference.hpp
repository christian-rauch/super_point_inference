#pragma once
#include <tuple>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

struct SuperPointImpl;

class SuperPoint
{
public:
  SuperPoint(const std::string &model_path);

  ~SuperPoint();

  /**
   * @brief getFeatures
   * @param image W x H x 3 RGB colour image
   * @return tuple of
   *          (1) N x 2 keypoint coordinates (x,y)
   *          (2) N x D keypoint decriptor
   */
  std::tuple<Eigen::MatrixX2d, Eigen::MatrixXd> getFeatures(const cv::Mat &image) const;

  void setScoreMin(float min_score);

private:
  SuperPointImpl *impl;
  const bool valid;
  float point_min_score;
};
