#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <super_point_inference.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


std::vector<std::tuple<int, int, float>>
pairwise_matches(const Eigen::MatrixXf &last_keypoints,
                 const Eigen::MatrixXf &next_keypoints,
                 const cv::Mat_<bool> &last_mask = {})
{
  // remove 'last' keypoint coordinates outside of mask
  Eigen::MatrixXf last_keypoints_mask = Eigen::MatrixXf::Constant(last_keypoints.rows(), last_keypoints.cols(), std::numeric_limits<float>::signaling_NaN());
  std::vector<int> last_mask_id;
  if (!last_mask.empty()) {
    int kp_matches = 0;
    for(int i=0; i<last_keypoints.rows(); i++) {
        const Eigen::Array2f xy_norm = last_keypoints.leftCols(2).row(i);
        const cv::Point2i xy(xy_norm.x()*last_mask.cols, xy_norm.y()*last_mask.rows);
        if (last_mask.at<bool>(xy)) {
          // copy match over
          last_keypoints_mask.row(kp_matches) = last_keypoints.row(i);
          kp_matches++;
          // store original ID of last keypoint within valid segment
          last_mask_id.push_back(i);
        }
    }
    last_keypoints_mask.conservativeResize(kp_matches, Eigen::NoChange);
  }
  else {
    // use all keypoints
    last_keypoints_mask = last_keypoints;
  }

  // store correspondences (last_id, next_id)
  std::vector<std::tuple<int, int, float>> match_ids;

  if (last_keypoints_mask.rows()>0) {
    cv::Mat last_descr;
    cv::eigen2cv(Eigen::MatrixXf(last_keypoints_mask.rightCols(last_keypoints_mask.cols()-2)), last_descr);
    cv::Mat next_descr;
    cv::eigen2cv(Eigen::MatrixXf(next_keypoints.rightCols(next_keypoints.cols()-2)), next_descr);

    std::vector<cv::DMatch> matches;
    cv::BFMatcher(cv::NORM_L2, true).match(next_descr, last_descr, matches);

    for(const cv::DMatch &match : matches) {
      if (match.distance>0.7)
        continue;
      match_ids.push_back(std::make_tuple(last_mask.empty() ? match.trainIdx : last_mask_id[match.trainIdx],
                                          match.queryIdx,
                                          match.distance));
    }
  }

  return match_ids;
}

cv::Mat
draw_matches(const Eigen::MatrixXf &last_keypoints,
             const Eigen::MatrixXf &next_keypoints,
             const std::vector<std::tuple<int, int, float>> &correspondences,
             const cv::Mat &last_img, const cv::Mat &next_img = {})
{
  std::vector<cv::DMatch> matches;
  for(const auto &match : correspondences)
    matches.emplace_back(std::get<1>(match), std::get<0>(match), std::get<2>(match));

  std::vector<cv::KeyPoint> last_kp(last_keypoints.rows());
  for(size_t i=0; i<last_kp.size(); i++)
      last_kp[i].pt = cv::Point(last_keypoints.row(i)[0]*last_img.cols, last_keypoints.row(i)[1]*last_img.rows);

  std::vector<cv::KeyPoint> next_kp(next_keypoints.rows());
  for(size_t i=0; i<next_kp.size(); i++)
      next_kp[i].pt = cv::Point(next_keypoints.row(i)[0]*last_img.cols, next_keypoints.row(i)[1]*last_img.rows);

  cv::Mat img_matches;
  // use given current/next or empty image
  const cv::Mat current = next_img.empty() ? cv::Mat(last_img.size(), CV_8UC1, cv::Scalar(255)): next_img;
  cv::drawMatches(current, next_kp, last_img, last_kp, matches, img_matches,
                  cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  return img_matches;
}

int main(int argc, char *argv[]) {
  if (argc<4)
    return EXIT_FAILURE;

  // read command line arguments: <model_path> <image_1_path> <image_2_path>
  const std::string keypoint_predictor_path = argv[1];

  // load trained model
  const std::shared_ptr<FeatureMatchesInterface> sp = std::make_shared<SuperPoint>(keypoint_predictor_path);

  // load images
  std::vector<cv::Mat> img;
  for(int i=0; i<argc-2; i++) {
    img.push_back(cv::imread(argv[2+i]));
  }

  std::cout << "images: " << img.size() << std::endl;

  std::vector<Eigen::MatrixX2d> coord(img.size());
  std::vector<Eigen::MatrixXd> descr(img.size());
  std::vector<cv::Mat> feat(img.size());
  for(int i=0; i<img.size(); i++) {
    // convert to RGB for inference
    cv::Mat rgb;
    cv::cvtColor(img[i], rgb, cv::COLOR_BGR2RGB);
    std::tie(feat[i], coord[i], descr[i]) = sp->getFeatures(rgb);
    if (i>0) {
      const auto matches = pairwise_matches(descr[i-1].cast<float>(), descr[i].cast<float>());
      const cv::Mat img_matches = draw_matches(coord[i-1].cast<float>(), coord[i].cast<float>(), matches, img[i-1], img[i]);
      cv::imshow("matches "+std::to_string(i-1)+"-"+std::to_string(i), img_matches);
    }
  }
  cv::waitKey();

  return EXIT_SUCCESS;
}
