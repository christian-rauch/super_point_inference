#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <super_point_inference.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;


std::vector<std::tuple<int, int, float>>
pairwise_matches(const Eigen::MatrixXf &last_keypoints,
                 const Eigen::MatrixXf &next_keypoints)
{
  // store correspondences (last_id, next_id)
  std::vector<std::tuple<int, int, float>> match_ids;

  if (last_keypoints.rows()>0) {
    cv::Mat last_descr;
    cv::eigen2cv(Eigen::MatrixXf(last_keypoints.rightCols(last_keypoints.cols()-2)), last_descr);
    cv::Mat next_descr;
    cv::eigen2cv(Eigen::MatrixXf(next_keypoints.rightCols(next_keypoints.cols()-2)), next_descr);

    // match descriptor pairs (query=next, train=last)
    std::vector<cv::DMatch> matches;
    cv::BFMatcher(cv::NORM_L2, true).match(next_descr, last_descr, matches);

    for(const cv::DMatch &match : matches) {
      if (match.distance>0.7)
        continue;
      match_ids.push_back(std::make_tuple(match.trainIdx, match.queryIdx, match.distance));
    }
  }

  return match_ids;
}

cv::Mat
draw_matches(const Eigen::MatrixXf &last_keypoints,
             const Eigen::MatrixXf &next_keypoints,
             const std::vector<std::tuple<int, int, float>> &correspondences,
             const cv::Mat &last_img, const cv::Mat &next_img)
{
  std::vector<cv::DMatch> matches;
  for(const auto &match : correspondences)
    matches.emplace_back(std::get<1>(match), std::get<0>(match), std::get<2>(match));

  std::vector<cv::KeyPoint> last_kp(last_keypoints.rows());
  for(size_t i=0; i<last_kp.size(); i++)
      last_kp[i].pt = cv::Point(last_keypoints.row(i)[0]*last_img.cols, last_keypoints.row(i)[1]*last_img.rows);

  std::vector<cv::KeyPoint> next_kp(next_keypoints.rows());
  for(size_t i=0; i<next_kp.size(); i++)
      next_kp[i].pt = cv::Point(next_keypoints.row(i)[0]*next_img.cols, next_keypoints.row(i)[1]*next_img.rows);

  cv::Mat img_matches;
  cv::drawMatches(next_img, next_kp, last_img, last_kp, matches, img_matches,
                  cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  return img_matches;
}

int main(int argc, char *argv[]) {
  if (argc<4) {
    std::cerr << "wrong set of arguments, expected:" << std::endl << argv[0] << " <model_path> <image_1_path> ... <image_N_path>" << std::endl;
    return EXIT_FAILURE;
  }

  // read command line arguments: <model_path> <image_1_path> ... <image_N_path>
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
      // export matches
      const std::string match_name = "matches "+std::to_string(i-1)+"-"+std::to_string(i);
      cv::imshow(match_name, img_matches);
      cv::imwrite(fs::temp_directory_path() / fs::path(match_name+".png"), img_matches);
    }
  }
  cv::waitKey();

  return EXIT_SUCCESS;
}
