#include <super_pixel_inference.hpp>
#include <torch/torch.h>

SuperPixel::SuperPixel() {
    //
}

std::tuple<cv::Mat, std::vector<KeyPoint>>
&SuperPixel::getScore(const cv::Mat &colour, const cv::Mat &depth) {
    //
}
