#include <super_point_inference.hpp>
#include <torch/torch.h>
#include <torch/script.h>

//#define BENCHMARK

#ifdef BENCHMARK
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <iostream>
#endif

#include <opencv2/opencv.hpp>

#ifdef BENCHMARK
auto sync() {
    cudaDeviceSynchronize();
    return std::chrono::high_resolution_clock::now();
}
#endif

struct SuperPointImpl
{
    const torch::Device device;
    torch::jit::script::Module module;
};

SuperPoint::SuperPoint(const std::string &model_path)
    : impl(new SuperPointImpl({.device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU})),
      valid(!model_path.empty()),
      point_min_score(0.015)
{
    // load model
    if(valid) {
        try {
            impl->module = torch::jit::load(model_path, impl->device);
        } catch (const c10::Error &e) {
            std::cerr << "invalid model file: " << e.what() << std::endl;
        }
    }
#ifdef BENCHMARK
    at::globalContext().setBenchmarkCuDNN(true);
#endif
}

SuperPoint::~SuperPoint()
{
    delete impl;
}

std::tuple<Eigen::MatrixX2d, Eigen::MatrixXd>
SuperPoint::getFeatures(const cv::Mat &image) const
{
    if(!valid) {
        // return empty matrices if we do not have a model loaded
        return std::make_tuple<Eigen::MatrixX2d, Eigen::MatrixXd>({}, {});
    }

#ifdef BENCHMARK
    const auto tinput = sync();
#endif

    // convert RGB to grey-scale image in [0,1]
    cv::Mat img;
    cv::cvtColor(image, img, cv::COLOR_RGB2GRAY);
    img.convertTo(img, CV_32F);
    img /= 255;

    // check that image dimension is dividable by cell
    static constexpr int cell = 8;
    if (img.rows%cell!=0 || img.cols%cell!=0)
        throw std::runtime_error("image dimensions ("+std::to_string(img.cols)+" x "+std::to_string(img.rows)+") must be multiple of cell size ("+std::to_string(cell)+")");

    // convert image to tensor
    const torch::Tensor input = torch::from_blob(img.data, {1, 1, img.rows, img.cols}).to(impl->device);

    // inference on single image
#ifdef BENCHMARK
    const auto tinf = sync();
#endif
    const auto out = impl->module.forward({input}).toTuple();

    torch::Tensor tensor_semi = out->elements()[0].toTensor(); // heatmap
    torch::Tensor tensor_feat = out->elements()[1].toTensor(); // descriptors

#ifdef BENCHMARK
    const auto thm = sync();
#endif

    // softmax
    tensor_semi = torch::nn::functional::softmax(tensor_semi.squeeze(0), 0).permute({1, 2, 0});

    // reshape to original input image dimension
    const torch::Tensor heatmap = tensor_semi
            .slice(2, 0, -1)                                        // remove dust bin
            .reshape({img.rows/cell, img.cols/cell, cell, cell})
            .permute({0, 2, 1, 3})
            .reshape({img.rows, img.cols});

    // suppress non-maxima in local 3x3 neighbourhood
    // TODO: make 'kernel_size' of 3 configurable
    const torch::Tensor heatmap_nms = torch::nn::functional::max_pool2d(
                heatmap.unsqueeze(0).unsqueeze(1),
                torch::nn::MaxPool2dOptions({3,3}).stride({1,1}).padding(1)
                ).squeeze(0).squeeze(0);

    const torch::Tensor mask_nms = torch::logical_and((heatmap>point_min_score), (heatmap==heatmap_nms));

#ifdef BENCHMARK
    const auto tkp = sync();
#endif

    // point coordinates in normalised [0,1] image space (x,y)
    const torch::Tensor pts_norm = torch::roll(torch::nonzero(mask_nms).to(torch::kDouble) / torch::tensor(input.sizes().slice(2,2), impl->device), 1, 1);

    // centred point coordinates normalised in [-1,+1]
    const torch::Tensor pts_norm_c = (pts_norm*2-1).to(torch::kFloat);

    // get descriptors at keypoint coordinates
    torch::Tensor desc = torch::nn::functional::grid_sample(
                tensor_feat, pts_norm_c.unsqueeze(0).unsqueeze(0),
                torch::nn::functional::GridSampleFuncOptions().align_corners(false)
                ).squeeze(0).squeeze(1).transpose(1, 0).to(torch::kDouble).cpu();

    assert(pts_norm.sizes()[0]==desc.sizes()[0]);

    // normalise
    desc = desc / torch::norm(desc, 2, 1, true);

#ifdef BENCHMARK
    const auto tformat = sync();
#endif

    // N x 2 keypoint coordinates {(x_0, y_0), ..., (x_N, y_N)}
    const Eigen::MatrixX2d coord = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>(pts_norm.cpu().contiguous().data_ptr<double>(), pts_norm.sizes()[0], pts_norm.sizes()[1]);

    // N x D keypoint descriptors {f_0, ..., f_n} with f as 1 x D row-vector
    const Eigen::MatrixXd descr = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(desc.contiguous().data_ptr<double>(), desc.sizes()[0], desc.sizes()[1]);

#ifdef BENCHMARK
    const auto tend = sync();

    std::cout << "input load (ms): " << std::chrono::duration<float, std::milli>(tinf - tinput).count() << std::endl;
    std::cout << "inference (ms): " << std::chrono::duration<float, std::milli>(thm - tinf).count() << std::endl;
    std::cout << "heatmap (ms): " << std::chrono::duration<float, std::milli>(tkp - thm).count() << std::endl;
    std::cout << "keypoint (ms): " << std::chrono::duration<float, std::milli>(tformat - tkp).count() << std::endl;
    std::cout << "format (ms): " << std::chrono::duration<float, std::milli>(tend - tformat).count() << std::endl;
    std::cout << "TOTAL (ms): " << std::chrono::duration<float, std::milli>(tend - tinput).count() << std::endl;
#endif

    return {coord, descr};
}

void
SuperPoint::setScoreMin(float min_score)
{
    this->point_min_score = min_score;
}
