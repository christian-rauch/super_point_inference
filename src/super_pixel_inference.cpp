#include <super_pixel_inference.hpp>
#include <torch/torch.h>
#include <torch/script.h>

//#define BENCHMARK

#ifdef BENCHMARK
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <iostream>
#endif

#include <opencv2/opencv.hpp>

void convertND(const torch::Tensor &tensor, cv::Mat &img) {
    assert(tensor.dtype()==torch::kFloat32);
    const auto sizes = tensor.sizes();
    img.create(sizes[0], sizes[1], CV_32FC(sizes[2]));
    std::memcpy(img.data, tensor.data_ptr<float>(), sizeof(float)*tensor.numel());
}

auto sync() {
#ifdef BENCHMARK
    cudaDeviceSynchronize();
#endif
    return std::chrono::high_resolution_clock::now();
}

struct SuperPixelImpl
{
    const torch::Device device;
    torch::jit::script::Module module;
};

SuperPixel::SuperPixel(const std::string &model_path)
    : impl(new SuperPixelImpl({.device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU})),
      valid(!model_path.empty())
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

SuperPixel::~SuperPixel()
{
    delete impl;
}

std::tuple<cv::Mat, Eigen::MatrixX2d, Eigen::MatrixXd>
SuperPixel::getFeatures(const cv::Mat &image) const
{
    if(!valid) {
        // return empty matrices if we do not have a model loaded
        return std::make_tuple<cv::Mat, Eigen::MatrixX2d, Eigen::MatrixXd>({}, {}, {});
    }

    const auto tinput = sync();

    // convert RGB to grey-scale image in [0,1]
    cv::Mat img;
    cv::cvtColor(image, img, cv::COLOR_RGB2GRAY);
    img.convertTo(img, CV_32F);
    img /= 255;

    // convert image to tensor
    const torch::Tensor input = torch::from_blob(img.data, {1, 1, img.rows, img.cols}).to(impl->device);

    // inference on single image
    const auto tinf = sync();
    const auto out = impl->module.forward({input}).toTuple();

    torch::Tensor tensor_semi = out->elements()[0].toTensor(); // heatmap
    torch::Tensor tensor_feat = out->elements()[1].toTensor(); // descriptors

    const auto thm = sync();

    // softmax
    tensor_semi = torch::nn::functional::softmax(tensor_semi.squeeze(0), 0).permute({1, 2, 0});

    // reshape to original input image dimension
    static const int cell = 8;
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

    // TODO: make 'conf_thresh' 0.015 configurable
    const torch::Tensor mask_nms = torch::logical_and((heatmap>0.015), (heatmap==heatmap_nms));

    const auto tkp = sync();

    // point coordinates in image space
    const torch::Tensor pts = torch::nonzero(mask_nms);

    // point coordinates in normalised [0,1] space
    const torch::Tensor pts_norm = torch::true_divide(pts.to(torch::kDouble), torch::tensor(input.sizes().slice(2,2), impl->device));

    // centred point coordinates normalised in [-1,+1]
    const torch::Tensor pts_norm_c = (pts_norm*2-1).to(torch::kFloat);

    // get descriptors at keypoint coordinates
    torch::Tensor desc = torch::nn::functional::grid_sample(
                tensor_feat, pts_norm_c.unsqueeze(0).unsqueeze(0),
                torch::nn::functional::GridSampleFuncOptions().align_corners(false)
                ).squeeze(0).squeeze(1).transpose(1, 0).to(torch::kDouble).cpu();

    assert(pts_norm.sizes()[0]==desc.sizes()[0]);

    const auto tformat = sync();

    // N x 2 keypoint coordinates {(x_0, y_0), ..., (x_N, y_N)}
    // swap (y,x) -> (x,y)
    const auto yx = torch::stack({pts_norm.slice(1,1,2), pts_norm.slice(1,0,1)}).transpose(1,0).to(torch::kDouble).cpu();
    const Eigen::MatrixX2d coord = Eigen::Map<Eigen::MatrixX2d>(yx.data_ptr<double>(), yx.sizes()[0], yx.sizes()[1]);

    // N x D keypoint descriptors {f_0, ..., f_n} with f as 1 x D row-vector
    const Eigen::MatrixXd descr = Eigen::Map<Eigen::MatrixXd>(desc.data_ptr<double>(), desc.sizes()[0], desc.sizes()[1]);

    // featuremap with dense descriptors
    cv::Mat feat;
    convertND(tensor_feat.squeeze(0).permute({1, 2, 0}).cpu(), feat);

    const auto tend = sync();

#ifdef BENCHMARK
    std::cout << "input load (ms): " << std::chrono::duration<float, std::milli>(tinf - tinput).count() << std::endl;
    std::cout << "inference (ms): " << std::chrono::duration<float, std::milli>(thm - tinf).count() << std::endl;
    std::cout << "heatmap (ms): " << std::chrono::duration<float, std::milli>(tkp - thm).count() << std::endl;
    std::cout << "keypoint (ms): " << std::chrono::duration<float, std::milli>(tformat - tkp).count() << std::endl;
    std::cout << "format (ms): " << std::chrono::duration<float, std::milli>(tend - tformat).count() << std::endl;
    std::cout << "TOTAL (ms): " << std::chrono::duration<float, std::milli>(tend - tinput).count() << std::endl;
#endif

    return {feat, coord, descr};
}
