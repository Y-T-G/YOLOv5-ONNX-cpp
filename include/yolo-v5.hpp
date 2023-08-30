/*
Licensed under the MIT License < http://opensource.org/licenses/MIT>.
SPDX - License - Identifier : MIT
Copyright(c) 2023 Mohammed Yasin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <numeric>

#include "processing.hpp"

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}


class YOLOv5
{
private:
    // Needs to be member for working correctly
    Ort::Session session{nullptr};
    Ort::Env env{nullptr};
    Ort::SessionOptions session_options{nullptr};
    Ort::AllocatorWithDefaultOptions allocator{nullptr};
    Ort::MemoryInfo memoryInfo{nullptr};

    std::vector<int> modelInputShape;

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;

    float scoreTresh;
    float iouTresh;

    bool isDynamic;

    std::vector<int64_t> inputDims;
    std::vector<int64_t> outputDims;

    std::vector<int> imgSize;
    cv::Mat pre_process_onnx(cv::Mat& img, cv::Mat& blob, std::vector<float>& ratios);
    void letterbox(cv::Mat& source, cv::Mat& dst, std::vector<float>& ratios, const cv::Scalar& color);
    void YOLOv5::post_process_onnx(std::vector<Ort::Value>& outputTensors, std::vector<float>& bboxes, std::vector<int>& bboxes_shape, std::vector<float>& scores, std::vector<int>& scores_shape);

    Postprocessor postprocessor;

public:
    YOLOv5(std::string model_path, std::vector<int> imgsz, bool cuda, float scoreTresh, float iouTresh);
    void predict_and_draw(cv::Mat& img, std::vector<std::string> labels);
};