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

#include <onnxruntime_cxx_api.h>

#include "processing.hpp"
#include "yolo-v5.hpp"
#include "utils.hpp"
#include "draw.hpp"


YOLOv5::YOLOv5(std::string modelPath, std::vector<int> imgsz, bool gpu, float score, float iou)
    : postprocessor(score, iou, 1000, 300, false) // define postprocessor
{
    size_t batchSize = 1;

    // Setup ORT environment
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeSession");
    session_options = Ort::SessionOptions();
    allocator = Ort::AllocatorWithDefaultOptions();
    memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

#ifdef _WIN32

    // Convert char* to wstring
    std::wstring w_modelPath = convertToWChar(modelPath.c_str());

    // Load model
    session = Ort::Session(env, w_modelPath.c_str(), session_options);

#else

    // Load model
    session = Ort::Session(env, modelPath.c_str(), session_options);

#endif // _WIN32

    // Input tensor info
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    inputDims = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

    //Output tensor info
    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    outputDims = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

    // Check if dynamic dims
    isDynamic = false;
    if (inputDims[2] == -1 && inputDims[3] == -1)
    {
        std::cout << "Dynamic input shape" << std::endl;
        isDynamic = true;
        inputDims[0] = batchSize;
    }

    // Set model input shape
    modelInputShape = { 1, 3, imgsz[1], imgsz[0] };

    scoreTresh = score;
    iouTresh = iou;

}

void YOLOv5::letterbox(cv::Mat& source, cv::Mat& dst, std::vector<float>& ratios,
    const cv::Scalar& color = cv::Scalar(114, 114, 114))
{

    //Calculate padding
    int maxSize = std::max(source.cols, source.rows);

    int xPad = maxSize - source.cols;
    int yPad = maxSize - source.rows;

    cv::copyMakeBorder(source, dst, 0, yPad, 0, xPad, cv::BORDER_CONSTANT, color); // padding

    // Ensure that the resized dimensions are divisible by 32
    int maxInputSize = std::max(modelInputShape[2], modelInputShape[3]);
    int targetSize = ((maxInputSize - 1) / 32 + 1) * 32;

    inputDims[2] = targetSize;
    inputDims[3] = targetSize;

    float xRatio = (float)maxSize / (float)targetSize;
    float yRatio = (float)maxSize / (float)targetSize;

    cv::resize(dst, dst, cv::Size(targetSize, targetSize), 0, 0, cv::INTER_LINEAR);

    ratios.push_back(xRatio);
    ratios.push_back(yRatio);
}


cv::Mat YOLOv5::pre_process_onnx(cv::Mat& img, cv::Mat& blob, std::vector<float>& ratios)
{
    //Convert from BGR to RGB
    cv::cvtColor(img, blob, cv::COLOR_BGR2RGB);

    // Letterbox resize
    letterbox(blob, blob, ratios);

    //Normalize
    blob.convertTo(blob, CV_32F, 1.0f / 255.0f);

    // hwc to chw
    cv::dnn::blobFromImage(blob, blob);

    return blob;
}

void YOLOv5::post_process_onnx(std::vector<Ort::Value>& outputTensors, std::vector<float>& bboxes, std::vector<int>& bboxes_shape, std::vector<float>& scores, std::vector<int>& scores_shape)
{

    const float* output = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    int num_preds = static_cast<int>(outputShape[1]);
    int bbox_size = 5; // Number of elements in each bounding box (x, y, width, height, score)
    int num_classes = static_cast<int>(outputShape[2] - bbox_size);

    // Extract bboxes and scores
    bboxes = std::vector<float>(num_preds * bbox_size);
    bboxes_shape = std::vector<int>{num_preds, bbox_size};

    scores = std::vector<float>(num_preds * (outputShape[2] - bbox_size));
    scores_shape = std::vector<int>{num_preds, num_classes};

    for (int i = 0; i < num_preds; ++i) {

        for (int j = 0; j < bbox_size; ++j) {
            bboxes[i * bbox_size + j] = output[i * outputShape[2] + j];
        }

        for (int j = bbox_size; j < outputShape[2]; ++j) {
            scores[i * (outputShape[2] - bbox_size) + (j - bbox_size)] = output[i * outputShape[2] + j];
        }
    }

}

void YOLOv5::predict_and_draw(cv::Mat& img, std::vector<std::string> labels)
{
    try
    {
        cv::Mat blob;
        std::vector<float> ratios;
        
        //Preprocess image
        cv::Mat input_img = pre_process_onnx(img, blob, ratios);

        // Set batch size to 1
        inputDims[0] = 1;

        // Create input and output tensors
        size_t inputTensorSize = input_img.total();
        std::vector<float> inputTensorValues(inputTensorSize);
        inputTensorValues.assign(input_img.begin<float>(), input_img.end<float>());

        std::vector<Ort::Value> inputTensors;
        std::vector<Ort::Value> outputTensors;

        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
            inputDims.size()));

        std::string input_name = "images";
        std::string output_name = "output0";

        inputNames.push_back(input_name.c_str());
        outputNames.push_back(output_name.c_str());

        // Inference
        outputTensors = session.Run(Ort::RunOptions{ nullptr },inputNames.data(),
            inputTensors.data(), 1, outputNames.data(), 1);

        // Split output into bboxes and scores for further processing
        std::vector<float> bboxes;
        std::vector<int> bboxes_shape;
        std::vector<float> scores;
        std::vector<int> scores_shape;
        post_process_onnx(outputTensors, bboxes, bboxes_shape, scores, scores_shape);

        // Extract bboxes, apply NMS
        std::vector<std::vector<Box>> results = postprocessor.forward(bboxes.data(), scores.data(), bboxes_shape, scores_shape);

        // Visualize
        drawBoxes(img, results, ratios[0], ratios[1], labels);

    }
    catch (std::exception& ex)
    {
        LogError("Inference Error: ", ex.what());
        throw ex;
    }
    catch (...)
    {
        LogError("Inference Error: ", "Unexpected exception");
        throw;
    }
}

