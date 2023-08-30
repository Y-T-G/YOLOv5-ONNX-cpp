
![GitHub release (latest by
date)](https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Visual
Studio](https://img.shields.io/badge/Visual%20Studio-5C2D91.svg?style=for-the-badge&logo=visual-studio&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white)
[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE)

# YOLOv5-ONNX.cpp

YOLOv5 is a popular detector by [Ultralytics](https://github.com/ultralytics/yolov5). This
project implements the YOLOv5 object detector in C++ utilizing the
ONNX Runtime to speed up inference performance.

## Features

* Supports both **image** and **video** inference.
* **Faster** CPU inference speeds.
* Supports **dynamic** input sizes.

## Getting Started

The following instructions demonstrates how to build this
project on a Linux system. Windows is currently not supported by the DeepSparse library.

### Prerequisites

* **CMake v3.8+** - found at
[https://cmake.org/](https://cmake.org/)

* **MSVC 2017++ (Windows Build) or GCC/G++ compiler (Linux Build)**

* **Python 3.8+** - Python is used to install the deepsparse library which is required for the build. Download [here](https://www.python.org/downloads/).

* **OpenCV v4.0+** - Download
[here](https://github.com/opencv/opencv/releases/).

* **ONNX Runtime v1.5.1+** - Download [here](https://github.com/microsoft/onnxruntime/releases).

## Building the project

1. Set the `OpenCV_DIR` environment variable to point to
your `../../opencv/build` directory (if not set).
2. Set the `ONNX_INCLUDE_DIR` to point the ONNX runtime header filesdirectory.
3. Set the `ONNX_LIB_DIR` to point the ONNX runtime lib directory.
4. Run the following build commands:
    a. [Windows] VS Developer Command Prompt:

    ```bash
    cd \d <yolo-nas-openvino-cpp-directory>
    cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release
    cd build

    MSBuild yolo-nas-openvino-cpp.sln /property:Configuration=Release
    ```

    b. [Linux] Bash:

    ```bash
    cd <yolov5-onnx-cpp-directory>
    cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release
    cd build

    make
    ```

3. The compiled `.exe` will be inside the `Release` folder for Windows build, while the executable will be in root folder for Linux build.

## Inference

1. Clone the [YOLOv5 repo](https://github.com/ultralytics/yolov5) and install the dependencies.
1. Export the ONNX file:

    ```bash
    python export.py --weights=yolov5n.onnx --include onnx --dynamic
    ```

2. To run the inference, execute the following command:

    ```bash
    yolo-nas-deepsparse-cpp --model <ONNX_MODEL_PATH> [-i <IMAGE_PATH> | -v <VIDEO_PATH>] [--labels <LABEL_PATH>] [--imgsz IMAGE_SIZE] [--iou-thresh IOU_THRESHOLD] [--score-thresh CONFIDENCE_THRESHOLD]
    ```

## Authors

* **Mohammed Yasin** - [@Y-T-G](https://github.com/Y-T-G)

## Acknowledgements

Thanks to [@Hyuto](https://github.com/Hyuto) for his work on
[ONNX
implementation](https://github.com/Hyuto/yolo-nas-onnx/tree/master/yolo-nas-cpp) of
YOLO-NAS in C++ which was utilized in this project.

## License

This project is licensed under the
[MIT](https://mit-license.org/) License - see the
[LICENSE](LICENSE) file for details. DeepSparse Community edition is only for evaluation, research, and non-production. See the [DeepSparse Community License](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE-NEURALMAGIC) for more details.
