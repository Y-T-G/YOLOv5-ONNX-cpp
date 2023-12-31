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

#include <iostream>

#include "cli.hpp"
#include "yolo-v5.hpp"

#include <chrono>

int predictImage(YOLOv5& model, Args args) {

	cv::Mat img = cv::imread(args.source);

	model.predict_and_draw(img, args.labels);

	if (args.save) {
		cv::imwrite("output.jpg", img);
	}
	else {
		cv::imshow(args.source, img);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	return 0;
}

int predictVideo(YOLOv5& model, Args args) {

	cv::VideoCapture cap = cv::VideoCapture(args.source);
	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;
	float latency;

	cv::VideoWriter writer;

	if (args.save) {
		auto frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
		auto frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		auto fps = cap.get(cv::CAP_PROP_FPS);
		writer = cv::VideoWriter("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height), true);
	}

	while (cap.isOpened()) {
		cv::Mat frame;

		cap >> frame;

		if (!frame.empty()) {
			begin = std::chrono::steady_clock::now();
			model.predict_and_draw(frame, args.labels);
			end = std::chrono::steady_clock::now();
			if (args.save) {
				writer.write(frame);
			}
			else {
				cv::imshow(args.source, frame);

				if (cv::waitKey(30) == 27)
				{
					break;
				}
			}

			latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
			std::cout << "Latency = " << latency << "ms\t";
			std::cout << "FPS = " << 1000.0 / latency << std::endl;

		}

		else {
			break;
		}
	}

	cap.release();
	if (args.save) {
		writer.release();
		cv::destroyAllWindows();
	}

	return 0;
}

int main(int argc, char** argv)
{

	Args args = parseArgs(argc, argv);

	YOLOv5 model(args.modelPath, args.imgSize, args.gpu, args.scoreThresh, args.iouThresh);

	cv::Mat img = cv::imread(args.source);

	if (args.type == IMAGE) {
		predictImage(model, args);
	}

	else  if (args.type == VIDEO) {
		predictVideo(model, args);
	}

	return 0;
}