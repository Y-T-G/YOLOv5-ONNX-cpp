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

#include <vector>
#include <algorithm>

struct Box {
    float x1, y1, x2, y2, confidence, class_id;
};

class Postprocessor {
public:
    Postprocessor(float score_threshold, float nms_threshold, int nms_top_k, int max_predictions, bool multi_label_per_box = true);
    std::vector<std::vector<Box>> forward(float* pred_bboxes, float* pred_scores, std::vector<int> output_shape_bboxes, std::vector<int> output_shape_scores);

private:
    std::vector<std::vector<Box>> _filter_max_predictions(std::vector<std::vector<Box>>& res) const;
    std::vector<size_t> performNMS(const std::vector<Box>& boxes, const std::vector<float>& scores, const std::vector<size_t>& indices, float iou_threshold) const;
    float calculateIntersection(const Box& box1, const Box& box2) const;
    float calculateArea(const Box& box) const;

    float score_threshold;
    float nms_threshold;
    int nms_top_k;
    int max_predictions;
    bool multi_label_per_box;
};