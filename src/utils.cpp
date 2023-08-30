/*
Licensed under the MIT License < http://opensource.org/licenses/MIT>.
SPDX - License - Identifier : MIT
Copyright(c) 2023 Mohammed Yasin
Copyright(c) 2023 Wahyu Setianto

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


#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#endif

#include "utils.hpp"

std::string BCODE[] = {"\033[94m", "\033[93m", "\033[91m", "\033[0m", "\033[1m"};

enum BCOLORS
{
    OKBLUE,
    WARNING,
    FAIL,
    ENDC,
    BOLD
};

std::string LogInfo(std::string header, std::string body)
{
    return BCODE[BOLD] + BCODE[OKBLUE] + header + ": " + BCODE[ENDC] + body;
}

std::string LogWarning(std::string header, std::string body)
{
    return "⚠️ " + BCODE[BOLD] + BCODE[WARNING] + header + ": " + BCODE[ENDC] + body;
}

std::string LogError(std::string header, std::string body)
{
    return "❌ " + BCODE[BOLD] + BCODE[FAIL] + header + ": " + BCODE[ENDC] + body;
}

void exists(std::string path)
{
    std::filesystem::path filePath(path);
    if (!std::filesystem::exists(filePath))
    {
        std::cerr << LogError("File Not Found", path) << std::endl;
        std::abort();
    }
}

#ifdef _WIN32
std::wstring convertToWChar(const char* str) {
    // Convert char* to wstring using MultiByteToWideChar
    int wstrSize = MultiByteToWideChar(CP_UTF8, 0, str, -1, nullptr, 0);
    if (wstrSize == 0) {
        std::cerr << "Failed to get wstring size" << std::endl;
    }

    std::wstring wstr(wstrSize, L'\0');
    if (MultiByteToWideChar(CP_UTF8, 0, str, -1, &wstr[0], wstrSize) == 0) {
        std::cerr << "Failed to convert char* to wstring" << std::endl;
    }

    return wstr;
}
#endif

std::vector<std::string> retrieve_labels(std::string labels_path) {

    std::ifstream file(labels_path);

    if (file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;

        std::vector<std::string> lines;
        std::string line;

        while (std::getline(file, line)) {
            lines.push_back(line);
        }

        file.close();

        return lines;
    }
    else
    {
        std::cerr << "Failed to open label file. Using COCO labels." << std::endl;
        return COCO_LABELS;
    }
}