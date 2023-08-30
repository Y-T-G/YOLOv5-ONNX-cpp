cmake_minimum_required (VERSION 3.12)

project ("yolo-v5-onnx-cpp")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ONNX_INCLUDE_DIRS "Path to built ONNX include header directory." STRING)
message(STATUS "ONNX_INCLUDE_DIRS: ${ONNX_INCLUDE_DIRS}")

option(ONNX_LIB_DIRS "Path to built ONNX runtime lib directory." STRING)
message(STATUS "ONNX_LIB_DIRS: ${ONNX_LIB_DIRS}")

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v2.9
)
FetchContent_MakeAvailable(argparse)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNX_INCLUDE_DIRS})
link_directories(${ONNX_LIB_DIRS})

file(GLOB SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} argparse)
target_link_libraries(${PROJECT_NAME} onnxruntime)
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include")

# Windows might have an onnxruntime.dll in the system directory
# Manually copy actual dll to local directory to avoid conflict
function(copy_ort_dlls PROJECT_NAME)
    if(MSVC)
        file(GLOB ORT_DLLS ${ONNX_LIB_DIRS}/*.dll)
        foreach(ORT_DLL ${ORT_DLLS})
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${ORT_DLL} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
        endforeach()
    endif()
endfunction()

copy_ort_dlls(${PROJECT_NAME})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
