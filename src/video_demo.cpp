#include <cassert>
#include <string>

#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpResize.hpp>
#include <opencv2/opencv.hpp>

#include "engine.hpp"

int main(int argc, char* argv[]) {
  assert(argc == 2 || argc == 3);
  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
  auto engine = std::make_unique<tensorrt::Engine>(std::string(argv[1]));
  nvinfer1::Dims dims;
  dims.nbDims = 4;
  dims.d[0] = 1;
  dims.d[1] = 3;
  dims.d[2] = 640;
  dims.d[3] = 640;
  engine->set_input_shape("images", dims);
  cv::VideoCapture cap;
  if (argc == 2) {
    cap.open(0);
  } else {
    cap.open(argv[2]);
  }

  if (!cap.isOpened()) {
    return -1;
  }

  cv::Mat frame;
  cap.read(frame);
  nvcv::TensorDataStridedCuda::Buffer input_buffer;
  input_buffer.strides[3] = sizeof(uint8_t);
  input_buffer.strides[2] = frame.channels() * input_buffer.strides[3];
  input_buffer.strides[1] = frame.cols * input_buffer.strides[2];
  input_buffer.strides[0] = frame.rows * input_buffer.strides[1];
  CHECK_CUDA_ERROR(cudaMalloc(&input_buffer.basePtr, input_buffer.strides[0]));

  nvcv::Tensor::Requirements input_reqs =
    nvcv::Tensor::CalcRequirements(1, {frame.cols, frame.rows}, nvcv::FMT_BGR8);

  nvcv::TensorDataStridedCuda input_data(
    nvcv::TensorShape{input_reqs.shape, input_reqs.rank, input_reqs.layout},
    nvcv::DataType{input_reqs.dtype}, input_buffer);

  nvcv::Tensor input_tensor = nvcv::TensorWrapData(input_data);

  nvcv::Tensor::Requirements input_layer_reqs =
    nvcv::Tensor::CalcRequirements(1, {640, 640}, nvcv::FMT_RGBf32p);

  int64_t input_layer_size = CalcTotalSizeBytes(nvcv::Requirements{input_layer_reqs.mem}.cudaMem());
  nvcv::TensorDataStridedCuda::Buffer input_layer_buffer;
  std::copy(input_layer_reqs.strides, input_layer_reqs.strides + NVCV_TENSOR_MAX_RANK,
            input_layer_buffer.strides);

  CHECK_CUDA_ERROR(cudaMalloc(&input_layer_buffer.basePtr, input_layer_size));

  nvcv::TensorDataStridedCuda input_layer_data(
    nvcv::TensorShape{input_layer_reqs.shape, input_layer_reqs.rank, input_layer_reqs.layout},
    nvcv::DataType{input_layer_reqs.dtype}, input_layer_buffer);
  nvcv::Tensor input_layer_tensor = TensorWrapData(input_layer_data);
  engine->set_tensor_address(engine->get_io_tensor_name(0), input_layer_buffer.basePtr);

  int* num_detections;
  float* boxes;
  float* scores;
  int* classes;
  float* outputs;
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&num_detections, sizeof(int)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&boxes, 1 * 100 * 4 * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&scores, 1 * 100 * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&classes, 1 * 100 * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&outputs, 1 * 8400 * 85 * sizeof(float)));
  engine->set_tensor_address("num_detections", num_detections);
  engine->set_tensor_address("detection_boxes", boxes);
  engine->set_tensor_address("detection_scores", scores);
  engine->set_tensor_address("detection_classes", classes);
  cvcuda::CvtColor cvt_color_op;
  cvcuda::Resize resize_op;
  cvcuda::ConvertTo convert_op;
  cvcuda::Reformat reformat_op;

  nvcv::Tensor rgb_tensor(1, {frame.cols, frame.rows}, nvcv::FMT_RGB8);
  nvcv::Tensor resized_tensor(1, {640, 640}, nvcv::FMT_RGB8);
  nvcv::Tensor float_tensor(1, {640, 640}, nvcv::FMT_RGBf32);

  // Warm up
  {
    cvt_color_op(stream, input_tensor, rgb_tensor, NVCV_COLOR_BGR2RGB);
    resize_op(stream, rgb_tensor, resized_tensor, NVCV_INTERP_LINEAR);
    convert_op(stream, resized_tensor, float_tensor, 1.0f / 255.0f, 0.0f);
    reformat_op(stream, float_tensor, input_layer_tensor);
    engine->infer(stream);
  }

  while (cap.read(frame)) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(input_buffer.basePtr, frame.data, input_buffer.strides[0],
                                     cudaMemcpyHostToDevice, stream));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cvt_color_op(stream, input_tensor, rgb_tensor, NVCV_COLOR_BGR2RGB);
    resize_op(stream, rgb_tensor, resized_tensor, NVCV_INTERP_LINEAR);
    convert_op(stream, resized_tensor, float_tensor, 1.0f / 255.0f, 0.0f);
    reformat_op(stream, float_tensor, input_layer_tensor);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float operatorms = 0;
    cudaEventElapsedTime(&operatorms, start, stop);
    std::cout << "Time for Preprocess : " << operatorms << " ms" << std::endl;
    cudaEventRecord(start);
    engine->infer(stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    operatorms = 0;
    cudaEventElapsedTime(&operatorms, start, stop);
    std::cout << "Time for Infer : " << operatorms << " ms" << std::endl;
    cudaStreamSynchronize(stream);
    for (int i = 0; i < num_detections[0]; ++i) {
      cv::rectangle(
        frame, cv::Point(boxes[4 * i] / 640 * frame.cols, boxes[4 * i + 1] / 640 * frame.rows),
        cv::Point(boxes[4 * i + 2] / 640 * frame.cols, boxes[4 * i + 3] / 640 * frame.rows),
        cv::Scalar(0, 0, 255), 1, 8, 0);
    }
    cv::imshow("win", frame);
    const int key = cv::waitKey(1);
    if (key == 'q') {
      break;
    } else if (key == 's') {
      cv::imwrite("img.png", frame);
    }
  }
  cv::destroyAllWindows();
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
  return 0;
}
