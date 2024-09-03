#include <iostream>

#include "NvInfer.h"

namespace {
#define CHECK_CUDA_ERROR(result) \
  { CheckCudaErrors(result, __FILE__, __LINE__); }
inline void CheckCudaErrors(cudaError_t result, const char* filename, int line_number) {
  if (result != cudaSuccess) {
    std::cout << "CUDA Error: " + std::string(cudaGetErrorString(result)) +
                   " (error code: " + std::to_string(result) + ") at " + std::string(filename) +
                   " in line " + std::to_string(line_number)
              << std::endl;
  }
}
}  // namespace

namespace tensorrt {

class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR ||
        severity == Severity::kINFO) {
      std::cerr << "tensorrt: " << msg << std::endl;
    }
  }
};

class Engine {
private:
  cudaEvent_t start_, stop_;

  Logger logger_;
  nvinfer1::IExecutionContext* context_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;

public:
  Engine(std::string plan);
  ~Engine();

  bool infer(cudaStream_t stream);
  bool set_tensor_address(const char* name, void* data);
  bool set_input_shape(const char* name, nvinfer1::Dims const& dims);
};
}  // namespace tensorrt
