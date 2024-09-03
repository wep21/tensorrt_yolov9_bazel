#include "engine.hpp"

#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include <fstream>

namespace tensorrt {
Engine::~Engine() {
  delete (context_);
  delete (engine_);
  CHECK_CUDA_ERROR(cudaEventDestroy(start_));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop_));
  return;
}

Engine::Engine(std::string plan) {
  std::fstream cache(plan, std::ifstream::in);
  CHECK_CUDA_ERROR(cudaEventCreate(&start_));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop_));
  if (cache.is_open()) {
    std::cout << "load plan file." << std::endl;
    char* data;
    unsigned int length;

    // get length of file:
    cache.seekg(0, cache.end);
    length = cache.tellg();
    cache.seekg(0, cache.beg);

    data = (char*)malloc(length);
    if (data == NULL) {
      std::cout << "Can't malloc data.\n";
      exit(-1);
    }

    cache.read(data, length);
    // create context
    auto runtime = nvinfer1::createInferRuntime(logger_);
    initLibNvInferPlugins(&logger_, "");

    if (runtime == nullptr) {
      std::cout << "load TRT cache0." << std::endl;
      std::cerr << ": runtime null!" << std::endl;
      exit(-1);
    }
    engine_ = (runtime->deserializeCudaEngine(data, length));
    if (engine_ == nullptr) {
      std::cerr << ": engine null!" << std::endl;
      exit(-1);
    }
    free(data);
    cache.close();
  } else {
    std::cerr << plan << " is not open" << std::endl;
    exit(-1);
  }

  context_ = engine_->createExecutionContext();
  return;
}

bool Engine::infer(cudaStream_t stream) {
  auto status = context_->enqueueV3(stream);

  return status;
}

bool Engine::set_tensor_address(const char* name, void* data) {
  auto status = context_->setTensorAddress(name, data);
  return status;
}

bool Engine::set_input_shape(const char* name, nvinfer1::Dims const& dims) {
  auto status = context_->setInputShape(name, dims);
  return status;
}
}  // namespace tensorrt
