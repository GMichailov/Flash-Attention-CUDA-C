#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_runtime_api.h>

using pipe_t = cuda::pipeline<cuda::thread_scope_block>;