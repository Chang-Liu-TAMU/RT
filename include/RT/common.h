#pragma once
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>
#include <algorithm>
#include <map>
#include <utility>
#include <functional>
#include <string>
#include <exception>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define CLIP(X, LEFT, RIGHT) (X < LEFT? LEFT: X > RIGHT? RIGHT: X)
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

inline void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}
inline void checkGlError(const char* _s) {
    auto error = glGetError();
    while (error != GL_NO_ERROR) {
        std::cout << "GL Error: " << error << " " << _s << std::endl;
        error = glGetError();
    }
}